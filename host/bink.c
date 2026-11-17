/* BinkW32.DLL: the game's movies (Ubisoft logos, loading loops, cinematics), decoded with FFmpeg's Bink demuxer and
 * decoders (third_party/ffmpeg-android-arm64: libavformat/libavcodec/libavutil built with only bink, binkaudio_dct,
 * binkaudio_rdft and the file protocol; LGPL).
 *
 * Everything runs on the calling (game) thread: BinkWait/BinkDoFrame read packets ahead into a small queue, audio
 * packets are decoded on the spot and pushed as float PCM into a host stream of the DirectSound mixer (dsound.c),
 * which resamples it to the output rate. The movie's clock is that stream's audible position (wall clock when the
 * movie has no audio, or the stream runs dry), so video stays locked to the sound.
 *
 * HBINK (guest memory, the fields the game reads): +0 Width, +4 Height, +8 Frames, +12 FrameNum (1-based, the frame
 * BinkDoFrame decodes next), +16 LastFrameNum, +20 FrameRate, +24 FrameRateDiv, +32 OpenFlags, +44 FrameSize. */
#include "win32.h"

#include <pthread.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/log.h>

#define W(name) void bink_##name(CPU *c)

int dsound_hs_open(uint32_t rate);
void dsound_hs_close(int id);
void dsound_hs_flush(int id);
void dsound_hs_pause(int id, int p);
void dsound_hs_volume(int id, float v);
uint32_t dsound_hs_write(int id, const float *l, const float *r, uint32_t n);
uint32_t dsound_hs_queued(int id);
double dsound_hs_clock(int id);

#define BINKFILEHANDLE 0x00800000u
#define BINKSNDTRACK   0x00004000u
#define VQ_MAX 120          /* video packets read ahead (4 s at 30 fps) while filling the audio queue */
#define MAX_TRACKS 8

typedef struct {
    int stream, track_id, hs;
    AVCodecContext *dec;
    uint32_t rate;
    uint64_t lost;           /* frames that did not fit the mixer's ring */
} Track;

typedef struct Movie {
    uint32_t g;              /* guest HBINK */
    char name[256];
    AVFormatContext *fmt;
    int vs;                  /* video stream index */
    AVCodecContext *vdec;
    AVFrame *cur, *tmp;
    AVPacket *pkt;
    AVPacket *vq[VQ_MAX];
    int vq_head, vq_n, eof;
    Track tr[MAX_TRACKS];
    int ntr;
    uint32_t frames, rate, div, w, h;
    uint32_t demuxed_v;      /* video packets read since the last rewind */
    uint32_t decoded;        /* frame number in `cur` (0: none) */
    uint32_t skip_audio_below;   /* after BinkGoto: audio of the frames before the target is dropped */
    /* clock */
    int started, paused;
    double t0, pause_at, base;   /* wall time of frame 1, when paused, movie time at the last rewind/goto */
    double last_a, last_aw;
    /* stats */
    int wrapped;             /* BinkNextFrame went past the last frame: rewind at the next BinkDoFrame */
    int skipping;            /* the player pressed Skip: report the last frame, decode nothing more */
    uint32_t shown, copies, late, ndec;
    double t_first;
    double dec_time, copy_time;
    uint32_t logged_fmt;
    struct Movie *next;
} Movie;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static Movie *g_movies;
static uint32_t g_sel_tracks[MAX_TRACKS], g_nsel = 0;
static int g_ff_init;

static Movie *find(uint32_t g) { for (Movie *m = g_movies; m; m = m->next) if (m->g == g) return m; return NULL; }

static void ff_log(void *avcl, int level, const char *fmt, va_list vl) {
    (void)avcl;
    if (level > AV_LOG_WARNING || !g_log) return;
    fprintf(stderr, "[bink] ffmpeg: ");
    vfprintf(stderr, fmt, vl);
}

/* ------------------------------------------------------------------ clock */

static double wall(Movie *m) {
    if (!m->started) return 0;
    return (m->paused ? m->pause_at : host_seconds()) - m->t0;
}
/* seconds into the movie that are audible / due now */
static double movie_time(Movie *m) {
    double w = wall(m);
    if (m->ntr && m->tr[0].hs >= 0 && m->started) {
        int hs = m->tr[0].hs;
        double a = dsound_hs_clock(hs);
        if (a > m->last_a || dsound_hs_queued(hs) > 0) { m->last_a = a; m->last_aw = w; return m->base + a; }
        return m->base + m->last_a + (w - m->last_aw);   /* stream ran dry (or ended): carry on with the wall clock */
    }
    return m->base + w;
}
static double frame_time(Movie *m, uint32_t f) { return f ? (double)(f - 1) * m->div / m->rate : 0; }

/* ------------------------------------------------------------------ demux / decode */

static void push_audio(Movie *m, Track *t, AVFrame *f) {
    int n = f->nb_samples;
    if (n <= 0) return;
    const float *l = (const float *)f->extended_data[0];
    const float *r = f->ch_layout.nb_channels > 1 ? (const float *)f->extended_data[1] : l;
    if (f->format != AV_SAMPLE_FMT_FLTP) {
        static int once; if (!once++) fprintf(stderr, "[bink] unexpected audio sample format %d\n", f->format);
        return;
    }
    uint32_t put = dsound_hs_write(t->hs, l, r, (uint32_t)n);
    if (put < (uint32_t)n) {
        if (!t->lost) fprintf(stderr, "[bink] %s: audio ring full, dropping %u frames\n", m->name, (uint32_t)n - put);
        t->lost += (uint32_t)n - put;
    }
}

static void decode_audio(Movie *m, Track *t, AVPacket *p) {
    if (avcodec_send_packet(t->dec, p) < 0) return;
    while (avcodec_receive_frame(t->dec, m->tmp) == 0) {
        push_audio(m, t, m->tmp);
        av_frame_unref(m->tmp);
    }
}

/* reads one packet; returns 0 at the end of the file */
static int read_one(Movie *m) {
    if (m->eof) return 0;
    if (av_read_frame(m->fmt, m->pkt) < 0) { m->eof = 1; return 0; }
    AVPacket *p = m->pkt;
    if (p->stream_index == m->vs) {
        if (m->vq_n < VQ_MAX) {
            m->vq[(m->vq_head + m->vq_n) % VQ_MAX] = av_packet_clone(p);
            m->vq_n++;
        }
        m->demuxed_v++;
    } else {
        for (int i = 0; i < m->ntr; i++)
            if (m->tr[i].stream == p->stream_index) {
                if (m->demuxed_v + 1 >= m->skip_audio_below) decode_audio(m, &m->tr[i], p);
                break;
            }
    }
    av_packet_unref(p);
    return 1;
}

/* keep a video packet queued and ~0.5 s of audio ahead of the speakers */
static void pump(Movie *m) {
    while (!m->eof && m->vq_n < VQ_MAX) {
        int audio_ok = 1;
        for (int i = 0; i < m->ntr; i++)
            if (m->tr[i].hs >= 0 && dsound_hs_queued(m->tr[i].hs) < m->tr[i].rate / 2) audio_ok = 0;
        if (m->vq_n > 0 && audio_ok) break;
        if (!read_one(m)) break;
    }
}

static void rewind_movie(Movie *m) {
    av_seek_frame(m->fmt, m->vs, 0, AVSEEK_FLAG_BACKWARD);   /* the bink demuxer always goes back to frame 1 */
    avcodec_flush_buffers(m->vdec);
    for (int i = 0; i < m->ntr; i++) { avcodec_flush_buffers(m->tr[i].dec); dsound_hs_flush(m->tr[i].hs); }
    while (m->vq_n) { av_packet_free(&m->vq[m->vq_head]); m->vq_head = (m->vq_head + 1) % VQ_MAX; m->vq_n--; }
    m->vq_head = 0; m->eof = 0; m->demuxed_v = 0; m->decoded = 0;
    av_frame_unref(m->cur);
}

/* decodes up to frame `f` (1-based) */
static void decode_to(Movie *m, uint32_t f) {
    if (f < 1) f = 1;
    if (f > m->frames) f = m->frames;
    if (f < m->decoded) rewind_movie(m);
    double t = host_seconds();
    while (m->decoded < f) {
        pump(m);
        if (!m->vq_n) break;
        AVPacket *p = m->vq[m->vq_head];
        m->vq_head = (m->vq_head + 1) % VQ_MAX; m->vq_n--;
        int r = avcodec_send_packet(m->vdec, p);
        av_packet_free(&p);
        m->decoded++; m->ndec++;
        if (r < 0) continue;
        if (avcodec_receive_frame(m->vdec, m->tmp) == 0) { av_frame_unref(m->cur); av_frame_move_ref(m->cur, m->tmp); }
    }
    m->dec_time += host_seconds() - t;
}

/* ------------------------------------------------------------------ open / close */

static void free_movie(Movie *m) {
    for (int i = 0; i < m->ntr; i++) { dsound_hs_close(m->tr[i].hs); avcodec_free_context(&m->tr[i].dec); }
    while (m->vq_n) { av_packet_free(&m->vq[m->vq_head]); m->vq_head = (m->vq_head + 1) % VQ_MAX; m->vq_n--; }
    avcodec_free_context(&m->vdec);
    av_frame_free(&m->cur); av_frame_free(&m->tmp); av_packet_free(&m->pkt);
    if (m->fmt) avformat_close_input(&m->fmt);
    free(m);
}

static AVCodecContext *open_dec(AVStream *st) {
    const AVCodec *cd = avcodec_find_decoder(st->codecpar->codec_id);
    if (!cd) return NULL;
    AVCodecContext *d = avcodec_alloc_context3(cd);
    if (!d) return NULL;
    avcodec_parameters_to_context(d, st->codecpar);
    d->thread_count = 1;
    if (avcodec_open2(d, cd, NULL) < 0) { avcodec_free_context(&d); return NULL; }
    return d;
}

static Movie *open_movie(const char *host, const char *name, uint32_t flags) {
    Movie *m = calloc(1, sizeof *m);
    snprintf(m->name, sizeof m->name, "%s", name);
    m->vs = -1;
    if (avformat_open_input(&m->fmt, host, av_find_input_format("bink"), NULL) < 0) {
        fprintf(stderr, "[bink] can't open %s (%s)\n", name, host);
        free(m);
        return NULL;
    }
    for (unsigned i = 0; i < m->fmt->nb_streams; i++)
        if (m->fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) { m->vs = (int)i; break; }
    if (m->vs < 0 || !(m->vdec = open_dec(m->fmt->streams[m->vs]))) {
        fprintf(stderr, "[bink] %s: no usable video stream\n", name);
        avformat_close_input(&m->fmt); free(m);
        return NULL;
    }
    AVStream *vst = m->fmt->streams[m->vs];
    m->w = (uint32_t)vst->codecpar->width; m->h = (uint32_t)vst->codecpar->height;
    m->frames = vst->duration > 0 ? (uint32_t)vst->duration : (uint32_t)vst->nb_frames;
    m->rate = (uint32_t)vst->time_base.den; m->div = (uint32_t)vst->time_base.num;
    if (!m->rate || !m->div) { m->rate = 30; m->div = 1; }
    /* audio: the tracks BinkSetSoundTrack chose (with BINKSNDTRACK), otherwise track 0 */
    uint32_t want[MAX_TRACKS], nwant = 0;
    if ((flags & BINKSNDTRACK) && g_nsel) { for (uint32_t i = 0; i < g_nsel; i++) want[nwant++] = g_sel_tracks[i]; }
    else want[nwant++] = 0;
    int nat = 0;
    for (unsigned i = 0; i < m->fmt->nb_streams; i++) {
        AVStream *st = m->fmt->streams[i];
        if (st->codecpar->codec_type != AVMEDIA_TYPE_AUDIO) continue;
        nat++;
        int sel = 0;
        for (uint32_t k = 0; k < nwant; k++) if ((uint32_t)st->id == want[k]) sel = 1;
        if (!sel || m->ntr >= MAX_TRACKS) { st->discard = AVDISCARD_ALL; continue; }
        Track *t = &m->tr[m->ntr];
        t->stream = (int)i; t->track_id = st->id; t->rate = (uint32_t)st->codecpar->sample_rate;
        if (!(t->dec = open_dec(st))) { st->discard = AVDISCARD_ALL; continue; }
        t->hs = dsound_hs_open(t->rate);
        if (t->hs < 0) { avcodec_free_context(&t->dec); st->discard = AVDISCARD_ALL; continue; }
        dsound_hs_pause(t->hs, 1);    /* sound starts with the first frame */
        m->ntr++;
    }
    m->cur = av_frame_alloc(); m->tmp = av_frame_alloc(); m->pkt = av_packet_alloc();
    m->skip_audio_below = 0;
    fprintf(stderr, "[bink] open %s: %ux%u, %u frames at %u/%u fps, %d audio track(s), playing %d (flags %08x)\n",
            name, m->w, m->h, m->frames, m->rate, m->div, nat, m->ntr, flags);
    return m;
}

/* ------------------------------------------------------------------ imports */

W(BinkOpen) {
    uint32_t name = ARG(0), flags = ARG(1);
    pthread_mutex_lock(&g_lock);
    if (!g_ff_init) { g_ff_init = 1; av_log_set_callback(ff_log); }
    char host[1024], gname[512];
    int ok = 0;
    if (flags & BINKFILEHANDLE) {   /* name is a Win32 file handle */
        WinHandle *h = handle_get(name, H_FILE);
        if (h) { snprintf(host, sizeof host, "%s", h->path); snprintf(gname, sizeof gname, "handle %x", name); ok = 1; }
    } else if (name) {
        snprintf(gname, sizeof gname, "%s", GSTR(name));
        ok = map_path(gname, host, sizeof host);
    }
    Movie *m = ok ? open_movie(host, gname, flags) : NULL;
    if (!m) {
        if (!ok) fprintf(stderr, "[bink] BinkOpen(%s): file not found\n", name && !(flags & BINKFILEHANDLE) ? GSTR(name) : "?");
        pthread_mutex_unlock(&g_lock);
        RET(2, 0);
    }
    uint32_t b = heap_alloc(1024, 1);
    m->g = b;
    W32(b, m->w); W32(b + 4, m->h); W32(b + 8, m->frames); W32(b + 12, 1); W32(b + 16, 0);
    W32(b + 20, m->rate); W32(b + 24, m->div); W32(b + 32, flags); W32(b + 44, m->w * m->h);
    m->next = g_movies; g_movies = m;
    pthread_mutex_unlock(&g_lock);
    RET(2, b);
}

W(BinkClose) {
    uint32_t b = ARG(0);
    pthread_mutex_lock(&g_lock);
    for (Movie **pp = &g_movies; *pp; pp = &(*pp)->next)
        if ((*pp)->g == b) {
            Movie *m = *pp; *pp = m->next;
            fprintf(stderr, "[bink] close %s: %u frames shown (%u late), %u copies, decode %.2f ms/frame, copy %.2f ms, %.1f s\n",
                    m->name, m->shown, m->late, m->copies, m->ndec ? m->dec_time * 1e3 / m->ndec : 0,
                    m->copies ? m->copy_time * 1e3 / m->copies : 0, m->started ? host_seconds() - m->t_first : 0);
            if (m->ntr) fprintf(stderr, "[bink]   audio: %.2f s played (base %.2f), %u frames queued, %llu dropped, video clock %.2f s at frame %u\n",
                                dsound_hs_clock(m->tr[0].hs), m->base, dsound_hs_queued(m->tr[0].hs), (unsigned long long)m->tr[0].lost,
                                movie_time(m), m->decoded);
            free_movie(m);
            heap_free(b);
            break;
        }
    pthread_mutex_unlock(&g_lock);
    RET(1, 0);
}

static void start_clock(Movie *m) {
    if (m->started) return;
    m->started = 1;
    m->t0 = m->t_first = host_seconds();
    for (int i = 0; i < m->ntr; i++) dsound_hs_pause(m->tr[i].hs, m->paused);
    if (m->paused) m->pause_at = m->t0;
}

W(BinkDoFrame) {
    pthread_mutex_lock(&g_lock);
    Movie *m = find(ARG(0));
    if (m) {
        uint32_t f = R32(m->g + 12);
        start_clock(m);
        if (m->wrapped) {   /* looping: back to frame 1, sound and clock included */
            m->wrapped = 0;
            rewind_movie(m);
            m->base = 0; m->last_a = 0; m->last_aw = 0;
            m->t0 = host_seconds(); if (m->paused) m->pause_at = m->t0;
        }
        if (!m->skipping) decode_to(m, f);
        m->shown++;
        if (movie_time(m) > frame_time(m, f) + 0.1) m->late++;
    }
    pthread_mutex_unlock(&g_lock);
    RET(1, 0);
}

W(BinkNextFrame) {
    pthread_mutex_lock(&g_lock);
    Movie *m = find(ARG(0));
    if (m) {
        uint32_t f = R32(m->g + 12);
        W32(m->g + 16, f);
        if (m->skipping) { W32(m->g + 16, m->frames); W32(m->g + 12, m->frames);   /* the game ends a movie when LastFrameNum == Frames */ }
        else if (f >= m->frames) { m->wrapped = 1; W32(m->g + 12, 1); }   /* Bink loops: past the last frame comes frame 1 */
        else W32(m->g + 12, f + 1);
    }
    pthread_mutex_unlock(&g_lock);
    RET(1, 0);
}

/* 1 while it is too early for the next frame; 0 when it is due */
W(BinkWait) {
    pthread_mutex_lock(&g_lock);
    Movie *m = find(ARG(0));
    int r = 0;
    if (m) {
        if (!m->started || m->skipping) { if (!m->skipping) pump(m); pthread_mutex_unlock(&g_lock); RET(1, 0); }
        uint32_t f = R32(m->g + 12);
        if (m->wrapped) f = m->frames + 1;   /* frame 1 of the next loop comes one frame after the last */
        else if (f == m->decoded) f++;       /* the current frame is already out: wait for the next one */
        pump(m);
        r = !m->paused ? movie_time(m) < frame_time(m, f) : 1;
    }
    pthread_mutex_unlock(&g_lock);
    RET(1, r);
}

W(BinkGoto) {
    pthread_mutex_lock(&g_lock);
    Movie *m = find(ARG(0));
    uint32_t f = ARG(1);
    if (m) {
        fprintf(stderr, "[bink] goto %s frame %u (flags %x)\n", m->name, f, ARG(2));
        if (f < 1) f = 1;
        if (f > m->frames) f = m->frames;
        rewind_movie(m);
        m->wrapped = 0;
        m->skip_audio_below = f;
        decode_to(m, f);
        m->base = frame_time(m, f); m->last_a = 0; m->last_aw = 0;
        if (m->started) { m->t0 = host_seconds(); if (m->paused) m->pause_at = m->t0; }
        W32(m->g + 12, f); W32(m->g + 16, f);
    }
    pthread_mutex_unlock(&g_lock);
    RET(3, 0);
}

W(BinkPause) {
    pthread_mutex_lock(&g_lock);
    Movie *m = find(ARG(0));
    int p = ARG(1) != 0, r = 0;
    if (m) {
        if (p != m->paused && m->started) {
            double now = host_seconds();
            if (p) m->pause_at = now; else m->t0 += now - m->pause_at;
        }
        m->paused = p;
        if (m->started) for (int i = 0; i < m->ntr; i++) dsound_hs_pause(m->tr[i].hs, p);
        r = p;
    }
    pthread_mutex_unlock(&g_lock);
    RET(2, r);
}

/* volume 0..65536, 32768 = normal */
W(BinkSetVolume) {
    pthread_mutex_lock(&g_lock);
    Movie *m = find(ARG(0));
    uint32_t id = ARG(1);
    float v = (float)(int32_t)ARG(2) / 32768.f;
    if (v < 0) v = 0;
    if (m) {
        int hit = 0;
        for (int i = 0; i < m->ntr; i++) if ((uint32_t)m->tr[i].track_id == id) { dsound_hs_volume(m->tr[i].hs, v); hit = 1; }
        if (!hit) for (int i = 0; i < m->ntr; i++) dsound_hs_volume(m->tr[i].hs, v);
        static int n; if (n++ < 8) LOG("[bink] SetVolume(%s, track %u, %d)\n", m->name, id, (int32_t)ARG(2));
    }
    pthread_mutex_unlock(&g_lock);
    RET(3, 0);
}

W(BinkSetSoundTrack) {
    uint32_t n = ARG(0), p = ARG(1);
    pthread_mutex_lock(&g_lock);
    g_nsel = n > MAX_TRACKS ? MAX_TRACKS : n;
    for (uint32_t i = 0; i < g_nsel; i++) g_sel_tracks[i] = p ? R32(p + 4 * i) : i;
    fprintf(stderr, "[bink] SetSoundTrack(%u):", n);
    for (uint32_t i = 0; i < g_nsel; i++) fprintf(stderr, " %u", g_sel_tracks[i]);
    fprintf(stderr, "\n");
    pthread_mutex_unlock(&g_lock);
    RET(2, 0);
}

W(BinkSetSoundSystem) { LOG("[bink] SetSoundSystem(%08x, %08x)\n", ARG(0), ARG(1)); RET(2, 1); }
W(BinkOpenDirectSound) { RET(1, 1); }

/* ------------------------------------------------------------------ BinkCopyToBuffer */

enum { S_8P, S_24, S_24R, S_32, S_32R, S_32A, S_32RA, S_4444, S_5551, S_555, S_565, S_655, S_664, S_YUY2, S_UYVY, S_YV12 };
static const int k_bpp[16] = { 1, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1 };

static int32_t t_y[256], t_rv[256], t_gu[256], t_gv[256], t_bu[256];
static void init_tables(void) {   /* BT.601, studio range (what FFmpeg reports for Bink: yuv420p(tv)) */
    static int done; if (done) return; done = 1;
    for (int i = 0; i < 256; i++) {
        t_y[i] = (int32_t)(1.164383 * (i - 16) * 65536 + 32768);
        t_rv[i] = (int32_t)(1.596027 * (i - 128) * 65536);
        t_gu[i] = (int32_t)(-0.391762 * (i - 128) * 65536);
        t_gv[i] = (int32_t)(-0.812968 * (i - 128) * 65536);
        t_bu[i] = (int32_t)(2.017232 * (i - 128) * 65536);
    }
}
static inline uint8_t clamp8(int32_t v) { v >>= 16; return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }

/* one source row to BGRA */
static void row_bgra(const AVFrame *f, int y, int w, uint8_t *o) {
    const uint8_t *Y = f->data[0] + (size_t)y * f->linesize[0];
    const uint8_t *U = f->data[1] + (size_t)(y >> 1) * f->linesize[1];
    const uint8_t *V = f->data[2] + (size_t)(y >> 1) * f->linesize[2];
    const uint8_t *A = f->format == AV_PIX_FMT_YUVA420P ? f->data[3] + (size_t)y * f->linesize[3] : NULL;
    for (int x = 0; x < w; x += 2) {
        int u = U[x >> 1], v = V[x >> 1];
        int32_t rv = t_rv[v], g = t_gu[u] + t_gv[v], bu = t_bu[u];
        for (int k = 0; k < 2 && x + k < w; k++) {
            int32_t yy = t_y[Y[x + k]];
            uint8_t *p = o + 4 * (x + k);
            p[0] = clamp8(yy + bu); p[1] = clamp8(yy + g); p[2] = clamp8(yy + rv); p[3] = A ? A[x + k] : 255;
        }
    }
}

static void pack_row(int fmt, const uint8_t *s, int w, int sx, uint8_t *d) {
    int ow = w * sx;
    for (int x = 0; x < ow; x++) {
        const uint8_t *p = s + 4 * (x / sx);
        unsigned b = p[0], g = p[1], r = p[2], a = p[3];
        uint16_t v;
        switch (fmt) {
        case S_32: case S_32A: d[0] = (uint8_t)b; d[1] = (uint8_t)g; d[2] = (uint8_t)r; d[3] = (uint8_t)a; d += 4; continue;
        case S_32R: case S_32RA: d[0] = (uint8_t)r; d[1] = (uint8_t)g; d[2] = (uint8_t)b; d[3] = (uint8_t)a; d += 4; continue;
        case S_24: d[0] = (uint8_t)b; d[1] = (uint8_t)g; d[2] = (uint8_t)r; d += 3; continue;
        case S_24R: d[0] = (uint8_t)r; d[1] = (uint8_t)g; d[2] = (uint8_t)b; d += 3; continue;
        case S_4444: v = (uint16_t)((a >> 4) << 12 | (r >> 4) << 8 | (g >> 4) << 4 | b >> 4); break;
        case S_5551: v = (uint16_t)((a >= 128) << 15 | (r >> 3) << 10 | (g >> 3) << 5 | b >> 3); break;
        case S_555: v = (uint16_t)((r >> 3) << 10 | (g >> 3) << 5 | b >> 3); break;
        case S_565: v = (uint16_t)((r >> 3) << 11 | (g >> 2) << 5 | b >> 3); break;
        case S_655: v = (uint16_t)((r >> 2) << 10 | (g >> 3) << 5 | b >> 3); break;
        case S_664: v = (uint16_t)((r >> 2) << 10 | (g >> 2) << 4 | b >> 4); break;
        default: return;
        }
        d[0] = (uint8_t)v; d[1] = (uint8_t)(v >> 8); d += 2;
    }
}

static void pack_yuv422(int fmt, const AVFrame *f, int y, int w, int sx, uint8_t *d) {
    const uint8_t *Y = f->data[0] + (size_t)y * f->linesize[0];
    const uint8_t *U = f->data[1] + (size_t)(y >> 1) * f->linesize[1];
    const uint8_t *V = f->data[2] + (size_t)(y >> 1) * f->linesize[2];
    int ow = w * sx;
    for (int x = 0; x + 1 < ow; x += 2, d += 4) {
        int x0 = x / sx, x1 = (x + 1) / sx;
        uint8_t u = U[x0 >> 1], v = V[x0 >> 1];
        if (fmt == S_YUY2) { d[0] = Y[x0]; d[1] = u; d[2] = Y[x1]; d[3] = v; }
        else { d[0] = u; d[1] = Y[x0]; d[2] = v; d[3] = Y[x1]; }
    }
}

/* BinkCopyToBuffer(hbink, dest, destpitch, destheight, destx, desty, flags) */
W(BinkCopyToBuffer) {
    uint32_t b = ARG(0), dest = ARG(1), pitch = ARG(2), dh = ARG(3), dx = ARG(4), dy = ARG(5), flags = ARG(6);
    pthread_mutex_lock(&g_lock);
    Movie *m = find(b);
    int ret = 0;
    if (m && dest && m->cur->data[0]) {
        double t = host_seconds();
        int fmt = (int)(flags & 15);
        uint32_t scale = flags & 0x70000000u;
        int sx = 1, sy = 1;
        if (scale == 0x30000000u) sx = 2;                                  /* BINKCOPY2XW */
        else if (scale == 0x10000000u || scale == 0x20000000u) sy = 2;     /* BINKCOPY2XH, 2XHI */
        else if (scale == 0x40000000u || scale == 0x50000000u) sx = sy = 2; /* BINKCOPY2XWH, 2XWHI */
        if (m->logged_fmt++ < 2 || (g_log && m->copies == 0))
            fprintf(stderr, "[bink] CopyToBuffer(%s): dest %08x pitch %u height %u at %u,%u flags %08x (format %d)\n",
                    m->name, dest, pitch, dh, dx, dy, flags, fmt);
        init_tables();
        const AVFrame *f = m->cur;
        int w = f->width, h = f->height, bpp = k_bpp[fmt];
        if ((uint32_t)w > m->w) w = (int)m->w;
        if (dx * (uint32_t)bpp + (uint32_t)(w * sx * bpp) > pitch) w = (int)((pitch / (uint32_t)bpp - dx) / (uint32_t)sx);
        int rows = h * sy;
        if (dh && (uint32_t)rows + dy > dh) rows = dh > dy ? (int)(dh - dy) : 0;
        uint8_t *rowbuf = NULL;
        if (fmt == S_8P || fmt == S_YV12) {
            static int once; if (!once++) fprintf(stderr, "[bink] surface format %d not supported\n", fmt);
            rows = 0;
        } else if (fmt != S_YUY2 && fmt != S_UYVY) rowbuf = malloc((size_t)w * 4);
        int last_src = -1;
        for (int oy = 0; oy < rows && w > 0; oy++) {
            int y = oy / sy;
            uint8_t *d = HP(dest + (dy + (uint32_t)oy) * pitch + dx * (uint32_t)bpp);
            if (fmt == S_YUY2 || fmt == S_UYVY) { pack_yuv422(fmt, f, y, w, sx, d); continue; }
            if ((fmt == S_32 || fmt == S_32A) && sx == 1) { row_bgra(f, y, w, d); continue; }   /* straight into the surface */
            if (y != last_src) { row_bgra(f, y, w, rowbuf); last_src = y; }
            pack_row(fmt, rowbuf, w, sx, d);
        }
        free(rowbuf);
        m->copies++;
        m->copy_time += host_seconds() - t;
    }
    pthread_mutex_unlock(&g_lock);
    RET(7, ret);
}

/* any movie open (logos, loading screens, cinematics): the Android touch overlay hides its game controls then
   (android/host/pop3_android_stubs.c gamephase_current) */
/* Skip (touch overlay): every open movie jumps to its last frame, so the game's own end-of-movie check ends it.
   Called from the UI thread. */
static volatile int g_skip;
void bink_skip(void) {
    pthread_mutex_lock(&g_lock);
    for (Movie *m = g_movies; m; m = m->next) {
        if (m->skipping) continue;
        m->skipping = 1;
        for (int i = 0; i < m->ntr; i++) dsound_hs_pause(m->tr[i].hs, 1);
        fprintf(stderr, "[bink] skip %s at frame %u of %u\n", m->name, R32(m->g + 12), m->frames);
    }
    pthread_mutex_unlock(&g_lock);
}

int bink_movie_open(void) {
    pthread_mutex_lock(&g_lock);
    int open = g_movies != NULL;
    pthread_mutex_unlock(&g_lock);
    return open;
}
