/* Render recorder: the game-thread half of the renderer (see render_cmd.h).
 *
 * Implements the Direct3D backend interface by doing all CPU work here - expanding primitives to
 * lists, running vertex shaders, converting changed textures to RGBA, building uniforms and pipeline
 * keys - and emitting self-contained commands for the executor (render_wgpu.c). Natively every
 * command runs immediately; in the browser commands are queued per frame for the page's thread. */
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif
#include "render.h"
#include "render_cmd.h"
#include "vshader.h"

#include <math.h>
#include <sys/stat.h>
#include <unistd.h>

int texture_to_rgba8(uint32_t fmt, const uint8_t *src, uint32_t pitch, uint32_t w, uint32_t h, uint8_t *dst,
                     const uint8_t *palette);
extern uint8_t g_palettes[][1024];

/* WGPUPrimitiveTopology values (kept numeric so this file doesn't need webgpu.h) */
enum { TOPO_POINTS = 1, TOPO_LINES = 2, TOPO_TRIANGLES = 4 };

/* ------------------------------------------------------------------ command buffer
 * One recorder for the whole process: Direct3D calls come from the game thread, but objects can be
 * released from others, so every entry point holds g_rec_lock (recursive). */

#include <pthread.h>
static pthread_mutex_t g_rec_lock;
static pthread_once_t g_rec_once = PTHREAD_ONCE_INIT;
static void rec_lock_init(void) {
    pthread_mutexattr_t a;
    pthread_mutexattr_init(&a);
    pthread_mutexattr_settype(&a, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&g_rec_lock, &a);
}
static void rec_enter(void) { pthread_once(&g_rec_once, rec_lock_init); pthread_mutex_lock(&g_rec_lock); }
static void rec_leave(void) { pthread_mutex_unlock(&g_rec_lock); }

static uint8_t *g_cb;
static size_t g_cb_len, g_cb_cap;

static void *cmd_alloc(uint32_t type, size_t size) {
    size = (size + 7) & ~(size_t)7;
    if (g_cb_len + size > g_cb_cap) {
        g_cb_cap = (g_cb_len + size) * 2 + (1u << 20);
        g_cb = realloc(g_cb, g_cb_cap);
    }
    CmdHdr *h = (CmdHdr *)(g_cb + g_cb_len);
    memset(h, 0, size);
    h->type = type;
    h->size = (uint32_t)size;
    g_cb_len += size;
    return h;
}

#ifdef __EMSCRIPTEN__
/* browser: completed command chunks go to the page's thread (web_gpu.c) */
void web_queue_chunk(uint8_t *buf, size_t len, int ends_frame);
int web_queued_frames(void);
int web_gpu_ready(void);

static void cmd_end(void) {}

static void cmd_flush(int ends_frame) {
    if (!g_cb_len && !ends_frame) return;
    web_queue_chunk(g_cb, g_cb_len, ends_frame);   /* takes ownership */
    g_cb = NULL; g_cb_len = g_cb_cap = 0;
}
#else
/* native: run each command as soon as it is recorded */
static void cmd_end(void) {
    size_t off = 0;
    while (off < g_cb_len) {
        const CmdHdr *h = (const CmdHdr *)(g_cb + off);
        gpu_exec(h);
        off += h->size;
    }
    g_cb_len = 0;
}
static void cmd_flush(int ends_frame) { (void)ends_frame; }
#endif

/* ------------------------------------------------------------------ debugging aids
 * RC_SCREENSHOTS="120,600" saves /tmp/rc_frame_N.png; `touch /tmp/rc_snap` or RC_CAPTURE=N logs
 * every draw of a frame and saves the back buffer after each one (native only). */

#include <fcntl.h>
#include <unistd.h>
static int g_snap_detail, g_snap_saved_fd = -1;
static float g_cam_c[12][4], g_cam_last[12][4];   /* c4-c7 of the frame's first shader draw (this frame / the last whole frame) */
static int g_cam_seen;
/* the camera's view-projection rows as the game passed them (c4..c7); 0 if none yet */
int camera_constants(float out[12][4]) { memcpy(out, g_cam_last, sizeof g_cam_last); return 1; }
static uint32_t g_f_drawn;
extern char g_game_root[4096];
double g_prof_draw, g_prof_present; uint64_t g_prof_verts, g_prof_draws_vs;   /* seconds spent recording draws / presenting (RC_LOG_FPS prints them per frame) */
static uint32_t g_trace_vs;

static int want_screenshot(uint32_t frame) {
    static uint32_t last_check;
    static int cap_frame = -1;
    if (cap_frame < 0) {
        cap_frame = getenv("RC_CAPTURE") ? atoi(getenv("RC_CAPTURE")) : 0;
        g_trace_vs = getenv("RC_TRACE_VS") ? (uint32_t)strtoul(getenv("RC_TRACE_VS"), NULL, 16) : 0;
    }
    if (cap_frame && frame + 1 == (uint32_t)cap_frame) { g_snap_detail = 1; return 0; }
    if (frame - last_check >= 15) {
        last_check = frame;
        char snap_p[4200];   /* debugging: <game>/../snap.txt (Android: files/snap.txt) asks for a detailed frame */
        snprintf(snap_p, sizeof snap_p, "%s/../snap.txt", g_game_root);
        if (!access("/tmp/rc_snap", F_OK) || !access(snap_p, F_OK)) {
            unlink("/tmp/rc_snap"); unlink(snap_p);
            g_snap_detail = 1;
            /* the frame's log goes to <game>/../snap_out.txt: too many lines for the Android log pipe */
            snprintf(snap_p, sizeof snap_p, "%s/../snap_out.txt", g_game_root);
            int fd = open(snap_p, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd >= 0) { fflush(stderr); g_snap_saved_fd = dup(2); dup2(fd, 2); close(fd); }
            return 0;
        }
    }
    if (g_snap_detail) {
        g_snap_detail = 0;
        fprintf(stderr, "[snap] frame %u: %u draws\n", frame, g_f_drawn);
        if (g_snap_saved_fd >= 0) { fflush(stderr); dup2(g_snap_saved_fd, 2); close(g_snap_saved_fd); g_snap_saved_fd = -1; }
        return 1;
    }
    static const char *list = (const char *)1;
    if (list == (const char *)1) list = getenv("RC_SCREENSHOTS");
    if (!list) return 0;
    char tmp[64];
    snprintf(tmp, sizeof tmp, "%u", frame);
    for (const char *p = list; *p;) {
        size_t n = strcspn(p, ",");
        if (n == strlen(tmp) && !strncmp(p, tmp, n)) return 1;
        p += n + (p[n] == ',');
    }
    return 0;
}

/* ------------------------------------------------------------------ render target tracking */

static CmdTarget g_last_target;   /* zero: back buffer, not yet sent */
static int g_target_sent;
static uint32_t g_tgt_lw, g_tgt_lh;

static void select_target(Device *d) {
    const DevState *st = &d->st;
    Surface *rt = st->render_target && st->render_target != d->backbuffer ? com_data(st->render_target) : NULL;
    Surface *ds = st->depth_stencil && st->depth_stencil != d->depth ? com_data(st->depth_stencil) : NULL;
    CmdTarget t;
    memset(&t, 0, sizeof t);
    if (rt) { t.color_id = rt->gpu_id; t.color_w = rt->width; t.color_h = rt->height; t.lw = rt->width; t.lh = rt->height; }
    else { t.lw = d->width; t.lh = d->height; }
    if (ds) { t.depth_id = ds->gpu_id; t.depth_w = ds->width; t.depth_h = ds->height; }
    g_tgt_lw = t.lw; g_tgt_lh = t.lh;
    if (g_target_sent && !memcmp(&t.color_id, &g_last_target.color_id, sizeof t - sizeof t.h)) return;
    if (g_snap_detail)
        fprintf(stderr, "[target] color %u (%ux%u, logical %ux%u) depth %u\n", t.color_id, t.color_w, t.color_h, t.lw, t.lh,
                t.depth_id);
    CmdTarget *c = cmd_alloc(CMD_TARGET, sizeof *c);
    memcpy(&c->color_id, &t.color_id, sizeof t - sizeof t.h);
    cmd_end();
    g_last_target = t;
    g_target_sent = 1;
}

/* ------------------------------------------------------------------ vertices */

typedef struct { uint32_t stride, pos_type, has_normal, has_psize, has_diffuse, has_specular, ntex,
                 off_normal, off_diffuse, off_specular, off_tex[8]; } FvfLayout;

static void fvf_layout(uint32_t fvf, FvfLayout *L) {
    memset(L, 0, sizeof *L);
    uint32_t off = 0;
    L->pos_type = fvf & 0xE;
    switch (L->pos_type) {
    case 0x2: off = 12; break;              /* XYZ */
    case 0x4: off = 16; break;              /* XYZRHW */
    case 0x6: off = 16; break; case 0x8: off = 20; break; case 0xA: off = 24; break;   /* XYZB1-3 */
    case 0xC: off = 28; break; case 0xE: off = 32; break;
    }
    if (fvf & 0x10) { L->has_normal = 1; L->off_normal = off; off += 12; }
    if (fvf & 0x20) { L->has_psize = 1; off += 4; }
    if (fvf & 0x40) { L->has_diffuse = 1; L->off_diffuse = off; off += 4; }
    if (fvf & 0x80) { L->has_specular = 1; L->off_specular = off; off += 4; }
    L->ntex = (fvf >> 8) & 0xF;
    for (uint32_t i = 0; i < L->ntex && i < 8; i++) {
        L->off_tex[i] = off;
        uint32_t f = (fvf >> (16 + 2 * i)) & 3;
        off += f == 0 ? 8 : f == 1 ? 12 : f == 2 ? 16 : 4;
    }
    L->stride = off;
}

static void convert_vertex(const uint8_t *v, const FvfLayout *L, CVert *o) {
    float f[4] = { 0, 0, 0, 1 };
    memcpy(f, v, L->pos_type == 0x4 ? 16 : 12);
    o->x = f[0]; o->y = f[1]; o->z = f[2]; o->w = L->pos_type == 0x4 ? f[3] : 1.f;
    o->diffuse = 0xFFFFFFFFu; o->specular = 0;
    if (L->has_diffuse) memcpy(&o->diffuse, v + L->off_diffuse, 4);
    if (L->has_specular) memcpy(&o->specular, v + L->off_specular, 4);
    memset(o->tc, 0, sizeof o->tc);
    for (uint32_t t = 0; t < 4; t++) {
        o->tc[t][3] = 1.f;
        if (t < L->ntex) memcpy(o->tc[t], v + L->off_tex[t], 8);
    }
}

static void mat_mul(const float *a, const float *b, float *r) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a[i * 4 + k] * b[k * 4 + j];
            r[i * 4 + j] = s;
        }
}

static uint32_t pack_color(const float *f) {   /* RGBA floats -> D3DCOLOR (ARGB) */
    uint32_t b[4];
    for (int i = 0; i < 4; i++) { float v = f[i] < 0 ? 0 : f[i] > 1 ? 1 : f[i]; b[i] = (uint32_t)(v * 255.f + 0.5f); }
    return b[2] | b[1] << 8 | b[0] << 16 | b[3] << 24;
}

static void unpack_color(uint32_t c, float *o) {
    o[0] = ((c >> 16) & 255) / 255.f; o[1] = ((c >> 8) & 255) / 255.f; o[2] = (c & 255) / 255.f; o[3] = (c >> 24) / 255.f;
}

/* ------------------------------------------------------------------ textures */

static uint32_t sampler_key(const uint32_t *tss) {
    uint32_t mag = tss[16] >= 2, min = tss[17] >= 2, mip = tss[18];
    uint32_t au = tss[13] & 7, av = tss[14] & 7;
    uint32_t bc = 0;
    if (au == 4 || av == 4) {   /* D3DTADDRESS_BORDER: nearest of the border colours Vulkan has (ARGB in tss[15]) */
        uint32_t c = tss[15], a = c >> 24, l = ((c >> 16 & 255) + (c >> 8 & 255) + (c & 255)) / 3;
        bc = a < 128 ? 0 : l >= 128 ? 2 : 1;
    }
    return mag | min << 1 | (mip & 3) << 2 | au << 4 | av << 7 | bc << 10;   /* < 4096 */
}

/* send a texture's contents if they changed since the last upload */
/* DXT textures go to the GPU as BC blocks when it can sample them: 4-8x less GPU memory than RGBA8 (a match
   needs hundreds of MB uncompressed, more than some GPUs/browsers give a page). WebGPU wants whole blocks,
   so the base level must be a multiple of 4; other textures are converted. */
static uint32_t bc_format(const Texture *t) {
    if (!format_is_dxt(t->format)) return TEXFMT_RGBA8;
    if (!g_gpu_bc || t->lw[0] % 4 || t->lh[0] % 4) {   /* no BC sampling: decoded on the GPU (the CPU took ~20 ms per batch on Mali) */
        static int cpu = -1;
        if (cpu < 0) cpu = getenv("RC_CPU_DXT") != NULL;
        if (cpu) return TEXFMT_RGBA8;
        return t->format == FMT_DXT1 ? TEXFMT_DXT1_DEC : t->format <= FMT_DXT3 ? TEXFMT_DXT3_DEC : TEXFMT_DXT5_DEC;
    }
    return t->format == FMT_DXT1 ? TEXFMT_BC1 : t->format <= FMT_DXT3 ? TEXFMT_BC2 : TEXFMT_BC3;
}

/* ------------------------------------------------------------------ texture fingerprints (UI draws)
 * Menus and the HUD show in which 2D textures the game draws (the menu cursor most of all). Each texture
 * gets a hash of its level 0 contents at upload; every 2D draw (pre-transformed FVF or the UI vertex shader
 * 0x1007d) notes its texture's hash and where its first vertex is. ui_frame_done() hands each frame's list to
 * the phase detector (host/gamephase.c); RC_LOG_UITEX=1 logs it every 2 s for finding new signals. */
static uint32_t tex_fingerprint(const Texture *t) {
    uint32_t h = 2166136261u ^ t->format ^ (t->lw[0] << 8) ^ (t->lh[0] << 20);
    uint32_t rows = t->lh[0], row = t->pitch[0];
    if (!t->mem[0][0] || !row) return 0;
    if (format_is_dxt(t->format)) rows = (t->lh[0] + 3) / 4;
    else if (row > t->lw[0] * 4) row = t->lw[0] * 4;
    for (uint32_t y = 0; y < rows; y += rows > 64 ? rows / 64 : 1) {   /* up to ~64 sampled rows */
        const uint8_t *p = HP(t->mem[0][0] + y * t->pitch[0]);
        for (uint32_t x = 0; x + 3 < row; x += 4) { h ^= p[x] | p[x + 1] << 8 | (uint32_t)p[x + 2] << 16 | (uint32_t)p[x + 3] << 24; h *= 16777619u; }
    }
    return h ? h : 1;
}
#define UI_TEX_MAX 128
typedef struct { uint32_t hash, wh; float x, y; } UiTex;
static UiTex g_ui[UI_TEX_MAX], g_ui_last[UI_TEX_MAX];
static uint32_t g_ui_n, g_ui_last_n;
static void ui_note(const Texture *t, float x, float y) {
    if (!t->content_hash) return;
    for (uint32_t i = 0; i < g_ui_n; i++) if (g_ui[i].hash == t->content_hash) return;
    if (g_ui_n < UI_TEX_MAX) { UiTex *u = &g_ui[g_ui_n++]; u->hash = t->content_hash; u->wh = t->lw[0] | t->lh[0] << 16; u->x = x; u->y = y; }
}
static void ui_frame_done(uint32_t frame) {
    if (g_cam_seen) memcpy(g_cam_last, g_cam_c, sizeof g_cam_c);
    g_cam_seen = 0;
    {
        static int log = -1;
        if (log < 0) log = getenv("RC_LOG_CAM") != NULL;
        if (log) {
            fprintf(stderr, "[cam] %u", frame);
            for (int r = 0; r < 12; r++) fprintf(stderr, " c%d %.3f %.3f %.3f %.2f", r, g_cam_last[r][0], g_cam_last[r][1], g_cam_last[r][2], g_cam_last[r][3]);
            fprintf(stderr, "\n");
        }
    }
    memcpy(g_ui_last, g_ui, g_ui_n * sizeof *g_ui);
    g_ui_last_n = g_ui_n;
    g_ui_n = 0;
    extern void gamephase_frame(const uint32_t *hashes, const float *xy, uint32_t n);
    uint32_t h[UI_TEX_MAX]; float xy[2 * UI_TEX_MAX];
    for (uint32_t i = 0; i < g_ui_last_n; i++) { h[i] = g_ui_last[i].hash; xy[2 * i] = g_ui_last[i].x; xy[2 * i + 1] = g_ui_last[i].y; }
    gamephase_frame(h, xy, g_ui_last_n);
    static int log = -1;
    static double last;
    if (log < 0) log = getenv("RC_LOG_UITEX") != NULL;
    if (log && host_seconds() - last >= 2.0) {
        last = host_seconds();
        fprintf(stderr, "[uitex] frame %u:", frame);
        for (uint32_t i = 0; i < g_ui_last_n; i++)
            fprintf(stderr, " %08x %ux%u@%.0f,%.0f", g_ui_last[i].hash, g_ui_last[i].wh & 0xffff, g_ui_last[i].wh >> 16, g_ui_last[i].x, g_ui_last[i].y);
        fprintf(stderr, "\n");
    }
}

static void sync_texture_impl(Device *d, Texture *t);
static void sync_texture(Device *d, Texture *t) {
    if (t->sent_version == t->version) return;
    extern double g_hx[]; extern uint32_t g_hxb[];
    double t0 = host_seconds();
    sync_texture_impl(d, t);
    g_hx[1] += host_seconds() - t0; g_hxb[1] += t->lw[0] * t->lh[0] * 4;
}
static void sync_texture_impl(Device *d, Texture *t) {
    t->content_hash = tex_fingerprint(t);
    uint32_t fmt = bc_format(t), bs = TEXFMT_BLOCK_BYTES(fmt);
    size_t total = 0;
    for (uint32_t l = 0; l < t->levels && l < 14; l++)
        total += fmt ? (size_t)((t->lw[l] + 3) / 4) * ((t->lh[l] + 3) / 4) * bs : (size_t)t->lw[l] * t->lh[l] * 4;
    CmdUploadTex *c = cmd_alloc(CMD_UPLOAD_TEX, sizeof *c + total);
    c->id = t->gpu_id;
    c->levels = t->levels < 14 ? t->levels : 14;
    c->fmt = fmt;
    if (fmt) {
        uint8_t *out = (uint8_t *)(c + 1);
        for (uint32_t l = 0; l < c->levels; l++) {
            uint32_t bw = (t->lw[l] + 3) / 4, bh = (t->lh[l] + 3) / 4;
            c->w[l] = t->lw[l]; c->h_[l] = t->lh[l];
            for (uint32_t by = 0; by < bh; by++) memcpy(out + (size_t)by * bw * bs, HP(t->mem[0][l] + by * t->pitch[l]), (size_t)bw * bs);
            out += (size_t)bw * bh * bs;
        }
        cmd_end();
        t->sent_version = t->version;
        return;
    }
    uint8_t *out = (uint8_t *)(c + 1);
    const uint8_t *pal = d->st.palette_current < 1024 ? g_palettes[d->st.palette_current] : NULL;
    for (uint32_t l = 0; l < c->levels; l++) {
        uint32_t w = t->lw[l], h = t->lh[l];
        c->w[l] = w; c->h_[l] = h;
        if (!texture_to_rgba8(t->format, HP(t->mem[0][l]), t->pitch[l], w, h, out, pal)) {
            static int warned;
            if (!warned++) fprintf(stderr, "[render] unsupported texture format %x\n", t->format);
            memset(out, 255, (size_t)w * h * 4);
        }
        out += (size_t)w * h * 4;
    }
    cmd_end();
    t->sent_version = t->version;
}

/* ------------------------------------------------------------------ cached vertex buffers
 * Most of a frame is drawn from vertex buffers that never change after they are filled. Their vertices are
 * converted once (CVert, or the input registers of a vertex shader run on the GPU) and kept on the GPU, keyed
 * by the buffers, their versions, offsets, strides and the layout (FVF or shader); draws then send only
 * indices. Buffers that change after being converted and DrawPrimitiveUP data keep the per-draw path
 * (D3DUSAGE_DYNAMIC says nothing here: churn is detected instead). Entries not used for a while, or least
 * recently used ones over the budget, are dropped.
 * RC_VB_CACHE=0 turns it off; RC_VB_CACHE_MB sets the budget (default 64); RC_VB_CACHE_AB=N alternates it
 * on and off every N frames (to compare screenshots of a paused game). */

#define VBC_MAX_STREAMS 4
#define VBC_CAP 4096
#define VBC_BUCKETS 1024
typedef struct { uint32_t layout, n; struct { uint32_t id, version, stride, offset; } s[VBC_MAX_STREAMS]; } VbKey;
typedef struct { VbKey k; uint32_t vb_id, bytes, last_frame; int32_t next; } VbEntry;
static VbEntry g_vbc[VBC_CAP];
static int32_t g_vbc_bucket[VBC_BUCKETS], g_vbc_free = -1, g_vbc_used;
static size_t g_vbc_bytes;
static uint32_t g_vbc_hits, g_vbc_misses, g_vbc_frame;
/* vertices of uncached draws by reason, for the [vbcache] log */
enum { VW_UP, VW_CPU_VS, VW_NOBUF, VW_CHANGED, VW_RANGE, VW_OTHER, VW_N };
static uint64_t g_vbc_why[VW_N];

/* set by an executor that handles CMD_UPLOAD_VB and indexed draws (render_wgpu.c) */
__attribute__((weak)) int g_gpu_vb_cache;

static uint32_t vbkey_hash(const VbKey *k) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < 2 + 4 * (size_t)k->n; i++) h = (h ^ ((const uint32_t *)k)[i]) * 16777619u;
    return h % VBC_BUCKETS;
}

static void vbc_drop(int32_t e) {
    VbEntry *v = &g_vbc[e];
    for (int32_t *p = &g_vbc_bucket[vbkey_hash(&v->k)]; *p >= 0; p = &g_vbc[*p].next)
        if (*p == e) { *p = v->next; break; }
    CmdDestroy *c = cmd_alloc(CMD_DESTROY_VB, sizeof *c);
    c->id = v->vb_id;
    cmd_end();
    g_vbc_bytes -= v->bytes;
    v->vb_id = 0;
    v->next = g_vbc_free;
    g_vbc_free = e;
}

/* drops entries unused for 10 s, then least recently used ones (not this frame's) until `need` more fits */
static void vbc_evict(size_t budget, size_t need) {
    for (int32_t e = 0; e < g_vbc_used; e++)
        if (g_vbc[e].vb_id && g_vbc_frame - g_vbc[e].last_frame > 600) vbc_drop(e);
    while (g_vbc_bytes + need > budget) {
        int32_t old = -1;
        for (int32_t e = 0; e < g_vbc_used; e++)
            if (g_vbc[e].vb_id && g_vbc[e].last_frame != g_vbc_frame && (old < 0 || g_vbc[e].last_frame < g_vbc[old].last_frame)) old = e;
        if (old < 0) break;
        vbc_drop(old);
    }
}

static size_t g_vbc_budget;
/* the cache and indexed draws are on (off: every draw sends its vertices expanded, as before them) */
static int vbc_active(void) {
    static int enabled = -1, ab;
    if (enabled < 0) {
        enabled = g_gpu_vb_cache && (!getenv("RC_VB_CACHE") || atoi(getenv("RC_VB_CACHE")));
        g_vbc_budget = (size_t)(getenv("RC_VB_CACHE_MB") ? atoi(getenv("RC_VB_CACHE_MB")) : 64) << 20;
        for (int i = 0; i < VBC_BUCKETS; i++) g_vbc_bucket[i] = -1;
        /* RC_VB_CACHE_AB=N (checking): on only in every other N-frame period, to compare screenshots */
        ab = getenv("RC_VB_CACHE_AB") ? atoi(getenv("RC_VB_CACHE_AB")) : 0;
        fprintf(stderr, "[vbcache] %s\n", enabled ? "on" : "off");
    }
    return enabled && !(ab > 0 && (g_vbc_frame / (uint32_t)ab) & 1);
}

/* the cached GPU buffer holding this draw's vertices (converted now if needed), or 0: use the per-draw path.
   gpu_in: input registers of a vertex shader run on the GPU (sh set), else fixed function with FVF layout L
   and stream 0's stride. Indices (idx) are relative to the streams' offsets (Direct3D 9 SetStreamSource). */
static uint32_t vbc_lookup(Device *d, Shader *sh, int gpu_in, const FvfLayout *L, uint32_t stride, const uint32_t *idx,
                           uint32_t n) {
    if (!vbc_active()) return 0;
    size_t budget = g_vbc_budget;
    const DevState *st = &d->st;
    VbKey k;
    memset(&k, 0, sizeof k);
    k.layout = st->vs;
    Buffer *bufs[VBC_MAX_STREAMS];
    uint32_t streams[VBC_MAX_STREAMS], mask = sh ? vs_stream_mask(sh) : 1, nv = 0xFFFFFFFFu;
    for (uint32_t s = 0; s < 16; s++) {
        if (!(mask & (1u << s))) continue;
        Buffer *b = com_data(st->stream_vb[s]);
        uint32_t str = sh ? st->stream_stride[s] : stride, off = st->stream_offset[s];
        if (!b || !str || k.n == VBC_MAX_STREAMS || b->index || off >= b->size) { g_vbc_why[VW_NOBUF] += n; return 0; }
        if (b->gpu_dynamic) { g_vbc_why[VW_CHANGED] += n; return 0; }
        if (!b->gpu_id) b->gpu_id = gpu_new_id();
        k.s[k.n].id = b->gpu_id; k.s[k.n].version = b->version; k.s[k.n].stride = str; k.s[k.n].offset = off;
        bufs[k.n] = b; streams[k.n++] = s;
        if ((b->size - off) / str < nv) nv = (b->size - off) / str;
    }
    if (!k.n || !nv || nv == 0xFFFFFFFFu) { g_vbc_why[VW_OTHER] += n; return 0; }
    for (uint32_t i = 0; i < n; i++) if (idx[i] >= nv) { g_vbc_why[VW_RANGE] += n; return 0; }

    uint32_t h = vbkey_hash(&k);
    for (int32_t e = g_vbc_bucket[h]; e >= 0; e = g_vbc[e].next)
        if (!memcmp(&g_vbc[e].k, &k, sizeof k)) { g_vbc[e].last_frame = g_vbc_frame; g_vbc_hits++; return g_vbc[e].vb_id; }

    /* a buffer that changed since it was last converted is probably rewritten as the game goes (DYNAMIC
       usage alone says nothing) */
    for (uint32_t i = 0; i < k.n; i++) {
        Buffer *b = bufs[i];
        if (b->gpu_converted && b->gpu_conv_version != b->version && ++b->gpu_changes >= 3) { b->gpu_dynamic = 1; g_vbc_why[VW_CHANGED] += n; return 0; }
    }
    uint32_t vsize = sh ? 16u * (uint32_t)(gpu_in > 0 ? gpu_in : 1) : (uint32_t)sizeof(CVert);
    if ((uint64_t)nv * vsize > budget / 4) { g_vbc_why[VW_OTHER] += n; return 0; }
    uint32_t bytes = nv * vsize;
    vbc_evict(budget, bytes);
    if (g_vbc_free < 0 && g_vbc_used == VBC_CAP) { g_vbc_why[VW_OTHER] += n; return 0; }
    g_vbc_misses++;
    int32_t e = g_vbc_free >= 0 ? g_vbc_free : g_vbc_used++;
    if (g_vbc_free == e) g_vbc_free = g_vbc[e].next;
    VbEntry *v = &g_vbc[e];
    v->k = k; v->vb_id = gpu_new_id(); v->bytes = bytes; v->last_frame = g_vbc_frame;
    v->next = g_vbc_bucket[h];
    g_vbc_bucket[h] = e;
    g_vbc_bytes += bytes;
    for (uint32_t i = 0; i < k.n; i++) { bufs[i]->gpu_converted = 1; bufs[i]->gpu_conv_version = bufs[i]->version; }

    CmdUploadVb *c = cmd_alloc(CMD_UPLOAD_VB, sizeof *c + bytes);
    c->id = v->vb_id;
    c->bytes = bytes;
    if (sh) {
        VsStreams vss;
        memset(&vss, 0, sizeof vss);
        for (uint32_t i = 0; i < k.n; i++) { vss.data[streams[i]] = HP(bufs[i]->mem) + k.s[i].offset; vss.stride[streams[i]] = k.s[i].stride; }
        if (gpu_in > 0) vs_fetch_n(sh, &vss, NULL, 0, nv, (float *)(c + 1));
    } else {
        const uint8_t *vb = HP(bufs[0]->mem) + k.s[0].offset;
        CVert *o = (CVert *)(c + 1);
        for (uint32_t i = 0; i < nv; i++) convert_vertex(vb + (size_t)i * stride, L, &o[i]);
    }
    cmd_end();
    return v->vb_id;
}

static void vbc_end_frame(void) {
    g_vbc_frame++;
    static int ab = -1;
    if (ab < 0) ab = getenv("RC_VB_CACHE_AB") ? atoi(getenv("RC_VB_CACHE_AB")) : 0;
    if (ab > 0 && g_vbc_frame % (uint32_t)ab == 0)
        fprintf(stderr, "[vbcache] A/B: cache %s\n", (g_vbc_frame / (uint32_t)ab) & 1 ? "off" : "on");
    static int log = -1;
    if (log < 0) log = getenv("RC_LOG_FPS") != NULL;
    if (log && g_vbc_frame % 120 == 0) {
        int32_t live = 0;
        for (int32_t e = 0; e < g_vbc_used; e++) live += g_vbc[e].vb_id != 0;
        fprintf(stderr, "[vbcache] %d buffers, %.1f MB; per frame: %.1f cached draws, %.2f conversions; uncached "
                "vertices: UP %.0f, CPU VS %.0f, no buffer %.0f, changed %.0f, range %.0f, other %.0f\n", live,
                g_vbc_bytes / 1048576.0, (g_vbc_hits + g_vbc_misses) / 120.0, g_vbc_misses / 120.0,
                g_vbc_why[VW_UP] / 120.0, g_vbc_why[VW_CPU_VS] / 120.0, g_vbc_why[VW_NOBUF] / 120.0,
                g_vbc_why[VW_CHANGED] / 120.0, g_vbc_why[VW_RANGE] / 120.0,
                g_vbc_why[VW_OTHER] / 120.0);
        g_vbc_hits = g_vbc_misses = 0;
        memset(g_vbc_why, 0, sizeof g_vbc_why);
    }
}

/* ------------------------------------------------------------------ clears */

static void clear_quad(Device *d, float x1, float y1, float x2, float y2, uint32_t flags, uint32_t color, float z) {
    if (x2 <= x1 || y2 <= y1 || !(flags & 3)) return;
    CmdDraw *c = cmd_alloc(CMD_DRAW, sizeof *c + 6 * sizeof(CVert));
    c->key = (PKey){ 0, 2, 1, 1, 1, (flags & 2) ? 1u : 0u, 8, 1, (flags & 1) ? 0xFu : 0u, TOPO_TRIANGLES, 0 };
    c->full_viewport = 1;
    c->minz = 0.f; c->maxz = 1.f;
    c->capture_index = -1;
    for (int i = 0; i < 4; i++) c->samp_key[i] = sampler_key(d->st.tss[i]);
    c->u.vp[2] = (float)g_tgt_lw; c->u.vp[3] = (float)g_tgt_lh;
    c->u.flags[0] = 1;   /* pre-transformed */
    c->u.st0[0] = 1;     /* stage disabled: output = diffuse */
    c->nverts = 6;
    CVert *v = (CVert *)(c + 1);
    const float px[6] = { x1, x2, x1, x1, x2, x2 }, py[6] = { y1, y1, y2, y2, y1, y2 };
    for (int i = 0; i < 6; i++) { v[i].x = px[i] - 0.5f; v[i].y = py[i] - 0.5f; v[i].z = z; v[i].w = 1.f; v[i].diffuse = color; }
    cmd_end();
}

static void rec_clear_impl(Device *d, uint32_t count, uint32_t rects, uint32_t flags, uint32_t color, float z,
                      uint32_t stencil) {
    select_target(d);
    const DevState *st = &d->st;
    float vx1 = (float)st->viewport.x, vy1 = (float)st->viewport.y;
    float vx2 = vx1 + (float)(st->viewport.w ? st->viewport.w : g_tgt_lw);
    float vy2 = vy1 + (float)(st->viewport.h ? st->viewport.h : g_tgt_lh);
    int full = !count && vx1 <= 0 && vy1 <= 0 && vx2 >= (float)g_tgt_lw && vy2 >= (float)g_tgt_lh;
    if (full) {
        CmdClear *c = cmd_alloc(CMD_CLEAR, sizeof *c);
        c->flags = flags & 7; c->color = color; c->z = z; c->stencil = stencil;
        cmd_end();
        return;
    }
    /* Direct3D clears only the viewport (or the given rects): draw quads instead of a pass clear */
    for (uint32_t i = 0; i < (count ? count : 1); i++) {
        float x1 = vx1, y1 = vy1, x2 = vx2, y2 = vy2;
        if (count && rects) {
            int32_t r[4];
            memcpy(r, HP(rects + 16 * i), 16);
            if (r[0] > x1) x1 = (float)r[0];
            if (r[1] > y1) y1 = (float)r[1];
            if (r[2] < x2) x2 = (float)r[2];
            if (r[3] < y2) y2 = (float)r[3];
        }
        clear_quad(d, x1, y1, x2, y2, flags, color, z);
    }
}

/* ------------------------------------------------------------------ draw */

static void rec_draw_impl(Device *d, uint32_t prim, uint32_t prim_count, uint32_t start_vertex, uint32_t num_vertices,
                     const uint8_t *vertices_up, uint32_t up_stride, const void *indices, int index32,
                     uint32_t start_index) {
    DevState *st = &d->st;
    (void)num_vertices;
    if (!prim_count) return;
    Shader *sh = (st->vs & 1) ? shader_get(st->vs) : NULL;   /* vertex shader handle (FVFs have bit 0 clear) */
    if ((st->vs & 1) && !sh) { if (g_snap_detail) fprintf(stderr, "[skip] unknown vertex shader %x n %u\n", st->vs, prim_count); return; }
    {   /* debugging: RC_MAX_DRAWS=N drops every draw after the Nth of each frame */
        static long maxd = -1;
        if (maxd < 0) maxd = getenv("RC_MAX_DRAWS") ? atol(getenv("RC_MAX_DRAWS")) : 0;
        if (maxd && g_f_drawn >= (uint32_t)maxd) { g_f_drawn++; return; }
    }
    g_f_drawn++;
    if (sh && !g_cam_seen && st->viewport.w > 600) {   /* the frame's first shader draw is level geometry: c4-c7 its world-view-projection (camera angle, host/dinput.c) */
        memcpy(g_cam_c, st->vs_const[0], sizeof g_cam_c);
        g_cam_seen = 1;
    }
    {   /* debugging: <game>/../hide.txt "lo hi" drops draws lo..hi of each frame (checked every 30 frames; delete the file to stop) */
        /* or a group of draws by name, to measure what a kind of draw costs (with RC_GPU_TIMES=1): "light" (additive
           ONE/ONE passes, e.g. Warrior Within's per-light redraws), "atest" (alpha-tested), "blend" (other blended),
           "opaque" (neither) */
        static uint32_t lo = 1, hi = 0, chk;
        static int grp;   /* 0 none, 1 light, 2 atest, 3 blend, 4 opaque */
        if (++chk % 2000 == 0) {
            char hp[4200]; snprintf(hp, sizeof hp, "%s/../hide.txt", g_game_root);
            FILE *f = fopen(hp, "r");
            lo = 1; hi = 0; grp = 0;
            if (f) {
                char w[16] = { 0 };
                if (fscanf(f, "%u %u", &lo, &hi) != 2) {
                    lo = 1; hi = 0;
                    rewind(f);
                    if (fscanf(f, "%15s", w) == 1)
                        grp = !strcmp(w, "light") ? 1 : !strcmp(w, "atest") ? 2 : !strcmp(w, "blend") ? 3 : !strcmp(w, "opaque") ? 4 : 0;
                }
                fclose(f);
            }
        }
        if (g_f_drawn - 1 >= lo && g_f_drawn - 1 <= hi) return;
        if (grp) {
            int blend = st->rs[27] != 0, light = blend && st->rs[19] == 2 && st->rs[20] == 2, atest = st->rs[15] != 0;
            int g = light ? 1 : atest ? 2 : blend ? 3 : 4;
            if (g == grp) return;
        }
    }
    {   /* debugging: RC_SKIP_PS=<hex handle>: drop draws that use that pixel shader */
        static long skip_ps = -1;
        if (skip_ps < 0) skip_ps = getenv("RC_SKIP_PS") ? strtol(getenv("RC_SKIP_PS"), NULL, 16) : 0;
        if (skip_ps && st->ps == (uint32_t)skip_ps) return;
    }
    {   /* debugging: RC_SKIP_UNTEXTURED=1 drops draws with no texture and plain diffuse colour (to see what is under them) */
        static int skip = -1;
        if (skip < 0) skip = getenv("RC_SKIP_UNTEXTURED") != NULL;
        if (skip && !st->texture[0] && st->tss[0][1] == 3 && !st->ps) return;
    }
    select_target(d);
    if (g_snap_detail) {
        Texture *t0 = com_data(st->texture[0]), *t1 = com_data(st->texture[1]);
        fprintf(stderr, "[draw] #%04u vs %x ps %x prim %u n %u | tex0 %s fmt %x tex1 %s fmt %x | tss0 op %u a %u,%u aop %u | "
                "tss1 op %u | blend %u %u/%u atest %u | tf %08x cw %x z %u/%u/%u cull %u stencil %u fn %u ref %u fail %u/%u/%u wmask %x vp %g %g %u,%u,%u,%u\n", g_f_drawn - 1, st->vs, st->ps, prim, prim_count,
                t0 ? (t0->type == RT_TEXTURE ? "2d" : "cube") : "-", t0 ? t0->format : 0,
                t1 ? (t1->type == RT_TEXTURE ? "2d" : "cube") : "-", t1 ? t1->format : 0,
                st->tss[0][1], st->tss[0][2], st->tss[0][3], st->tss[0][4], st->tss[1][1], st->rs[27], st->rs[19],
                st->rs[20], st->rs[15], st->rs[60], st->rs[168], st->rs[7], st->rs[14], st->rs[23], st->rs[22], st->rs[52], st->rs[56], st->rs[57], st->rs[53], st->rs[54], st->rs[55], st->rs[59], st->viewport.minz, st->viewport.maxz, st->viewport.x, st->viewport.y, st->viewport.w, st->viewport.h);
        for (int s = 0; s < 4; s++) {   /* every bound texture: size, format, render target or not, texture transform flags, addressing */
            Texture *ts = com_data(st->texture[s]);
            if (ts) fprintf(stderr, "    [tex%d] %ux%u fmt %x usage %x ttff %x addr %u/%u filt %u/%u\n", s, ts->lw[0], ts->lh[0], ts->format, ts->usage,
                            st->tss[s][24], st->tss[s][13], st->tss[s][14], st->tss[s][16], st->tss[s][17]);
        }
    }

    FvfLayout L;
    fvf_layout(sh ? 0x2 /* XYZ: transformed by the shader */ : st->vs, &L);
    const uint8_t *vb = NULL;
    uint32_t stride;
    if (vertices_up) { vb = vertices_up; stride = up_stride; }
    else {
        Buffer *b = com_data(st->stream_vb[0]);
        if (!b && !sh) { if (g_snap_detail) fprintf(stderr, "[skip] no vertex buffer, fvf %x n %u\n", st->vs, prim_count); return; }
        if (b) vb = HP(b->mem) + st->stream_offset[0];
        stride = st->stream_stride[0];
    }
    if (!stride && !sh) stride = L.stride;
    uint32_t base = indices && !vertices_up ? st->base_vertex : 0;

    /* expand to a list topology */
    uint32_t topo, n = 0;
    uint32_t cap = prim_count * 3 + 4;
    uint32_t *idx = malloc(4 * cap);
#define IDX(i) (indices ? (index32 ? ((const uint32_t *)indices)[start_index + (i)] : ((const uint16_t *)indices)[start_index + (i)]) + base : start_vertex + (i))
    switch (prim) {
    case 1: topo = TOPO_POINTS; for (uint32_t i = 0; i < prim_count; i++) idx[n++] = IDX(i); break;
    case 2: topo = TOPO_LINES; for (uint32_t i = 0; i < 2 * prim_count; i++) idx[n++] = IDX(i); break;
    case 3: topo = TOPO_LINES;
        for (uint32_t i = 0; i < prim_count; i++) { idx[n++] = IDX(i); idx[n++] = IDX(i + 1); } break;
    case 4: topo = TOPO_TRIANGLES; for (uint32_t i = 0; i < 3 * prim_count; i++) idx[n++] = IDX(i); break;
    case 5: topo = TOPO_TRIANGLES;
        for (uint32_t i = 0; i < prim_count; i++) {
            if (i & 1) { idx[n++] = IDX(i + 1); idx[n++] = IDX(i); idx[n++] = IDX(i + 2); }
            else { idx[n++] = IDX(i); idx[n++] = IDX(i + 1); idx[n++] = IDX(i + 2); }
        }
        break;
    case 6: topo = TOPO_TRIANGLES;
        for (uint32_t i = 0; i < prim_count; i++) { idx[n++] = IDX(0); idx[n++] = IDX(i + 1); idx[n++] = IDX(i + 2); }
        break;
    default: if (g_snap_detail) fprintf(stderr, "[skip] primitive type %u n %u\n", prim, prim_count); free(idx); return;
    }
    if (n && vb && (st->vs == 0x1007du || (!(st->vs & 1) && (st->vs & 0xE) == 0x4))) {   /* a 2D (UI) draw */
        Texture *t = com_data(st->texture[0]);
        if (t && t->type == RT_TEXTURE && !(t->usage & 1)) {
            const float *p = (const float *)(vb + (size_t)idx[0] * stride);
            ui_note(t, p[0], p[1]);
        }
    }
#undef IDX

    /* textures first: an upload is its own command and must precede the draw */
    for (int s = 0; s < 4; s++) {
        Texture *t = com_data(st->texture[s]);
        if (t && t->type == RT_TEXTURE && !(t->usage & 1)) sync_texture(d, t);
    }

    /* Vertex shaders run on the GPU (translated to WGSL by vshader.c): here only their input registers are
       fetched (packed types expanded to float4). RC_CPU_VS=1 runs them on the CPU instead, as do
       declaration-only (fixed-function) shaders and shaders that can't be decoded. */
    static int cpu_vs = -1;
    if (cpu_vs < 0) {
        cpu_vs = getenv("RC_CPU_VS") && atoi(getenv("RC_CPU_VS"));
        if (cpu_vs) fprintf(stderr, "[render] vertex shaders run on the CPU (RC_CPU_VS)\n");
    }
    int gpu_in = sh && !cpu_vs && !(g_snap_detail && st->vs == g_trace_vs) ? vs_gpu_inputs(sh) : -1;
    if (sh && gpu_in < 0 && !cpu_vs) {   /* log each shader that stays on the CPU once */
        static uint32_t logged[64]; static int nlogged;
        int seen = 0;
        for (int i = 0; i < nlogged; i++) seen |= logged[i] == st->vs;
        if (!seen && nlogged < 64) {
            logged[nlogged++] = st->vs;
            fprintf(stderr, "[render] vertex shader %x runs on the CPU (%s)\n", st->vs,
                    vs_is_fixed_function(sh) ? "declaration without a shader" : "not decodable");
        }
    }
    uint32_t vs_stride = gpu_in > 0 ? 16u * (uint32_t)gpu_in : 16u;
    /* vertices from an unchanging buffer: converted once and kept on the GPU, the draw sends indices */
    uint32_t vb_id = !vertices_up && (sh ? gpu_in > 0 : vb != NULL) ? vbc_lookup(d, sh, gpu_in, &L, stride, idx, n) : 0;
    if (vertices_up) g_vbc_why[VW_UP] += n;
    else if (sh && gpu_in <= 0) g_vbc_why[VW_CPU_VS] += n;
    /* otherwise indexed meshes convert each referenced vertex once (over the referenced range) rather than
       once per index, when that sends less */
    uint32_t vsize = gpu_in >= 0 ? vs_stride : (uint32_t)sizeof(CVert), lo = 0, nv = n, nidx = 0;
    if (!vb_id && indices && n && (sh ? gpu_in > 0 : 1) && vbc_active()) {
        uint32_t l = 0xFFFFFFFFu, h = 0;
        for (uint32_t i = 0; i < n; i++) { if (idx[i] < l) l = idx[i]; if (idx[i] > h) h = idx[i]; }
        if ((uint64_t)(h - l + 1) * vsize + 4ull * n < (uint64_t)n * vsize) { lo = l; nv = h - l + 1; nidx = n; }
    }
    size_t vbytes = (gpu_in >= 0 ? sizeof(float) * 96 * 4 : 0) + (vb_id ? (size_t)n * 4 : (size_t)nv * vsize + (size_t)nidx * 4);
    CmdDraw *c = cmd_alloc(CMD_DRAW, sizeof *c + vbytes);
    CVert *cv = (CVert *)(c + 1);
    c->nverts = vb_id ? n : nv;
    c->nidx = nidx;
    c->vs_stride = gpu_in >= 0 ? vs_stride : 0;
    c->vb_id = vb_id;
    c->capture_index = g_snap_detail ? (int32_t)(g_f_drawn - 1) : -1;
    if (vb_id) {
        float *consts = (float *)(c + 1);
        if (sh) vs_constants(sh, (const float (*)[4])st->vs_const, (float (*)[4])consts);
        memcpy(sh ? (void *)(consts + 96 * 4) : (void *)cv, idx, (size_t)n * 4);
    } else if (sh && gpu_in >= 0) {
        VsStreams vss;
        memset(&vss, 0, sizeof vss);
        if (vertices_up) { vss.data[0] = vertices_up; vss.stride[0] = up_stride; }
        else
            for (int s = 0; s < 16; s++) {
                Buffer *sb = com_data(st->stream_vb[s]);
                if (sb) { vss.data[s] = HP(sb->mem) + st->stream_offset[s]; vss.stride[s] = st->stream_stride[s]; }
            }
        float *consts = (float *)(c + 1);
        vs_constants(sh, (const float (*)[4])st->vs_const, (float (*)[4])consts);
        uint8_t *vout = (uint8_t *)(consts + 96 * 4);
        if (gpu_in) vs_fetch_n(sh, &vss, nidx ? NULL : idx, lo, nv, (float *)vout);   /* (zeroed by cmd_alloc otherwise) */
        if (nidx) { uint32_t *o = (uint32_t *)(vout + (size_t)nv * vs_stride); for (uint32_t i = 0; i < n; i++) o[i] = idx[i] - lo; }
    } else if (sh) {
        VsStreams vss;
        memset(&vss, 0, sizeof vss);
        if (vertices_up) { vss.data[0] = vertices_up; vss.stride[0] = up_stride; }
        else
            for (int s = 0; s < 16; s++) {
                Buffer *sb = com_data(st->stream_vb[s]);
                if (sb) { vss.data[s] = HP(sb->mem) + st->stream_offset[s]; vss.stride[s] = st->stream_stride[s]; }
            }
        float wv[16], ff_mvp[16];
        mat_mul(st->transform[256], st->transform[2], wv);
        mat_mul(wv, st->transform[3], ff_mvp);
        vss.ff_mvp = ff_mvp;
        /* run the shader once per distinct vertex: over the referenced range when it is compact */
        uint32_t lo = 0xFFFFFFFFu, hi = 0;
        for (uint32_t i = 0; i < n; i++) { if (idx[i] < lo) lo = idx[i]; if (idx[i] > hi) hi = idx[i]; }
        uint32_t range = n ? hi - lo + 1 : 0;
        int use_range = range <= 2 * n + 64;
        CVert *tmp = use_range ? malloc(sizeof(CVert) * (range ? range : 1)) : NULL;
        uint32_t count = use_range ? range : n;
        g_prof_verts += count; g_prof_draws_vs++;
        static long white = -1;
        if (white < 0) white = getenv("RC_FORCE_WHITE") ? strtol(getenv("RC_FORCE_WHITE"), NULL, 16) : 0;   /* debugging: ignore the diffuse output (of vertex shader handle N, or all with 1) */
        const int force_white = white && (white == 1 || (long)st->vs == white);
        /* shader outputs -> CVerts, chunk by chunk (no shared state, so a draw's vertices can run on several cores) */
        #define RUN_CHUNK(from, to) do { \
            VsOut ob[64]; \
            vs_run_batch(sh, (const float (*)[4])st->vs_const, &vss, use_range ? NULL : idx, lo, (from), (to), ob); \
            for (uint32_t i = (from); i < (to); i++) { \
                VsOut *o = &ob[i - (from)]; \
                CVert *v = use_range ? &tmp[i] : &cv[i]; \
                v->x = o->pos[0]; v->y = o->pos[1]; v->z = o->pos[2]; v->w = o->pos[3]; \
                if (force_white) { o->d0[0] = o->d0[1] = o->d0[2] = 0.5f; o->d0[3] = 1.f; } \
                v->diffuse = pack_color(o->d0); v->specular = pack_color(o->d1); \
                memcpy(v->tc, o->tex, sizeof v->tc); \
            } } while (0)
        if (!g_snap_detail) {
            extern int vs_is_fixed_function(Shader *);
            (void)vs_is_fixed_function(sh);   /* decodes the shader before several threads share it */
#ifdef __APPLE__
            if (count > 128) {
                const uint32_t chunk = 64, nchunks = (count + chunk - 1) / chunk;
                dispatch_apply(nchunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t k) {
                    uint32_t from = (uint32_t)k * chunk, to = from + chunk < count ? from + chunk : count;
                    RUN_CHUNK(from, to);
                });
            } else
#endif
            for (uint32_t from = 0; from < count; from += 64) { uint32_t to = from + 64 < count ? from + 64 : count; RUN_CHUNK(from, to); }
        } else
        for (uint32_t i = 0; i < (use_range ? range : n); i++) {
            VsOut o;
            extern int vs_trace;
            vs_trace = g_snap_detail && i == 0 && st->vs == g_trace_vs;
            vs_run(sh, (const float (*)[4])st->vs_const, &vss, use_range ? lo + i : idx[i], &o);
            vs_trace = 0;
            CVert *v = use_range ? &tmp[i] : &cv[i];
            v->x = o.pos[0]; v->y = o.pos[1]; v->z = o.pos[2]; v->w = o.pos[3];
            static long white = -1;
            if (white < 0) white = getenv("RC_FORCE_WHITE") ? strtol(getenv("RC_FORCE_WHITE"), NULL, 16) : 0;   /* debugging: ignore the diffuse output (of vertex shader handle N, or all with 1) */
            if (white && (white == 1 || (long)st->vs == white)) { o.d0[0] = o.d0[1] = o.d0[2] = 0.5f; o.d0[3] = 1.f; }
            if (g_snap_detail && i == 0 && o.d0[0] + o.d0[1] + o.d0[2] < 0.05f) {
                Buffer *b1 = com_data(st->stream_vb[1]);
                fprintf(stderr, "[stream] draw %u vs %x: s0 vb %x off %u stride %u | s1 vb %x off %u stride %u size %u first bytes %08x %08x\n", g_f_drawn - 1, st->vs,
                        st->stream_vb[0], st->stream_offset[0], st->stream_stride[0], st->stream_vb[1], st->stream_offset[1], st->stream_stride[1],
                        b1 ? b1->size : 0, b1 ? R32(b1->mem + st->stream_offset[1]) : 0, b1 ? R32(b1->mem + st->stream_offset[1] + 4) : 0);
            }
            if (g_snap_detail && i == 0 && (st->vs == 0x100ef || st->vs == 0x1008f) && g_f_drawn - 1 < 200) {
                fprintf(stderr, "[vsconst] draw %u vs %x:", g_f_drawn - 1, st->vs);
                for (int k = 0; k < 96; k++) if (st->vs_const[k][0] != 0 || st->vs_const[k][1] != 0 || st->vs_const[k][2] != 0 || st->vs_const[k][3] != 0)
                    fprintf(stderr, " c%d=(%g,%g,%g,%g)", k, st->vs_const[k][0], st->vs_const[k][1], st->vs_const[k][2], st->vs_const[k][3]);
                fprintf(stderr, "\n");
            }
            if (g_snap_detail && i == 0)
                fprintf(stderr, "[vsout] draw %u vs %x vertex %u: pos %g %g %g %g d0 %g %g %g %g tex0 %g %g\n", g_f_drawn - 1, st->vs,
                        use_range ? lo : idx[0], o.pos[0], o.pos[1], o.pos[2], o.pos[3], o.d0[0], o.d0[1], o.d0[2], o.d0[3], o.tex[0][0], o.tex[0][1]);
            v->diffuse = pack_color(o.d0); v->specular = pack_color(o.d1);
            memcpy(v->tc, o.tex, sizeof v->tc);
        }
        if (use_range) { for (uint32_t i = 0; i < n; i++) cv[i] = tmp[idx[i] - lo]; free(tmp); }
    } else {
        for (uint32_t i = 0; i < nv; i++) convert_vertex(vb + (size_t)(nidx ? lo + i : idx[i]) * stride, &L, &cv[i]);
        if (nidx) { uint32_t *o = (uint32_t *)(cv + nv); for (uint32_t i = 0; i < n; i++) o[i] = idx[i] - lo; }
    }
    free(idx);

    for (int s = 0; s < 4; s++) {
        Texture *t = com_data(st->texture[s]);
        c->samp_key[s] = sampler_key(st->tss[s]);
        if (t && t->type == RT_TEXTURE && (t->usage & 1)) {
            Surface *s0 = t->surfaces[0][0] ? com_data(t->surfaces[0][0]) : NULL;
            if (s0 && t->surfaces[0][0] == d->backbuffer) c->slot_kind[s] = TS_BACKBUF;
            else if (s0) { c->slot_kind[s] = TS_RTSURF; c->slot_id[s] = s0->gpu_id; }
        } else if (t && t->type == RT_TEXTURE) {
            c->slot_kind[s] = TS_TEX; c->slot_id[s] = t->gpu_id;
        }
    }
    c->pixel_shader = st->ps;

    Uniforms *u = &c->u;
    if (!sh && L.pos_type != 0x4) {
        float wv[16];
        mat_mul(st->transform[256], st->transform[2], wv);    /* WORLD * VIEW */
        mat_mul(wv, st->transform[3], u->mvp);                /* * PROJECTION */
    }
    u->vp[0] = (float)st->viewport.x; u->vp[1] = (float)st->viewport.y;
    u->vp[2] = (float)(st->viewport.w ? st->viewport.w : g_tgt_lw);
    u->vp[3] = (float)(st->viewport.h ? st->viewport.h : g_tgt_lh);
    u->flags[0] = sh ? 2 : L.pos_type == 0x4;
    u->flags[1] = st->rs[15] ? st->rs[25] : 0;                 /* ALPHATESTENABLE ? ALPHAFUNC */
    u->flags[2] = st->rs[29] != 0;                             /* SPECULARENABLE */
    u->flags[3] = !sh && L.pos_type != 0x4 && st->rs[137] && !L.has_normal;  /* LIGHTING without normals: ambient */
    for (int s = 0; s < 2; s++) {
        const uint32_t *t = st->tss[s];
        uint32_t *a = s ? u->st1 : u->st0, *b = s ? u->st1b : u->st0b;
        a[0] = t[1]; a[1] = t[2]; a[2] = t[3]; a[3] = t[4];
        b[0] = t[5]; b[1] = t[6]; b[2] = sh ? (uint32_t)s : t[11] & 7; b[3] = (t[26] & 255) | (t[27] & 255) << 8;
    }
    unpack_color(st->rs[60], u->tfactor);
    u->misc[0] = (st->rs[24] & 255) / 255.f;
    {   /* pixel shader 1.x: stages with D3DTTFF_PROJECTED (256) divide their coordinates by the last one (water's screen-space refraction) */
        uint32_t pj = 0;
        if (st->ps)
            for (int s = 0; s < 4; s++)
                if (st->tss[s][24] & 256) { uint32_t c = st->tss[s][24] & 255; pj |= (c < 2 || c > 4 ? 4u : c) << (3 * s); }
        u->misc[1] = (float)pj;
    }
    for (int i = 0; i < 4; i++) u->mat_diffuse[i] = 1.f;
    if (u->flags[3]) {
        /* no normals: lit color is emissive + ambient * material ambient (diffuse lighting needs a normal) */
        float amb[4];
        unpack_color(st->rs[139], amb);
        for (int i = 0; i < 3; i++) { u->mat_diffuse[i] = 0.f; u->ambient[i] = amb[i] * st->material[8 + i] + st->material[12 + i]; }
        u->mat_diffuse[3] = 1.f;
        if (st->rs[141] && L.has_diffuse && st->rs[147] == 1) {  /* COLORVERTEX + AMBIENTMATERIALSOURCE=COLOR1 */
            for (int i = 0; i < 3; i++) { u->mat_diffuse[i] = amb[i]; u->ambient[i] = st->material[12 + i]; }
        }
    }
    memcpy(u->psc, st->ps_const, sizeof u->psc);
    if (g_snap_detail && st->ps) fprintf(stderr, "[psc] draw %u ps %x c0=(%g,%g,%g,%g) c1=(%g,%g,%g,%g) rt %x tex0 %x tex1 %x\n", g_f_drawn - 1, st->ps,
        st->ps_const[0][0], st->ps_const[0][1], st->ps_const[0][2], st->ps_const[0][3], st->ps_const[1][0], st->ps_const[1][1], st->ps_const[1][2], st->ps_const[1][3],
        st->render_target, st->texture[0], st->texture[1]);
    for (int s = 0; s < 4; s++) {
        memcpy(u->bem[s], &st->tss[s][7], 16);        /* BUMPENVMAT00, 01, 10, 11 (float bits) */
        memcpy(&u->bel[s][0], &st->tss[s][22], 4);    /* BUMPENVLSCALE */
        memcpy(&u->bel[s][1], &st->tss[s][23], 4);    /* BUMPENVLOFFSET */
    }

    PKey k = { st->rs[27] != 0, st->rs[19], st->rs[20], st->rs[171] ? st->rs[171] : 1, st->rs[7] != 0 && st->depth_stencil,
               st->rs[14], st->rs[23], st->rs[22], st->rs[168] & 0xF, topo, st->ps };
    if (L.pos_type == 0x4) k.cull = 1;  /* keep 2D simple: no culling for pre-transformed vertices */
    /* D3DCULL_CW (2) culls clockwise triangles, which WebGPU front-face CW makes the front: cull modes map straight through
       (swapping them dropped every back-facing surface: sky, sea, ground) */
    { static int nc = -1; if (nc < 0) nc = getenv("RC_NO_CULL") != NULL; if (nc) k.cull = 1; }   /* debugging */
    k.vs = gpu_in >= 0 ? st->vs : 0;
    if (st->rs[52] && st->depth_stencil) {   /* STENCILENABLE: character shadows are drawn through the stencil buffer */
        const uint32_t *r = st->rs;
        k.sten = 1 | (r[56] & 15) << 4 | (r[53] & 15) << 8 | (r[54] & 15) << 12 | (r[55] & 15) << 16 | (r[185] ? 1u : 0u) << 20 |
                 (r[189] & 15) << 24;
        k.sten2 = (r[186] & 15) | (r[187] & 15) << 4 | (r[188] & 15) << 8 | (r[58] & 255) << 12 | (r[59] & 255) << 20;
        c->stencil_ref = r[57] & 255;
    }
    c->key = k;
    c->vp[0] = (float)st->viewport.x; c->vp[1] = (float)st->viewport.y;
    c->vp[2] = (float)(st->viewport.w ? st->viewport.w : g_tgt_lw);
    c->vp[3] = (float)(st->viewport.h ? st->viewport.h : g_tgt_lh);
    c->minz = st->viewport.minz;
    c->maxz = st->viewport.maxz > st->viewport.minz ? st->viewport.maxz : 1.f;
    cmd_end();
}

/* ------------------------------------------------------------------ frames, readback, resources */

static int rec_init(Device *d) {
    (void)d;
#ifdef __EMSCRIPTEN__
    while (!web_gpu_ready()) usleep(1000);   /* the page creates the WebGPU device before starting the game */
    return 1;
#else
    return gpu_init_native();
#endif
}

static void rec_present_impl(Device *d) {
    CmdPresent *c = cmd_alloc(CMD_PRESENT, sizeof *c);
    c->frame = d->frame;
    c->screenshot = (uint32_t)want_screenshot(d->frame);
    cmd_end();
    cmd_flush(1);
    g_target_sent = 0;   /* the executor ends every frame on the back buffer */
    vbc_end_frame();
    ui_frame_done(d->frame);
    g_f_drawn = 0;
#ifdef __EMSCRIPTEN__
    /* Don't run ahead of the display: two frames queued (a slow frame doesn't skip a vsync). Online every
       queued frame is a frame of input latency on top of the lockstep delay, so RC_QUEUE=1 (?queue=1 on
       the page) allows one there, for a machine that simulates a frame well within 16 ms. */
    static int depth = -1;
    if (depth < 0) {
        const char *q = getenv("RC_QUEUE");
        depth = q ? atoi(q) : 2;
        if (depth < 1) depth = 1;
    }
    while (web_queued_frames() >= depth) usleep(1000);
#endif
}

static void rec_readback_impl(Device *d, Surface *s) {
    int is_bb = com_data(d->backbuffer) == s;
    if (!is_bb && !s->is_rt) return;
    if (!s->mem) s->mem = heap_alloc(s->pitch * s->height + 16, 1);
    volatile int done = 0;
    CmdReadback *c = cmd_alloc(CMD_READBACK, sizeof *c);
    c->id = is_bb ? 0 : s->gpu_id;
    c->w = s->width; c->h_ = s->height; c->fmt = s->format; c->pitch = s->pitch; c->guest_mem = s->mem;
    c->done_ptr = (uint64_t)(uintptr_t)&done;
    cmd_end();
    cmd_flush(0);
    while (!__atomic_load_n(&done, __ATOMIC_ACQUIRE)) usleep(200);   /* natively it already ran */
}

static void rec_upload_surface_impl(Device *d, Surface *s) {
    if (!s->mem || s->is_depth || com_data(d->backbuffer) == s) return;
    CmdUploadSurf *c = cmd_alloc(CMD_UPLOAD_SURF, sizeof *c + (size_t)s->width * s->height * 4);
    c->id = s->gpu_id; c->w = s->width; c->h_ = s->height;
    if (!texture_to_rgba8(s->format, HP(s->mem), s->pitch, s->width, s->height, (uint8_t *)(c + 1), NULL))
        memset(c + 1, 0, (size_t)s->width * s->height * 4);
    cmd_end();
}

static void rec_destroy_texture_impl(Texture *t) {
    if (t->sent_version == 0xFFFFFFFFu) return;   /* never reached the GPU */
    CmdDestroy *c = cmd_alloc(CMD_DESTROY_TEX, sizeof *c);
    c->id = t->gpu_id;
    cmd_end();
}

static void rec_destroy_surface_impl(Surface *s) {
    CmdDestroy *c = cmd_alloc(CMD_DESTROY_SURF, sizeof *c);
    c->id = s->gpu_id;
    cmd_end();
}

/* ------------------------------------------------------------------ locked entry points */

static void rec_clear(Device *d, uint32_t count, uint32_t rects, uint32_t flags, uint32_t color, float z, uint32_t stencil) {
    rec_enter(); rec_clear_impl(d, count, rects, flags, color, z, stencil); rec_leave();
}
static void rec_draw(Device *d, uint32_t prim, uint32_t prim_count, uint32_t start_vertex, uint32_t num_vertices,
                     const uint8_t *vertices_up, uint32_t up_stride, const void *indices, int index32, uint32_t start_index) {
    rec_enter();
    double t0_ = host_seconds();
    rec_draw_impl(d, prim, prim_count, start_vertex, num_vertices, vertices_up, up_stride, indices, index32, start_index);
    g_prof_draw += host_seconds() - t0_;
    {   /* RC_DUMP_AFTER=frame:draw[,draw...]: after those draws of that frame write the target (back buffer or render target) as /tmp/rc_after_<draw>.ppm */
        static const char *spec = (const char *)1;
        if (spec == (const char *)1) spec = getenv("RC_DUMP_AFTER");
        if (spec) {
            unsigned fr = 0, dr = 0; const char *p = spec;
            while (*p) {
                if (sscanf(p, "%u:%u", &fr, &dr) == 2 && fr == d->frame && dr == g_f_drawn - 1) {
                    fprintf(stderr, "[dumpafter] atest %u func %u ref %u scissor %u rect %u,%u,%u,%u fog %u stencil %u draw %u: vs %x ps %x tex0 %x tex1 %x c0=(%g,%g,%g,%g) rs: blend %u src %u dst %u cw %x z %u/%u\n", d->st.rs[15], d->st.rs[25], d->st.rs[24], d->st.rs[174], d->st.scissor[0], d->st.scissor[1], d->st.scissor[2], d->st.scissor[3], d->st.rs[28], d->st.rs[52], dr, d->st.vs, d->st.ps, d->st.texture[0], d->st.texture[1],
                            d->st.ps_const[0][0], d->st.ps_const[0][1], d->st.ps_const[0][2], d->st.ps_const[0][3], d->st.rs[27], d->st.rs[19], d->st.rs[20], d->st.rs[168], d->st.rs[7], d->st.rs[14]);
                    extern Surface *g_last_stretch_dst;
                    uint32_t tobj = d->st.render_target;
                    Surface *t = getenv("RC_DUMP_AFTER_DST") ? g_last_stretch_dst : com_data(tobj);
                    if (t && (t->mem || t == com_data(d->backbuffer))) {
                        if (!t->mem) t->mem = heap_alloc(t->pitch * t->height + 16, 1);
                        rec_leave(); g_render->readback(d, t); rec_enter();
                        char path[64]; snprintf(path, sizeof path, "/tmp/rc_after_%u.ppm", dr);
                        FILE *f = fopen(path, "wb");
                        if (f) {
                            fprintf(f, "P6\n%u %u\n255\n", t->width, t->height);
                            for (uint32_t y = 0; y < t->height; y++) for (uint32_t x = 0; x < t->width; x++) {
                                uint32_t px = R32(t->mem + y * t->pitch + x * 4);
                                fputc((px >> 16) & 255, f); fputc((px >> 8) & 255, f); fputc(px & 255, f);
                            }
                            fclose(f);
                        }
                    }
                }
                p = strchr(p, ','); if (!p) break; p++;
            }
        }
    }
    rec_leave();
}
static void rec_present(Device *d) { rec_enter(); double t0_ = host_seconds(); rec_present_impl(d); g_prof_present += host_seconds() - t0_; rec_leave(); }
static void rec_readback(Device *d, Surface *s) { rec_enter(); rec_readback_impl(d, s); rec_leave(); }
static void rec_upload_surface(Device *d, Surface *s) { rec_enter(); rec_upload_surface_impl(d, s); rec_leave(); }
static void rec_destroy_texture(Texture *t) { rec_enter(); rec_destroy_texture_impl(t); rec_leave(); }
static void rec_destroy_surface(Surface *s) { rec_enter(); rec_destroy_surface_impl(s); rec_leave(); }

RenderBackend g_render_gpu = { "webgpu", rec_init, rec_clear, rec_draw, rec_present, rec_destroy_texture,
                               rec_readback, rec_upload_surface, rec_destroy_surface };
