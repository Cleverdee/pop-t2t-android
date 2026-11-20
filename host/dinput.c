/* DirectInput (DirectX 7 era): keyboard, mouse and gamepad devices. Device state comes from g_keys / the
 * window backend; until there is a real window everything reads as "nothing pressed". */
#include "com.h"

#include <pthread.h>
#include <math.h>
#include <unistd.h>

#define W(name) static void name(CPU *c)

enum { DEV_KEYBOARD = 1, DEV_MOUSE = 2, DEV_JOYSTICK = 3 };
/* Joystick axis ranges (DIPROP_RANGE/DEADZONE) are per axis, indexed by DIJOYSTATE offset / 4 (X Y Z Rx Ry Rz). */
typedef struct { int kind, pad; uint32_t format_size; int acquired; int32_t lo[6], hi[6], dead[6]; } DIDev;

static const uint8_t GUID_SysMouse[16] = { 0x60, 0x2B, 0x1D, 0x6F, 0xA0, 0xD5, 0xCF, 0x11, 0xBF, 0xC7,
                                           0x44, 0x45, 0x53, 0x54, 0x00, 0x00 };
static const uint8_t GUID_SysKeyboard[16] = { 0x61, 0x2B, 0x1D, 0x6F, 0xA0, 0xD5, 0xCF, 0x11, 0xBF, 0xC7,
                                              0x44, 0x45, 0x53, 0x54, 0x00, 0x00 };

/* DirectInput keyboard state is indexed by DIK scan codes; the window backend fills this. */
uint8_t g_dik[256];
int32_t g_mouse_dx, g_mouse_dy, g_mouse_buttons;
/* A press counts as held for at least 50 ms: the game samples state once per frame, so a very quick
   tap (or an automated click) would otherwise fall between two samples. */
static double g_dik_down_at[256], g_button_down_at[4];
#define MIN_HOLD (deterministic() ? 0.0 : 0.05)   /* deterministic runs: state is exactly what was pressed */

/* Buffered input (GetDeviceData): DIDEVICEOBJECTDATA records per device kind. */
typedef struct { uint32_t ofs, data, time, seq; } DIEvent;
static DIEvent g_queue[4][256];   /* by device kind; joysticks are polled only, so theirs stays empty */
static int g_qhead[4], g_qtail[4];
static uint32_t g_seq;
static pthread_mutex_t g_di_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_di_lock_pads = PTHREAD_MUTEX_INITIALIZER;

/* Gamepads: always-present virtual joysticks (the game enumerates devices only once, at startup, so a
 * controller connected later must already exist). Each one is presented as a "Logitech Dual Action USB",
 * which Data/controltypes.csv maps as a dual-analog pad: buttons 1-4 = Square Cross Circle Triangle,
 * 5-8 = L1 R1 L2 R2, 9 = Start, 10 = Select, 11-12 = L3 R3; left stick X/Y, right stick Z/Rz, d-pad POV.
 * The feeders (SDL natively, the Gamepad API in the browser) report the W3C "standard gamepad" layout
 * through input_gamepad() in input.c. RC_PADS (default 1, at most 4) sets how many are exposed. */
#define MAX_PADS 4
typedef struct { int connected; uint32_t buttons; int32_t axis[4]; } PadState;  /* axes -32768..32767 */
static PadState g_pads[MAX_PADS];
static int g_npads = -1;
static int num_pads(void) {
    if (g_npads < 0) {
        const char *e = getenv("RC_PADS");
        g_npads = e ? atoi(e) : 1;
        if (g_npads < 0) g_npads = 0;
        if (g_npads > MAX_PADS) g_npads = MAX_PADS;
    }
    return g_npads;
}
void dinput_pad(int idx, int connected, uint32_t std_buttons, const int32_t axis[4]) {
    if (idx < 0 || idx >= MAX_PADS) return;
    pthread_mutex_lock(&g_di_lock_pads);
    if (connected && !g_pads[idx].connected) LOG("[dinput] gamepad %d connected\n", idx);
    if (!connected && g_pads[idx].connected) LOG("[dinput] gamepad %d disconnected\n", idx);
    g_pads[idx].connected = connected;
    g_pads[idx].buttons = connected ? std_buttons : 0;
    for (int i = 0; i < 4; i++) g_pads[idx].axis[i] = connected ? axis[i] : 0;
    pthread_mutex_unlock(&g_di_lock_pads);
}
/* Logitech Dual Action: USB 046D:C216. DirectInput product GUIDs are {PIDVID-0000-0000-0000-"PIDVID"}. */
static void pad_guid(uint8_t *g, int instance, int idx) {
    static const uint8_t product[16] = { 0x6D, 0x04, 0x16, 0xC2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                         0x50, 0x49, 0x44, 0x56, 0x49, 0x44 };
    static const uint8_t inst[16] = { 0x00, 0x7A, 0x6A, 0x52, 0x06, 0x00, 0xD1, 0x11, 0x80, 0x01,
                                      0x44, 0x45, 0x53, 0x54, 0x00, 0x00 };
    memcpy(g, instance ? inst : product, 16);
    if (instance) g[0] = (uint8_t)idx;
}
static int pad_from_guid(const uint8_t *g) {
    uint8_t t[16];
    for (int i = 0; i < num_pads(); i++) {
        pad_guid(t, 1, i);
        if (!memcmp(g, t, 16)) return i;
    }
    pad_guid(t, 0, 0);
    return !memcmp(g, t, 16) ? 0 : -1;   /* some games create by product GUID */
}
#define PAD_NAME "Logitech Dual Action USB"   /* controltypes.csv: *Logitech Dual Action USB* */
#define PAD_DEVTYPE 0x00010404u   /* DIDEVTYPE_JOYSTICK | DIDEVTYPEJOYSTICK_GAMEPAD << 8 | DIDEVTYPE_HID */
#define PAD_BUTTONS 12


static void push(int kind, uint32_t ofs, uint32_t data) {
    pthread_mutex_lock(&g_di_lock);
    int next = (g_qtail[kind] + 1) % 256;
    if (next != g_qhead[kind]) {
        g_queue[kind][g_qtail[kind]] = (DIEvent){ ofs, data, (uint32_t)(guest_seconds() * 1000), ++g_seq };
        g_qtail[kind] = next;
    }
    pthread_mutex_unlock(&g_di_lock);
}
/* Keyboard state as the game's polls see it (GetDeviceState). Key events are queued and handed over in order, at
   most one new press per poll, and a key's release only after a poll has seen it down: a phone keyboard (or adb)
   types faster than the game polls (~17-60 times a second), and a press and release between two polls, or several
   keys down in one poll, lose letters (typing "Prince" in the profile box gave "P"). */
static struct { uint8_t dik, down; } g_kq[512];
static int g_kq_head, g_kq_tail;
static uint8_t g_kvis[256], g_kseen[256];   /* visible state; set once a poll has reported the key down */
void dinput_key(uint8_t dik, int down) {
    g_dik[dik] = (uint8_t)down;
    if (down) g_dik_down_at[dik] = host_seconds();
    pthread_mutex_lock(&g_di_lock);
    int next = (g_kq_tail + 1) % 512;
    if (next != g_kq_head) { g_kq[g_kq_tail].dik = dik; g_kq[g_kq_tail].down = (uint8_t)down; g_kq_tail = next; }
    pthread_mutex_unlock(&g_di_lock);
    push(1, dik, down ? 0x80 : 0);
}
/* one keyboard poll: apply queued events up to the next press that needs a poll of its own */
static void kq_poll(void) {
    pthread_mutex_lock(&g_di_lock);
    int pressed = 0;
    while (g_kq_head != g_kq_tail) {
        uint8_t dik = g_kq[g_kq_head].dik, down = g_kq[g_kq_head].down;
        if (down) {
            if (pressed || (g_kvis[dik] && !g_kseen[dik])) break;   /* already one new press this poll */
            g_kvis[dik] = 1; g_kseen[dik] = 0; pressed = 1;
        } else {
            if (g_kvis[dik] && !g_kseen[dik]) break;   /* pressed but no poll has reported it yet */
            g_kvis[dik] = 0;
        }
        g_kq_head = (g_kq_head + 1) % 512;
    }
    for (int i = 0; i < 256; i++) if (g_kvis[i]) g_kseen[i] = 1;
    pthread_mutex_unlock(&g_di_lock);
}
void dinput_mouse_button(int b, int down) {
    if (down) { g_mouse_buttons |= 1 << b; if (b < 4) g_button_down_at[b] = host_seconds(); }
    else g_mouse_buttons &= ~(1 << b);
    push(2, 12u + (uint32_t)b, down ? 0x80 : 0);
}
static void mouse_deliver(int dx, int dy) {
    g_mouse_dx += dx; g_mouse_dy += dy;
    if (dx) push(2, 0, (uint32_t)dx);
    if (dy) push(2, 4, (uint32_t)dy);
}
#ifdef __ANDROID__
/* Touch drags arrive in bursts at the display's rate (90 Hz on the Galaxy A17) while the game reads the mouse ~40
   times a second, so each game frame got two or three bursts' worth: the camera speed jumped from frame to frame.
   Motion is collected here and handed out at each read in proportion to the time since the previous read
   (exponential easing, MOUSE_TAU), so equal finger speed gives equal camera speed.

   Two Thrones turns each buffered mouse event into camera input as x = trunc(x - dx * speed * 1.1) (FUN_004305c0;
   y the same with +), speed being the Mouse speed option (FUN_0042aa90: [0xa038a4] + [0x98108c]). With speed * 1.1
   below 1 every small event is truncated to nothing: slow swipes barely turned the camera and fast ones did, which
   felt stiff. So motion is kept in the game's own units here, fractions carried over, and each read sends the event
   that the game's truncation turns into exactly the whole units owed. */
#define MOUSE_TAU 0.03
static double g_mpend_x, g_mpend_y, g_mlast, g_gx, g_gy;
static pthread_mutex_t g_mlock = PTHREAD_MUTEX_INITIALIZER;
void dinput_mouse_move(int dx, int dy) {
    pthread_mutex_lock(&g_mlock);
    g_mpend_x += dx; g_mpend_y += dy;
    pthread_mutex_unlock(&g_mlock);
}
/* the game's factor from mouse counts to camera units (0: not Two Thrones, or not set up yet) */
static float game_mouse_scale(void) {
    float k = (RF32(0xa038a4u) + RF32(0x98108cu)) * RF32(0x980810u);
    if (!(k > 0.01f && k < 50.f) || RF32(0x980810u) != 1.1f) return 0;
    static float logged;
    if (k != logged) { logged = k; fprintf(stderr, "[dinput] game mouse scale %.3f (Mouse speed option %.3f)\n", k, RF32(0xa038a4u)); }
    return k;
}
/* the event the game truncates to exactly u units: the smallest count whose scaled value reaches |u| */
static int count_for(int u, float k) {
    if (!u) return 0;
    int a = u < 0 ? -u : u;
    int r = (int)ceil((a + 0.002) / k);
    while (r > 1 && (r - 1) * (double)k >= a + 0.002) r--;
    while (r * (double)k < a + 0.002) r++;
    return u < 0 ? -r : r;
}
/* debugging (measuring the camera's response): <game>/../mtest.txt holding "units frames" makes the next
   reads send exactly that many camera units each (X axis), touch motion ignored meanwhile */
static int mtest_step(int *dx, int *dy) {
    static int units, frames, axis;
    static double last_check;
    if (!frames && host_seconds() - last_check > 0.5) {
        last_check = host_seconds();
        char path[4200];
        snprintf(path, sizeof path, "%s/../mtest.txt", g_game_root);
        FILE *f = fopen(path, "r");
        if (f) {
            axis = 0;
            if (fscanf(f, "%d %d %d", &units, &frames, &axis) < 2) frames = 0;
            fclose(f); remove(path);
            fprintf(stderr, "[mtest] %d units x %d frames\n", units, frames);
        }
    }
    if (!frames) return 0;
    frames--;
    float k = game_mouse_scale();
    int r = k > 0 ? count_for(units, k) : units;
    *dx = axis ? 0 : r; *dy = axis ? r : 0;
    return 1;
}
/* In play the game reads the mouse like a stick: per frame its camera takes v = x * (speed * 4.5 + 0.5) / 50
   (y: y * (speed + 1) / 20) as a turning rate, and measured on the phone (RC mtest hook above, screen recordings)
   nothing turns below v ~0.47 and the rate is at its maximum from ~0.7: a narrow on/off band, so a drag either did
   nothing or spun the camera. Here finger motion becomes the turn wanted this frame, as a fraction of the maximum
   rate, sent as the v inside that band that gives it; more than the maximum carries over to the next frames, so a
   fast flick still turns as far as the finger went. RC_CAM="dead_x,full_x,dead_y,full_y,gain_x,gain_y" tunes it. */
static float g_cam[6] = { 0.47f, 0.70f, 0.47f, 0.70f, 0.11f, 0.11f };
static double g_qx, g_qy;   /* turn owed, in fractions of a frame at the maximum rate */
static int band_units(double *q, float dead, float full, float per_unit) {
    double s = *q > 1 ? 1 : *q < -1 ? -1 : *q;
    if (fabs(s) < 0.03) return 0;   /* too little for a frame yet: wait for more */
    *q -= s;
    double v = dead + fabs(s) * (full - dead);
    int u = (int)ceil(v / per_unit);
    return s < 0 ? -u : u;
}
/* Finger-follow turning: the camera's actual heading is read from the game's view matrix every frame
   (render_rec.c camera_constants: c8-c10 are the camera's axes, Z up), so a drag sets a target heading and each
   mouse read sends the turn rate that brings the camera there, braking early for the camera's own easing
   (it lags the input by several frames). The finger moves the view by a fixed angle per distance, it stops
   when the finger stops, and auto-follow resumes once the target is reached and the finger rests.
   Measured on the phone (Mouse speed option 0.44): units 10 turn ~0.3 deg/frame, 12 ~0.9, 13 ~2.2, 15+ ~2.45
   (the maximum); positive units raise the heading. Rates between levels are dithered (sigma-delta).
   RC_CAMF="gain,lag,deg_per_count" tunes it; RC_CAMF=off restores the rate band alone.
   If the heading doesn't respond (fixed-camera rooms) it hands back to the band mapping for a while. */
static float g_camf[3] = { 0.4f, 0.f, 0.22f };   /* tuned live 2026-11-15: no lag term (the camera answers within a read) */
static int g_yf_on = -1, g_yf_act, g_yf_stall;
static double g_yf_target, g_yf_prev, g_yf_motion_t, g_yf_off_until, g_yf_sd;
static double wrap180(double a) { a = fmod(a + 180.0, 360.0); if (a < 0) a += 360.0; return a - 180.0; }
/* tuning on the phone without restarting, files next to game/ (each read every 0.5 s):
   camf.txt "gain lag deg" or "off" (deleted once read); camlog.txt present = log every read as [yf];
   ftest.txt "counts reads" = a fake swipe: that many mouse counts added on each of the next reads */
/* rate curve: deg per read against camera value v (camlv.txt "v r v r ..." replaces it; r rising) */
static float g_lv_v[16] = { 0.53f, 0.54f, 0.56f, 0.58f, 0.60f, 0.62f, 0.68f, 0.72f, 0.80f },   /* measured, speed option 0 */
            g_lv_r[16] = { 0.f, 0.3f, 0.43f, 0.67f, 0.85f, 1.06f, 1.65f, 2.21f, 2.6f };
static int g_lv_n = 9;
static float g_force_sp = 0.f;   /* the Mouse speed option forced to this (fine input steps); speedopt.txt "x" changes it, -1 = leave */
static int g_yf_log, g_ft_reads, g_yf_direct = 0;   /* direct heading writes: off (the matrix found was the Prince's facing) */
static float g_ft_counts;
#define YM_MAX 8
static uint32_t g_ym[YM_MAX];   /* the yaw matrix matches (ym_search below) */
static int g_ym_n;
static void ym_search(double yaw);
static void yf_poll(void) {
    static double last;
    double now = host_seconds();
    if (now - last < 0.5) return;
    last = now;
    char p[4200];
    FILE *f;
    snprintf(p, sizeof p, "%s/../camf.txt", g_game_root);
    if ((f = fopen(p, "r"))) {
        char w[64] = "";
        if (fgets(w, sizeof w, f)) {
            if (!strncmp(w, "off", 3)) g_yf_on = 0;
            else { g_yf_on = 1; sscanf(w, "%f %f %f", &g_camf[0], &g_camf[1], &g_camf[2]); }
        }
        fclose(f); remove(p);
        fprintf(stderr, "[yf] on %d gain %g lag %g deg %g\n", g_yf_on, g_camf[0], g_camf[1], g_camf[2]);
    }
    snprintf(p, sizeof p, "%s/../camlv.txt", g_game_root);
    if ((f = fopen(p, "r"))) {
        int n = 0;
        while (n < 16 && fscanf(f, "%f %f", &g_lv_v[n], &g_lv_r[n]) == 2) n++;
        if (n >= 2) g_lv_n = n;
        fclose(f); remove(p);
        fprintf(stderr, "[yf] rate curve: %d points\n", g_lv_n);
    }
    snprintf(p, sizeof p, "%s/../speedopt.txt", g_game_root);
    if ((f = fopen(p, "r"))) { if (fscanf(f, "%f", &g_force_sp) != 1) g_force_sp = -1; fclose(f); remove(p); fprintf(stderr, "[yf] speed option forced to %g\n", g_force_sp); }
    snprintf(p, sizeof p, "%s/../directcam.txt", g_game_root);
    if ((f = fopen(p, "r"))) { if (fscanf(f, "%d", &g_yf_direct) != 1) g_yf_direct = 1; fclose(f); remove(p); fprintf(stderr, "[yf] direct %d\n", g_yf_direct); }
    snprintf(p, sizeof p, "%s/../ymtest.txt", g_game_root);
    if ((f = fopen(p, "r"))) {   /* "search" or "<index> <deg>": rotate one yaw-matrix match (finding which is the camera) */
        char w[32] = ""; int idx = -1; float deg = 0;
        if (fgets(w, sizeof w, f)) { if (!strncmp(w, "search", 6)) ym_search(g_yf_prev); else if (sscanf(w, "%d %f", &idx, &deg) == 2 && idx >= 0 && idx < g_ym_n) {
            float *m = (float *)HP(g_ym[idx]); double t = atan2(-m[0], m[1]) + deg * (M_PI / 180.0);
            m[0] = -(float)sin(t); m[1] = (float)cos(t); m[4] = -(float)cos(t); m[5] = -(float)sin(t);
            fprintf(stderr, "[yf] rotated match %d (%08x) by %g\n", idx, g_ym[idx], deg); } }
        fclose(f); remove(p);
    }
    snprintf(p, sizeof p, "%s/../camlog.txt", g_game_root);
    g_yf_log = access(p, F_OK) == 0;
    snprintf(p, sizeof p, "%s/../ftest.txt", g_game_root);
    if ((f = fopen(p, "r"))) {
        if (fscanf(f, "%f %d", &g_ft_counts, &g_ft_reads) != 2) g_ft_reads = 0;
        fclose(f); remove(p);
        fprintf(stderr, "[yf] fake swipe %g counts x %d reads\n", g_ft_counts, g_ft_reads);
    }
}
/* The camera's yaw matrix in the game's heap (found with host/memscan.c, 2026-11-16): four rows of 4 floats,
   (-sin y, cos y, 0, 0), (-cos y, -sin y, 0, 0), (0, 0, 1, 0), (camera position, 1), y being the heading read from
   the view (yaw_follow). Writing it turns the camera at once and the game carries on from there, so a drag sets
   the heading directly. It moves between sessions and levels, so it is found by matching the current heading
   (every match is written: copies are recomputed from it each frame anyway). directcam.txt "0"/"1" toggles it. */
static double g_ym_search_t, g_ym_written = 1e9;
static int ym_angle(uint32_t a, double *deg) {
    const float *m = (const float *)HP(a);
    if (m[2] != 0 || m[3] != 0 || m[6] != 0 || m[7] != 0 || m[8] != 0 || m[9] != 0 || m[10] != 1.f || m[11] != 0 || m[15] != 1.f) return 0;
    if (fabsf(m[4] + m[1]) > 1e-3f || fabsf(m[5] - m[0]) > 1e-3f || fabsf(m[0] * m[0] + m[1] * m[1] - 1.f) > 1e-3f) return 0;
    *deg = atan2(-m[0], m[1]) * (180.0 / M_PI);
    return 1;
}
static void ym_search(double yaw) {
    extern void guest_mem_ranges(uint32_t r[4]);
    uint32_t r[4];
    guest_mem_ranges(r);
    g_ym_n = 0;
    double t0 = host_seconds();
    for (int k = 0; k < 2; k++)
        for (uint32_t a = (r[2 * k] + 15) & ~15u; a + 64 <= r[2 * k + 1] && g_ym_n < YM_MAX; a += 16) {
            const float *m = (const float *)HP(a);
            if (m[10] != 1.f || m[8] != 0 || m[15] != 1.f) continue;
            double d;
            if (ym_angle(a, &d) && fabs(wrap180(d - yaw)) < 1.0) g_ym[g_ym_n++] = a;
        }
    fprintf(stderr, "[yf] yaw matrix search: %d found in %.0f ms%s%08x\n", g_ym_n, (host_seconds() - t0) * 1e3, g_ym_n ? ", first " : "", g_ym_n ? g_ym[0] : 0);
    for (int i = 0; i < g_ym_n; i++) { const float *m = (const float *)HP(g_ym[i]); fprintf(stderr, "[yf]   %d: %08x pos %.1f %.1f %.1f\n", i, g_ym[i], m[12], m[13], m[14]); }
}
static int yaw_matrix_write(double yaw, double target) {
    int ok = 0;
    for (int i = 0; i < g_ym_n; i++) {   /* still there: showing the heading, or what was written last */
        double d;
        if (ym_angle(g_ym[i], &d) && (fabs(wrap180(d - yaw)) < 3.0 || fabs(wrap180(d - g_ym_written)) < 3.0)) ok = 1;
    }
    if (!ok) {
        if (host_seconds() - g_ym_search_t < 1.0) return 0;
        g_ym_search_t = host_seconds();
        ym_search(yaw);
        if (!g_ym_n) return 0;
    }
    float s = (float)sin(target * (M_PI / 180.0)), c = (float)cos(target * (M_PI / 180.0));
    for (int i = 0; i < g_ym_n; i++) {
        float *m = (float *)HP(g_ym[i]);
        m[0] = -s; m[1] = c; m[4] = -c; m[5] = -s;
    }
    g_ym_written = target;
    return 1;
}
static int yaw_follow(double rx, int *ux) {
    if (g_yf_on < 0) {
        const char *e = getenv("RC_CAMF");
        /* off by default: the follow loop (heading feedback) was tuned to one phone's frame rate and lag, and on others
           (a Snapdragon 888 at ~35 fps) it overshot, so a tap set the camera swinging left and right for a second and
           play drifted. Left/right now uses the rate band like up/down; RC_CAMF=on (or gains) turns following back on. */
        g_yf_on = e && strcmp(e, "off") != 0;
        if (e && g_yf_on) sscanf(e, "%f,%f,%f", &g_camf[0], &g_camf[1], &g_camf[2]);
    }
    yf_poll();
    *ux = 0;
    extern int camera_constants(float out[12][4]);
    float c[12][4];
    camera_constants(c);
    if (c[10][0] == 0 && c[10][1] == 0) return 0;
    double yaw = atan2(c[10][1], c[10][0]) * (180.0 / M_PI);
    double vel = wrap180(yaw - g_yf_prev);
    g_yf_prev = yaw;
    if (!g_yf_on) return 0;
    double now = host_seconds();
    if (now < g_yf_off_until) { g_yf_act = 0; return 0; }
    if (rx != 0) {
        if (!g_yf_act) { g_yf_act = 1; g_yf_target = yaw; g_yf_stall = 0; g_yf_sd = 0; }
        g_yf_target += rx * g_camf[2];
        g_yf_motion_t = now;
    }
    if (!g_yf_act) return 1;
    if (g_yf_direct && yaw_matrix_write(yaw, g_yf_target)) {   /* the heading itself is set: flicks land at once */
        *ux = 0;
        if (now - g_yf_motion_t > 0.15) g_yf_act = 0;   /* finger resting: the game's camera carries on from here */
        return 1;
    }
    double err = wrap180(g_yf_target - yaw);
    if (err > 90) { g_yf_target = yaw + 90; err = 90; }   /* never more than a quarter turn behind the finger */
    if (err < -90) { g_yf_target = yaw - 90; err = -90; }
    double rest = now - g_yf_motion_t;
    if ((rest > 0.12 && fabs(err) < 1.0 && fabs(vel) < 0.3) || rest > 1.5) { g_yf_act = 0; return 1; }   /* there: let go */
    double want = g_camf[0] * (err - vel * g_camf[1]);   /* deg/frame, braking for the camera's lag */
    /* the camera value v (x * (speed * 4.5 + 0.5) / 50) giving that rate: the measured curve, interpolated */
    double a = fabs(want);
    if (a < 0.05) a = 0;
    double v = 0;
    if (a > 0) {
        int i = 1;
        while (i < g_lv_n - 1 && g_lv_r[i] < a) i++;
        double t = (a - g_lv_r[i - 1]) / (g_lv_r[i] - g_lv_r[i - 1]);
        if (t > 1) t = 1;
        if (t < 0) t = 0;
        v = g_lv_v[i - 1] + t * (g_lv_v[i] - g_lv_v[i - 1]);
    }
    float sp = RF32(0xa038a4u);
    double uf = v / ((sp * 4.5 + 0.5) / 50.0);
    g_yf_sd += uf - floor(uf);   /* whole units only: carry the fraction (fine steps with the speed option at 0) */
    int u = (int)floor(uf);
    if (g_yf_sd >= 1) { g_yf_sd -= 1; u++; }
    if (v == 0) u = 0;
    /* a camera that doesn't turn while being pushed is fixed here: give the drag back to the old mapping for 2 s */
    if (a > 1.0 && fabs(vel) < 0.05) { if (++g_yf_stall > 25) { g_yf_off_until = now + 2.0; g_yf_act = 0; return 0; } }
    else if (fabs(vel) > 0.2) g_yf_stall = 0;
    *ux = want < 0 ? -u : u;
    return 1;
}

static void camera_units(double rx, double ry, float k, int *dx, int *dy) {
    static int init;
    if (!init) {
        init = 1;
        const char *e = getenv("RC_CAM");
        if (e) sscanf(e, "%f,%f,%f,%f,%f,%f", &g_cam[0], &g_cam[1], &g_cam[2], &g_cam[3], &g_cam[4], &g_cam[5]);
    }
    float sp = RF32(0xa038a4u);
    int ux;
    if (g_force_sp >= 0 && RF32(0xa038a4u) != g_force_sp) memcpy(g_mem + 0xa038a4u, &g_force_sp, 4);   /* fine input steps */
    if (g_ft_reads > 0) { rx += g_ft_counts; g_ft_reads--; }   /* fake swipe (yf_poll) */
    int follow = yaw_follow(rx, &ux);
    if (follow) g_qx = 0;   /* the camera's angle follows the finger (above) */
    else {
        g_qx += rx * g_cam[4];
        ux = band_units(&g_qx, g_cam[0], g_cam[1], (sp * 4.5f + 0.5f) / 50.f);
    }
    if (g_yf_log) fprintf(stderr, "[yf] %.3f rx %.1f yaw %.2f tgt %.2f act %d follow %d ux %d\n", host_seconds(), rx, g_yf_prev, g_yf_target, g_yf_act, follow, ux);
    g_qy += ry * g_cam[5];
    int uy = band_units(&g_qy, g_cam[2], g_cam[3], (sp + 1.f) / 20.f);
    *dx = count_for(ux, k); *dy = count_for(uy, k);
    g_gx = g_gy = 0;
}

static void mouse_release(void) {
    {
        int tdx, tdy, on = mtest_step(&tdx, &tdy);
        if (g_yf_log) {   /* the heading at every read, for step responses */
            extern int camera_constants(float out[12][4]);
            float c[12][4];
            camera_constants(c);
            fprintf(stderr, "[yfm] %.3f yaw %.2f mtest %d\n", host_seconds(), atan2(c[10][1], c[10][0]) * (180.0 / M_PI), on ? tdx : 0);
        }
        if (on) { mouse_deliver(tdx, tdy); return; }
    }
    pthread_mutex_lock(&g_mlock);
    double now = host_seconds(), dt = g_mlast ? now - g_mlast : 0.03;
    g_mlast = now;
    double f = dt > 0.2 ? 1.0 : 1.0 - exp(-dt / MOUSE_TAU);
    double rx = g_mpend_x * f, ry = g_mpend_y * f;
    if (fabs(g_mpend_x) < 0.5 && fabs(g_mpend_y) < 0.5) rx = g_mpend_x, ry = g_mpend_y;   /* the tail */
    g_mpend_x -= rx; g_mpend_y -= ry;
    int dx, dy;
    float k = game_mouse_scale();
    extern int gamephase_current(void);
    if (k > 0 && gamephase_current() == 0) camera_units(rx, ry, k, &dx, &dy);
    else if (k > 0) {
        g_qx = g_qy = 0;
        g_gx += rx * k; g_gy += ry * k;
        int ux = (int)g_gx, uy = (int)g_gy;   /* whole units, toward zero; the rest waits */
        g_gx -= ux; g_gy -= uy;
        dx = count_for(ux, k); dy = count_for(uy, k);
    } else {
        g_gx += rx; g_gy += ry;
        dx = (int)g_gx; dy = (int)g_gy;
        g_gx -= dx; g_gy -= dy;
    }
    pthread_mutex_unlock(&g_mlock);
    if (dx || dy) mouse_deliver(dx, dy);
}
#else
void dinput_mouse_move(int dx, int dy) { mouse_deliver(dx, dy); }
static void mouse_release(void) {}
#endif

static ComInterface IDirectInputDevice_i;

/* ------------------------------------------------------------------ joystick objects */

typedef struct { uint32_t ofs; uint8_t guid0; uint32_t type; const char *name; uint16_t page, usage; } PadObj;
/* guid0: last byte of the low word of DirectInput's object GUIDs {A36D02xx-C9F3-11CF-BFC7-444553540000} */
static const PadObj g_pad_axes[] = {
    {  0, 0xE0, 2u | 0u << 8, "X Axis", 1, 0x30 },        /* DIDFT_ABSAXIS | instance */
    {  4, 0xE1, 2u | 1u << 8, "Y Axis", 1, 0x31 },
    {  8, 0xE2, 2u | 2u << 8, "Z Axis", 1, 0x32 },
    { 20, 0xE3, 2u | 3u << 8, "Z Rotation", 1, 0x35 },
    { 32, 0xF2, 0x10u | 0u << 8, "Hat Switch", 1, 0x39 }, /* DIDFT_POV */
};
#define PAD_NOBJ (5 + PAD_BUTTONS)
static PadObj pad_obj(int i) {
    if (i < 5) return g_pad_axes[i];
    static char names[PAD_BUTTONS][16];
    int b = i - 5;
    snprintf(names[b], sizeof names[b], "Button %d", b + 1);
    return (PadObj){ 48u + (uint32_t)b, 0xF0, 4u | (uint32_t)b << 8 /* DIDFT_PSHBUTTON */, names[b], 9, (uint16_t)(b + 1) };
}
static void write_object_instance(uint32_t p, PadObj o) {
    uint32_t sz = R32(p);
    memset(HP(p + 4), 0, sz > 4 ? sz - 4 : 0);
    static const uint8_t base[16] = { 0xE0, 0x02, 0x6D, 0xA3, 0xF3, 0xC9, 0xCF, 0x11, 0xBF, 0xC7, 0x44, 0x45, 0x53, 0x54, 0x00, 0x00 };
    memcpy(HP(p + 4), base, 16);
    W8(p + 4, o.guid0);
    W32(p + 20, o.ofs);
    W32(p + 24, o.type);
    snprintf((char *)HP(p + 32), 260, "%s", o.name);
    if (sz >= 308) { W16(p + 304, o.page); W16(p + 306, o.usage); }   /* DirectX 5+ fields */
}
/* Axis slot (DIJOYSTATE offset / 4) addressed by a property header's dwObj/dwHow; -1 = whole device. */
static int pad_axis_slot(uint32_t obj, uint32_t how) {
    if (how == 0) return -1;                                      /* DIPH_DEVICE */
    if (how == 1) return obj < 24 && obj % 4 == 0 ? (int)(obj / 4) : -2;   /* DIPH_BYOFFSET */
    if (how == 2) {                                               /* DIPH_BYID */
        if ((obj & 3) == 0) return -2;
        static const int inst_slot[4] = { 0, 1, 2, 5 };
        uint32_t inst = (obj >> 8) & 0xFFFF;
        return inst < 4 ? inst_slot[inst] : -2;
    }
    return -2;
}
static int32_t pad_axis_value(const DIDev *d, int slot, int32_t raw) {
    int32_t mag = raw < 0 ? -raw : raw;
    if (d->dead[slot] > 0 && (int64_t)mag * 10000 / 32768 < d->dead[slot]) raw = 0;
    return d->lo[slot] + (int32_t)(((int64_t)raw + 32768) * ((int64_t)d->hi[slot] - d->lo[slot]) / 65535);
}
/* W3C standard gamepad buttons -> the Dual Action's DirectInput buttons (controltypes.csv order) */
static const int g_std_for_di[PAD_BUTTONS] = { 2, 0, 1, 3, 4, 5, 6, 7, 9, 8, 10, 11 };
static void pad_write_state(const DIDev *d, uint32_t p, uint32_t n) {
    PadState st;
    pthread_mutex_lock(&g_di_lock_pads);
    st = g_pads[d->pad];
    pthread_mutex_unlock(&g_di_lock_pads);
    uint8_t buf[272];
    memset(buf, 0, sizeof buf);
    int32_t v[6] = { st.axis[0], st.axis[1], st.axis[2], 0, 0, st.axis[3] };
    /* The game takes digital directions for this pad type from the left stick and ignores the POV hat,
       so the d-pad also deflects the left stick fully. */
    if (st.buttons >> 12 & 1) v[1] = -32768;
    if (st.buttons >> 13 & 1) v[1] = 32767;
    if (st.buttons >> 14 & 1) v[0] = -32768;
    if (st.buttons >> 15 & 1) v[0] = 32767;
    for (int k = 0; k < 6; k++) { int32_t a = pad_axis_value(d, k, v[k]); memcpy(buf + 4 * k, &a, 4); }
    uint32_t pov = 0xFFFFFFFFu;
    int up = st.buttons >> 12 & 1, down = st.buttons >> 13 & 1, left = st.buttons >> 14 & 1, right = st.buttons >> 15 & 1;
    if (up || down || left || right) {
        int dx = right - left, dy = down - up;
        static const uint32_t dirs[3][3] = { { 31500, 0, 4500 }, { 27000, 0xFFFFFFFFu, 9000 }, { 22500, 18000, 13500 } };
        pov = dirs[dy + 1][dx + 1];
    }
    memcpy(buf + 32, &pov, 4);
    memset(buf + 36, 0xFF, 12);   /* POVs 1-3 centred */
    for (int b = 0; b < PAD_BUTTONS; b++) buf[48 + b] = (st.buttons >> g_std_for_di[b] & 1) ? 0x80 : 0;
    memcpy(HP(p), buf, n < sizeof buf ? n : sizeof buf);
}

/* ------------------------------------------------------------------ IDirectInputDevice */

W(dev_GetCapabilities) {
    DIDev *d = com_data(THIS);
    uint32_t p = ARG(1);
    uint32_t sz = R32(p);
    memset(HP(p + 4), 0, sz > 4 ? sz - 4 : 0);
    W32(p + 4, 1 /* DIDC_ATTACHED */);
    if (d && d->kind == DEV_JOYSTICK) {
        W32(p + 4, 1 | 2 /* DIDC_ATTACHED | DIDC_POLLEDDEVICE */);
        W32(p + 8, PAD_DEVTYPE); W32(p + 12, 4); W32(p + 16, PAD_BUTTONS); W32(p + 20, 1);
        RET(2, S_OK);
    }
    W32(p + 8, d && d->kind == DEV_KEYBOARD ? 0x0403 : 0x0202);  /* dev type */
    W32(p + 12, d && d->kind == DEV_KEYBOARD ? 0 : 3);            /* axes */
    W32(p + 16, d && d->kind == DEV_KEYBOARD ? 128 : 4);          /* buttons */
    RET(2, S_OK);
}
W(dev_EnumObjects) {
    DIDev *d = com_data(THIS);
    uint32_t cb = ARG(1), ref = ARG(2), flags = ARG(3) & 0xFF;
    if (!d || d->kind != DEV_JOYSTICK) RET(4, S_OK);
    uint32_t p = heap_alloc(316, 1);
    for (int i = 0; i < PAD_NOBJ; i++) {
        PadObj o = pad_obj(i);
        if (flags && !(o.type & flags)) continue;   /* DIDFT_ALL = 0, else DIDFT_AXIS / BUTTON / POV masks */
        W32(p, 316);
        write_object_instance(p, o);
        uint32_t a[2] = { p, ref };
        if (!guest_stdcall(c, cb, 2, a)) break;     /* DIENUM_STOP */
    }
    heap_free(p);
    RET(4, S_OK);
}
/* Property GUIDs are small integers in DirectInput (MAKEDIPROPERTY): 4 = RANGE, 5 = DEADZONE. */
W(dev_GetProperty) {
    DIDev *d = com_data(THIS);
    uint32_t prop = ARG(1), p = ARG(2);
    if (d && d->kind == DEV_JOYSTICK && (prop == 4 || prop == 5)) {
        int slot = pad_axis_slot(R32(p + 8), R32(p + 12));
        if (slot == -2) RET(3, 0x80070002u /* DIERR_OBJECTNOTFOUND */);
        if (slot < 0) slot = 0;
        if (prop == 4) { W32(p + 16, (uint32_t)d->lo[slot]); W32(p + 20, (uint32_t)d->hi[slot]); }
        else W32(p + 16, (uint32_t)d->dead[slot]);
        RET(3, S_OK);
    }
    RET(3, 0x80070057u /* E_INVALIDARG -> use defaults */);
}
W(dev_SetProperty) {
    DIDev *d = com_data(THIS);
    uint32_t prop = ARG(1), p = ARG(2);
    if (d && d->kind == DEV_JOYSTICK && (prop == 4 || prop == 5)) {
        int slot = pad_axis_slot(R32(p + 8), R32(p + 12));
        if (slot == -2) RET(3, 0x80070002u);
        for (int k = 0; k < 6; k++) {
            if (slot >= 0 && k != slot) continue;
            if (prop == 4) { d->lo[k] = (int32_t)R32(p + 16); d->hi[k] = (int32_t)R32(p + 20); }
            else d->dead[k] = (int32_t)R32(p + 16);
        }
        LOG("[dinput] pad %d %s(slot %d) = %d %d\n", d->pad, prop == 4 ? "RANGE" : "DEADZONE", slot,
            (int)R32(p + 16), prop == 4 ? (int)R32(p + 20) : 0);
    }
    RET(3, S_OK);
}
W(dev_Acquire) { DIDev *d = com_data(THIS); if (d) d->acquired = 1; LOG("[dinput] Acquire\n"); RET(1, S_OK); }
W(dev_Unacquire) { DIDev *d = com_data(THIS); if (d) d->acquired = 0; RET(1, S_OK); }
W(dev_GetDeviceState) {
    DIDev *d = com_data(THIS);
    { static int n; if (n++ < 2) LOG("[dinput] GetDeviceState(%s, %u bytes)\n", d && d->kind == DEV_KEYBOARD ? "keyboard" : "mouse", ARG(1)); }
    uint32_t n = ARG(1), p = ARG(2);
    memset(HP(p), 0, n);
    if (d && d->kind == DEV_KEYBOARD) {
        if (deterministic()) {   /* deterministic runs: state is exactly what was pressed */
            for (uint32_t i = 0; i < n && i < 256; i++) W8(p + i, g_dik[i] ? 0x80 : 0);
        } else {
            kq_poll();
            for (uint32_t i = 0; i < n && i < 256; i++) W8(p + i, g_kvis[i] ? 0x80 : 0);
        }
    } else if (d && d->kind == DEV_JOYSTICK) {
        pad_write_state(d, p, n);
    } else if (d && d->kind == DEV_MOUSE && n >= 16) {
        mouse_release();
        { static int k; if (k++ < 3) fprintf(stderr, "[dinput] mouse GetDeviceState from %08x into %08x\n", R32(c->esp), p); }
        W32(p, (uint32_t)g_mouse_dx); W32(p + 4, (uint32_t)g_mouse_dy);
        g_mouse_dx = g_mouse_dy = 0;
        double now = host_seconds();
        for (int b = 0; b < 4; b++)
            W8(p + 12 + b, ((g_mouse_buttons >> b) & 1) || now - g_button_down_at[b] < MIN_HOLD ? 0x80 : 0);
    }
    RET(3, S_OK);
}
W(dev_GetDeviceData) {
    DIDev *d = com_data(THIS);
    { static int n; if (n++ < 2) LOG("[dinput] GetDeviceData(%s)\n", d && d->kind == DEV_KEYBOARD ? "keyboard" : "mouse"); }
    uint32_t size = ARG(1), out = ARG(2), pn = ARG(3), peek = ARG(4) & 1;
    int kind = d ? d->kind : 0;
    if (kind == DEV_MOUSE && !peek) mouse_release();
    if (kind == DEV_MOUSE) { static int k; if (k++ < 3) fprintf(stderr, "[dinput] mouse GetDeviceData from %08x into %08x size %u\n", R32(c->esp), out, size); }
    uint32_t max = pn ? R32(pn) : 0, n = 0;
    pthread_mutex_lock(&g_di_lock);
    int h = g_qhead[kind];
    while (h != g_qtail[kind] && n < max) {
        if (out) {
            memset(HP(out + n * size), 0, size);
            memcpy(HP(out + n * size), &g_queue[kind][h], size < 16 ? size : 16);
        }
        h = (h + 1) % 256;
        n++;
    }
    if (!peek) g_qhead[kind] = h;
    pthread_mutex_unlock(&g_di_lock);
    if (pn) W32(pn, n);
    RET(5, S_OK);
}
W(dev_SetDataFormat) {
    DIDev *d = com_data(THIS);
    if (d) d->format_size = R32(ARG(1) + 12);
    /* Joysticks: only the standard c_dfDIJoystick / c_dfDIJoystick2 layouts are supported */
    if (d && d->kind == DEV_JOYSTICK) {
        LOG("[dinput] pad %d SetDataFormat(%u bytes, %u objects)\n", d->pad, d->format_size, R32(ARG(1) + 16));
        if (d->format_size != 80 && d->format_size != 272)
            fprintf(stderr, "[dinput] unsupported joystick data format (%u bytes)\n", d->format_size);
    }
    RET(2, S_OK);
}
W(dev_SetEventNotification) { RET(2, S_OK); }
W(dev_SetCooperativeLevel) { RET(3, S_OK); }
static void guid_axis(uint8_t *g, int n) {   /* GUID_XAxis/YAxis/ZAxis = {A36D02E0+n-C9F3-11CF-BFC7-444553540000} */
    static const uint8_t base[16] = { 0xE0, 0x02, 0x6D, 0xA3, 0xF3, 0xC9, 0xCF, 0x11, 0xBF, 0xC7, 0x44, 0x45, 0x53, 0x54, 0x00, 0x00 };
    memcpy(g, base, 16);
    g[0] = (uint8_t)(0xE0 + n);
}
W(dev_GetObjectInfo) {
    DIDev *d = com_data(THIS);
    uint32_t p = ARG(1), obj = ARG(2), how = ARG(3);
    uint32_t sz = R32(p);
    LOG("[dinput] GetObjectInfo(%s, obj %u, how %u)\n", !d ? "?" : d->kind == DEV_KEYBOARD ? "keyboard" : d->kind == DEV_JOYSTICK ? "gamepad" : "mouse", obj, how);
    uint32_t ofs = obj;
    if (d && d->kind == DEV_JOYSTICK) {
        for (int i = 0; i < PAD_NOBJ; i++) {
            PadObj o = pad_obj(i);
            if ((how == 1 && o.ofs == obj) || (how == 2 && (o.type & 0xFFFF1F) == (obj & 0xFFFF1F))) {
                write_object_instance(p, o);
                RET(4, S_OK);
            }
        }
        RET(4, 0x80070002u);
    }
    if (how != 1 /* DIPH_BYOFFSET */) RET(4, 0x80070002u /* DIERR_OBJECTNOTFOUND */);
    memset(HP(p + 4), 0, sz > 4 ? sz - 4 : 0);
    uint8_t guid[16];
    uint32_t type;
    char name[32];
    if (d && d->kind == DEV_MOUSE) {
        if (ofs <= 8 && ofs % 4 == 0) {
            guid_axis(guid, (int)(ofs / 4));
            type = 1u | ((ofs / 4) << 8);                   /* DIDFT_RELAXIS | instance */
            snprintf(name, sizeof name, "%c Axis", "XYZ"[ofs / 4]);
        } else if (ofs >= 12 && ofs < 16) {
            static const uint8_t button[16] = { 0xF0, 0x02, 0x6D, 0xA3, 0xF3, 0xC9, 0xCF, 0x11, 0xBF, 0xC7, 0x44, 0x45, 0x53, 0x54, 0x00, 0x00 };
            memcpy(guid, button, 16);
            type = 4u | ((ofs - 12) << 8);                  /* DIDFT_PSHBUTTON | instance */
            snprintf(name, sizeof name, "Button %u", ofs - 12);
        } else RET(4, 0x80070002u);
    } else {
        if (ofs > 255) RET(4, 0x80070002u);
        static const uint8_t key[16] = { 0x20, 0x82, 0x72, 0x55, 0x3C, 0xD3, 0xCF, 0x11, 0xBF, 0xC7, 0x44, 0x45, 0x53, 0x54, 0x00, 0x00 };
        memcpy(guid, key, 16);
        type = 0xC | (ofs << 8);                            /* DIDFT_BUTTON | instance */
        snprintf(name, sizeof name, "Key %u", ofs);
    }
    memcpy(HP(p + 4), guid, 16);
    W32(p + 20, ofs);
    W32(p + 24, type);
    snprintf((char *)HP(p + 32), 260, "%s", name);
    RET(4, S_OK);
}
W(dev_GetDeviceInfo) {
    DIDev *d = com_data(THIS);
    uint32_t p = ARG(1), sz = R32(p);
    memset(HP(p + 4), 0, sz > 4 ? sz - 4 : 0);
    if (d && d->kind == DEV_JOYSTICK) {
        pad_guid(HP(p + 4), 1, d->pad);
        pad_guid(HP(p + 20), 0, d->pad);
        W32(p + 36, PAD_DEVTYPE);
        strcpy((char *)HP(p + 40), PAD_NAME);
        strcpy((char *)HP(p + 300), PAD_NAME);
        if (sz >= 580) { W16(p + 576, 1); W16(p + 578, 5); }   /* HID usage page 1, usage 5 = game pad */
        RET(2, S_OK);
    }
    int kb = d && d->kind == DEV_KEYBOARD;
    memcpy(HP(p + 4), kb ? GUID_SysKeyboard : GUID_SysMouse, 16);
    memcpy(HP(p + 20), kb ? GUID_SysKeyboard : GUID_SysMouse, 16);
    W32(p + 36, kb ? 0x0403 : 0x0202);
    strcpy((char *)HP(p + 40), kb ? "Keyboard" : "Mouse");
    strcpy((char *)HP(p + 300), kb ? "Keyboard" : "Mouse");
    RET(2, S_OK);
}
W(dev_RunControlPanel) { RET(3, S_OK); }
W(dev_Initialize) { RET(4, S_OK); }
W(dev_Poll) { RET(1, S_OK); }
#define DIERR_UNSUPPORTED 0x80004001u
W(dev_CreateEffect) { if (ARG(3)) W32(ARG(3), 0); RET(5, DIERR_UNSUPPORTED); }
W(dev_EnumEffects) { RET(4, S_OK); }
W(dev_GetEffectInfo) { RET(3, DIERR_UNSUPPORTED); }
W(dev_GetForceFeedbackState) { RET(2, DIERR_UNSUPPORTED); }
W(dev_SendForceFeedbackCommand) { RET(2, DIERR_UNSUPPORTED); }
W(dev_EnumCreatedEffectObjects) { RET(4, S_OK); }

static const ComMethod didev_methods[] = {
    { "QueryInterface", 2, com_QueryInterface }, { "AddRef", 0, com_AddRef }, { "Release", 0, com_Release },
    { "GetCapabilities", 1, dev_GetCapabilities }, { "EnumObjects", 3, dev_EnumObjects },
    { "GetProperty", 2, dev_GetProperty }, { "SetProperty", 2, dev_SetProperty }, { "Acquire", 0, dev_Acquire },
    { "Unacquire", 0, dev_Unacquire }, { "GetDeviceState", 2, dev_GetDeviceState },
    { "GetDeviceData", 4, dev_GetDeviceData }, { "SetDataFormat", 1, dev_SetDataFormat },
    { "SetEventNotification", 1, dev_SetEventNotification }, { "SetCooperativeLevel", 2, dev_SetCooperativeLevel },
    { "GetObjectInfo", 3, dev_GetObjectInfo }, { "GetDeviceInfo", 1, dev_GetDeviceInfo },
    { "RunControlPanel", 2, dev_RunControlPanel }, { "Initialize", 3, dev_Initialize },
    /* IDirectInputDevice2 / 7 */
    /* no force feedback: the pads don't report DIDC_FORCEFEEDBACK */
    { "CreateEffect", 4, dev_CreateEffect }, { "EnumEffects", 3, dev_EnumEffects }, { "GetEffectInfo", 2, dev_GetEffectInfo },
    { "GetForceFeedbackState", 1, dev_GetForceFeedbackState }, { "SendForceFeedbackCommand", 1, dev_SendForceFeedbackCommand },
    { "EnumCreatedEffectObjects", 3, dev_EnumCreatedEffectObjects }, { "Escape", 1, NULL }, { "Poll", 0, dev_Poll },
    { "SendDeviceData", 4, NULL }, { "EnumEffectsInFile", 4, NULL }, { "WriteEffectToFile", 4, NULL },
};
static ComInterface IDirectInputDevice_i = { "IDirectInputDeviceA", didev_methods, 29, 0 };

/* ------------------------------------------------------------------ IDirectInput */

static uint32_t create_device(const uint8_t *guid) {
    DIDev *d = calloc(1, sizeof *d);
    d->kind = !memcmp(guid, GUID_SysKeyboard, 16) ? DEV_KEYBOARD : !memcmp(guid, GUID_SysMouse, 16) ? DEV_MOUSE : 0;
    int pad = d->kind ? -1 : pad_from_guid(guid);
    if (pad >= 0) {
        d->kind = DEV_JOYSTICK;
        d->pad = pad;
        for (int k = 0; k < 6; k++) { d->lo[k] = 0; d->hi[k] = 65535; }   /* DirectInput's default range */
    }
    if (!d->kind) { free(d); return 0; }
    if (pad >= 0) LOG("[dinput] CreateDevice(gamepad %d)\n", pad);
    else LOG("[dinput] CreateDevice(%s)\n", d->kind == DEV_KEYBOARD ? "keyboard" : "mouse");
    return com_new(&IDirectInputDevice_i, 0, d);
}

W(di_CreateDevice) {
    uint32_t dev = create_device(HP(ARG(1)));
    W32(ARG(2), dev);
    RET(4, dev ? S_OK : 0x80040154u /* DIERR_DEVICENOTREG */);
}
W(di_CreateDeviceEx) {
    uint32_t dev = create_device(HP(ARG(1)));
    W32(ARG(3), dev);
    RET(5, dev ? S_OK : 0x80040154u);
}

/* Returns 0 when the callback asks to stop (DIENUM_STOP). */
static int enum_one(CPU *c, uint32_t cb, uint32_t ref, const uint8_t *guid, const uint8_t *product, uint32_t type,
                    const char *name) {
    uint32_t p = heap_alloc(580, 1);
    W32(p, 580);
    memcpy(HP(p + 4), guid, 16);
    memcpy(HP(p + 20), product, 16);
    W32(p + 36, type);
    strcpy((char *)HP(p + 40), name);
    strcpy((char *)HP(p + 300), name);
    if (type == PAD_DEVTYPE) { W16(p + 576, 1); W16(p + 578, 5); }
    uint32_t a[2] = { p, ref };
    uint32_t r = guest_stdcall(c, cb, 2, a);
    heap_free(p);
    return r != 0;
}
W(di_EnumDevices) {
    uint32_t type = ARG(1), cb = ARG(2), ref = ARG(3);
    LOG("[dinput] EnumDevices(type %u)\n", type);
    /* same order as Windows: system mouse first, then keyboard (the game's controller slots follow it) */
    uint32_t flags = ARG(4);
    if ((type == 0 || type == 2) && !enum_one(c, cb, ref, GUID_SysMouse, GUID_SysMouse, 0x0202, "Mouse")) RET(5, S_OK);
    if ((type == 0 || type == 3) && !enum_one(c, cb, ref, GUID_SysKeyboard, GUID_SysKeyboard, 0x0403, "Keyboard")) RET(5, S_OK);
    if ((type == 0 || type == 4) && !(flags & 0x100 /* DIEDFL_FORCEFEEDBACK */))
        for (int i = 0; i < num_pads(); i++) {
            uint8_t inst[16], prod[16];
            pad_guid(inst, 1, i);
            pad_guid(prod, 0, i);
            if (!enum_one(c, cb, ref, inst, prod, PAD_DEVTYPE, PAD_NAME)) break;
        }
    RET(5, S_OK);
}
W(di_GetDeviceStatus) { RET(2, S_OK); }
W(di_RunControlPanel) { RET(3, S_OK); }
W(di_Initialize) { RET(3, S_OK); }
W(di_FindDevice) { RET(4, 0x80070002u); }

static const ComMethod di_methods[] = {
    { "QueryInterface", 2, com_QueryInterface }, { "AddRef", 0, com_AddRef }, { "Release", 0, com_Release },
    { "CreateDevice", 3, di_CreateDevice }, { "EnumDevices", 4, di_EnumDevices },
    { "GetDeviceStatus", 1, di_GetDeviceStatus }, { "RunControlPanel", 2, di_RunControlPanel },
    { "Initialize", 2, di_Initialize }, { "FindDevice", 3, di_FindDevice }, { "CreateDeviceEx", 4, di_CreateDeviceEx },
};
static ComInterface IDirectInput_i = { "IDirectInputA", di_methods, 10, 0 };

void w_DirectInputCreateA(CPU *c) {
    LOG("[dinput] DirectInputCreateA(version %x)\n", ARG(1));
    W32(ARG(2), com_new(&IDirectInput_i, 0, NULL));
    RET(4, S_OK);
}

/* DirectInput 8: IDirectInput8A starts with the same methods as IDirectInput7A, and so do its devices */
void w_DirectInput8Create(CPU *c) {   /* (hinst, version, riid, ppv, outer) */
    LOG("[dinput] DirectInput8Create(version %x)\n", ARG(1));
    W32(ARG(3), com_new(&IDirectInput_i, 0, NULL));
    RET(5, S_OK);
}
