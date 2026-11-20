#include <stdlib.h>
/* Game phase for the Android touch overlay (android/.../TouchOverlay.java picks its layout from it):
 *   2 cut-scene   a Bink movie is open (logos, loading screens, cinematics)
 *   1 menu        the game's open UI pages are all menu pages (ui_menu below)
 *   0 gameplay    anything else
 * render_rec.c hands over every frame's 2D (UI) textures as content fingerprints, with the screen position
 * of each one's first vertex (back-buffer pixels); RC_LOG_UITEX=1 logs them for finding new signals.
 *
 * Menus are driven by that cursor, which the game moves by relative mouse motion. A tap on a menu item
 * (gamephase_menu_tap) steers the cursor there in closed loop, using the cursor's drawn position, and
 * clicks once it is on the spot. */
#include "win32.h"
#include <math.h>

int bink_movie_open(void);   /* host/bink.c */
void input_mouse_move(int dx, int dy);
void input_mouse_button(int b, int down);
uint32_t host_video_width(void), host_video_height(void);

#define CURSOR_TEX 0xb598ffc5u
#define CURSOR_HOT_X 2.f      /* the dagger's tip, from the sprite's top-left corner */
#define CURSOR_HOT_Y 3.f

static volatile int g_cursor_frames = 1000;   /* frames since the cursor was last drawn (large = not in a menu) */
static float g_cx, g_cy;                      /* its tip, back-buffer pixels */

/* a tap to reach: written by the UI thread, steered on the game thread */
static volatile int g_seek;   /* 1 seeking, 2 button down (released the next frame) */
static float g_tx, g_ty;
static int g_seek_frames, g_settle;
static float g_last_cx, g_last_cy;

static int ui_menu(void);

static void seek_step(void) {
    if (g_seek == 2) { input_mouse_button(0, 0); g_seek = 0; return; }
    if (g_seek != 1) return;
    g_seek_frames++;
    if (!ui_menu() || g_cursor_frames > 30 || g_seek_frames > 90) { g_seek = 0; return; }   /* left the menu, or gave up (~1.5 s) */
    float ex = g_tx - g_cx, ey = g_ty - g_cy;
    if (ex * ex + ey * ey < 5.f * 5.f || g_seek_frames > 75) {   /* there (or close enough): click */
        input_mouse_button(0, 1);
        g_seek = 2;
        return;
    }
    /* after a move, wait for the cursor to stop (the motion is eased in over a few reads: host/dinput.c) */
    if (g_settle > 0) {
        int still = g_cx == g_last_cx && g_cy == g_last_cy;
        g_last_cx = g_cx; g_last_cy = g_cy;
        if (!still ? --g_settle > 0 : --g_settle > 4) return;
    }
    /* pixels per mouse count depend on the render size and the game's mouse speed (about 1 at 540p, more above), so
       a fixed gain overshot and swung back and forth at 720p: measure it from the last move and aim at 0.8 of the way */
    static float ratio = 1.f;
    static int sent_x, sent_y;
    static float from_x, from_y;
    if (g_seek_frames > 1 && (abs(sent_x) >= 4 || abs(sent_y) >= 4)) {
        float moved = sqrtf((g_cx - from_x) * (g_cx - from_x) + (g_cy - from_y) * (g_cy - from_y));
        float sent = sqrtf((float)(sent_x * sent_x + sent_y * sent_y));
        float r = moved / sent;
        if (r > 0.2f && r < 8.f) ratio = r;
    }
    sent_x = (int)lroundf(ex * 0.8f / ratio); sent_y = (int)lroundf(ey * 0.8f / ratio);
    from_x = g_cx; from_y = g_cy;
    input_mouse_move(sent_x, sent_y);
    g_settle = 8;
    g_last_cx = g_cx; g_last_cy = g_cy;
}

void gamephase_frame(const uint32_t *hashes, const float *xy, uint32_t n) {
    int seen = 0;
    for (uint32_t i = 0; i < n; i++)
        if (hashes[i] == CURSOR_TEX && xy[2 * i] > -64.f && xy[2 * i] < 4096.f && xy[2 * i + 1] > -64.f && xy[2 * i + 1] < 4096.f) {
            /* (where the game draws it: in play too, parked off the picture or on its edge) */
            g_cx = xy[2 * i] + CURSOR_HOT_X; g_cy = xy[2 * i + 1] + CURSOR_HOT_Y; seen = 1; break;
        }
    g_cursor_frames = seen ? 0 : g_cursor_frames < 1000 ? g_cursor_frames + 1 : 1000;
    static int log_pages = -1;
    if (log_pages < 0) log_pages = getenv("RC_LOG_PAGES") != NULL;
    if (log_pages) {   /* debugging: the game's UI page list (FUN_004305c0 / FUN_0040b970) */
        static double last;
        if (host_seconds() - last > 2.0) {
            last = host_seconds();
            uint32_t base = R32(0xe6d83cu), b = base ? R32(base + 0x14) : 0, e = base ? R32(base + 0x18) : 0;
            fprintf(stderr, "[pages] base %08x count %d id9d84ec %08x:", base, b ? (int)(e - b) / 8 : -1, R32(0x9d84ecu));
            for (uint32_t q = b; b && q < e && q < b + 8 * 8; q += 8) fprintf(stderr, " %08x/%08x crc %08x", R32(q), R32(q + 4), R32(q) ? R32(R32(q) + 0x3c) : 0);
            fprintf(stderr, " -> %s\n", ui_menu() ? "menu" : "play");
        }
    }
    seek_step();
    extern void memscan_frame(void);
    static int scan = -1;
    if (scan < 0) scan = getenv("RC_MEMSCAN") != NULL;
    if (scan) memscan_frame();
}

/* The game's open UI pages: a list at [0xe6d83c] + 0x10 (8-byte entries, the page object first), each page
   naming itself by a CRC-32 of its name at +0x3c (FUN_004305c0 / FUN_0040b970). A menu is open when one of the
   menu pages is in it (P_MainMenu, P_InGameMenu for pause, ...); in play it holds P_Game, P_Tutorial, P_Combo_*. The dagger cursor can't tell them apart: the game draws it in play too, parked on the picture's edge. */
static uint32_t crc32s(const char *t) {   /* zlib's CRC-32, as the game names its pages */
    uint32_t c = 0xffffffffu;
    for (; *t; t++) { c ^= (uint8_t)*t; for (int k = 0; k < 8; k++) c = c >> 1 ^ (0xedb88320u & -(c & 1)); }
    return ~c;
}
/* the menu pages (names from the executable); anything else (P_Game, P_Tutorial, P_Combo_*) is play */
static int menu_page(uint32_t crc) {
    static const char *names[] = { "P_MainMenu", "P_ProfileMenu", "P_CreateProfile", "P_CreateProfile_VirtualKeyboard",
        "P_LoadSavedGame", "P_StartNewGame", "P_SpecialLoad", "P_Options", "P_GameOptions", "P_ControlsMenu",
        "P_ControlsGampad", "P_SoundsMenu", "P_VideoMenu", "P_VideoMenuAdv", "P_VideoAdvanced", "P_ExtraFeatures",
        "P_ArtworkGallery", "P_ArtworkCharacterGallery", "P_ArtworkIllustrationGallery", "P_GenericGallery", "P_Credits",
        "P_InGameMenu", "P_PreInGameMenu", "P_SaveGame", "P_SaveConfirmation", "P_GameOver", "P_MessageBox" };
    enum { N = sizeof names / sizeof *names };
    static uint32_t h[N];
    static int init;
    if (!init) { for (int i = 0; i < N; i++) h[i] = crc32s(names[i]); init = 1; }
    for (int i = 0; i < N; i++) if (h[i] == crc) return 1;
    return 0;
}
static int ui_menu(void) {
    uint32_t base = R32(0xe6d83cu);
    if (!base) return 0;
    uint32_t b = R32(base + 0x14), e = R32(base + 0x18);
    if (!b || e <= b || e - b > 64 * 8) return 0;
    for (uint32_t q = b; q < e; q += 8) {
        uint32_t page = R32(q);
        if (page && menu_page(R32(page + 0x3c))) return 1;
    }
    return 0;
}

int gamephase_current(void) { return bink_movie_open() ? 2 : ui_menu() ? 1 : 0; }

/* nx, ny: the tap within the window, 0..1 */
void gamephase_menu_tap(float nx, float ny) {
    extern void render_picture_rect(float r[4]);
    extern uint32_t host_video_width(void), host_video_height(void);
    float r[4];
    render_picture_rect(r);
    if (r[2] <= 0 || r[3] <= 0) return;
    g_tx = (nx - r[0]) / r[2] * (float)host_video_width();
    g_ty = (ny - r[1]) / r[3] * (float)host_video_height();
    g_seek_frames = 0; g_settle = 0;
    g_seek = 1;
}
