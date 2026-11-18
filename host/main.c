/* Native launcher: loads the game executable's image, wires up the Windows API layer and runs the game's
 * real entry point (the MSVC C runtime startup, which then calls WinMain).
 *
 * Usage: recomp [game-folder]      (default: ./game)
 * Env:   RC_EXE=name.exe      the executable inside the game folder (default: PROFILE_EXE)
 * Env:   RC_LOG=1          host-side log (files, registry, threads, ...)
 *        RC_LOG_IMPORTS=1  every Windows call the game makes
 *        RC_LOG_D3D=1|2    Direct3D object creation (1) or every Direct3D call (2)
 *        RC_LOG_FPS=1      frames per second
 *        RC_HEADLESS=1     no window (rendering skipped, Present paced at 60 Hz)
 *        RC_NO_AUDIO=1     mute
 */
#include "win32.h"
#include "render.h"

#include <limits.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>

static LoadedImage g_img;
static CPU g_cpu;

/* A packed exe (GOG's, UPX): the game starts in the exe's own unpacker stub, translated by tools/upx_stub.py
   (out_upx/upx_stub.c). It decompresses the game into the image, fills the import table through LoadLibraryA /
   GetProcAddress (host/kernel32.c g_upx_binding) and jumps to the original entry point, the translated game.
   There the image is compared with the unpacked exe the game was translated from. Weak: builds without the
   generated file still link (and refuse packed exes). */
extern const uint32_t upx_stub_entry __attribute__((weak)), upx_oep __attribute__((weak));
extern const uint32_t upx_check[][2] __attribute__((weak)), upx_ncheck __attribute__((weak));
extern const uint64_t upx_check_hash __attribute__((weak));
void register_upx_stub(void) __attribute__((weak));
extern int g_upx_binding, g_image_packed;   /* host/kernel32.c, runtime.c */
static GuestFn g_game_entry;

static void upx_at_original_entry(CPU *c) {
    g_upx_binding = 0;
    register_function(upx_oep, g_game_entry);
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint32_t i = 0; i < upx_ncheck; i++)
        for (const uint8_t *p = HP(upx_check[i][0]), *e = HP(upx_check[i][1]); p < e; p++)
            h = (h ^ *p) * 0x100000001b3ull;
    if (h == upx_check_hash)
        fprintf(stderr, "[upx] game unpacked: the image matches the translated exe\n");
    else
        fprintf(stderr, "[upx] WARNING: the unpacked image differs from the translated exe (%016llx, expected %016llx)\n",
                (unsigned long long)h, (unsigned long long)upx_check_hash);
    g_game_entry(c);
}

/* 0 if the packed exe can be started at its stub; otherwise a message for the player */
static const char *upx_prepare(void) {
    if (!register_upx_stub || !&upx_stub_entry || !upx_stub_entry)
        return "This build can't run a packed (UPX) game exe. Unpack it on a PC first (PrepareGame.bat).";
    if (g_img.entry != upx_stub_entry)
        return "The game exe is packed (UPX) but isn't the GOG version this app knows. "
               "Unpack it on a PC first (PrepareGame.bat), or use the GOG exe.";
    register_upx_stub();
    g_game_entry = lookup_function(upx_oep);
    if (!g_game_entry) return "The packed exe's original entry point was not recompiled.";
    register_function(upx_oep, upx_at_original_entry);
    g_upx_binding = 1;
    fprintf(stderr, "[upx] packed exe: starting at its unpacker stub %08x (game entry %08x)\n", upx_stub_entry, upx_oep);
    return NULL;
}

/* RC_WATCHDOG=secs: periodically print the game thread's guest call stack (return addresses found
   on its stack), to see where it is when it seems stuck. */
static void *watchdog(void *arg) {
    int secs = (int)(intptr_t)arg;
    for (;;) {
        sleep((unsigned)secs);
        if (deterministic()) {   /* a stuck game thread that doesn't move these spins on memory */
            extern uint32_t g_det_reads, g_det_sleeps, g_det_waits;
            extern uint64_t g_vtime_us;
            fprintf(stderr, "[watchdog] virtual time %.6f s, game thread: clock reads %u, sleeps %u, waits %u\n",
                    g_vtime_us * 1e-6, g_det_reads, g_det_sleeps, g_det_waits);
        }
        extern GuestThread *g_all_threads[64];
        extern int g_nthreads;
        for (int t = -1; t < g_nthreads && t < 64; t++) {
            CPU *c = t < 0 ? &g_cpu : &g_all_threads[t]->cpu;
            fprintf(stderr, "[watchdog] %s %x esp=%08x stack:", t < 0 ? "game thread" : "thread", t < 0 ? 0x100 : g_all_threads[t]->id,
                    c->esp);
            for (int k = 0, n = 0; k < 2048 && n < 16; k++) {
                uint32_t v = R32(c->esp + 4 * k);
                if (v >= g_image_lo && v < g_image_hi) { fprintf(stderr, " %08x", v); n++; }
            }
            fprintf(stderr, "\n");
        }
    }
    return NULL;
}

/* RC_POLL_ADDR=addr: watch one guest word from a host thread and, when it changes, print every thread's guest stack */
static void *poller(void *arg) {
    uint32_t addr = (uint32_t)(uintptr_t)arg, last = R32(addr);
    for (;;) {
        usleep(200);
        uint32_t v = R32(addr);
        if (v == last) continue;
        fprintf(stderr, "[poll] %08x: %08x -> %08x\n", addr, last, v);
        last = v;
        extern GuestThread *g_all_threads[64];
        extern int g_nthreads;
        for (int t = -1; t < g_nthreads && t < 64; t++) {
            CPU *c = t < 0 ? &g_cpu : &g_all_threads[t]->cpu;
            fprintf(stderr, "[poll] %s %x stack:", t < 0 ? "game thread" : "thread", t < 0 ? 0x100 : g_all_threads[t]->id);
            for (int k = 0, n = 0; k < 2048 && n < 14; k++) {
                uint32_t w = R32(c->esp + 4 * k);
                if (w >= g_image_lo && w < g_image_hi) { fprintf(stderr, " %08x", w); n++; }
            }
            fprintf(stderr, "\n");
        }
    }
    return NULL;
}

#ifdef __ANDROID__
#include <unistd.h>
#include <sys/syscall.h>
/* Android schedules the game thread on any core, and on phones with big and little cores (Galaxy A17: two
   Cortex-A76 at 2.2 GHz, six A55 at 2.0 GHz) it often lands on a little one, two to three times slower: frames
   that took 15 ms took 60-140. The game thread is kept on the fastest cores (RC_ANY_CORE=1: don't). */
static void pin_to_fast_cores(void) {
    if (getenv("RC_ANY_CORE")) return;
    unsigned f[64] = { 0 }, best = 0;
    int n = 0;
    for (int i = 0; i < 64; i++) {
        char path[96];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        FILE *fp = fopen(path, "r");
        if (!fp) continue;
        if (fscanf(fp, "%u", &f[i]) == 1 && f[i] > best) best = f[i];
        fclose(fp);
    }
    unsigned long long set = 0;
    for (int i = 0; i < 64; i++) if (best && f[i] == best) { set |= 1ull << i; n++; }
    if (!n) return;
    int r = (int)syscall(__NR_sched_setaffinity, 0, sizeof set, &set);
    fprintf(stderr, "[main] game thread on the %d fastest cores (%u kHz): %s\n", n, best, r ? "failed" : "ok");
}
#endif
static void *game_main(void *arg) {
#ifdef __ANDROID__
    pin_to_fast_cores();
#endif
    (void)arg;
    CPU *c = &g_cpu;
    thread_main_init(c);
    sched_main_init();
    t_self->cpu = *c;            /* keep the thread record consistent with the main CPU */
    fpu_sync_rounding(c);
    GuestFn entry = lookup_function(g_img.entry);
    if (!entry) { fprintf(stderr, "entry point %08x was not recompiled\n", g_img.entry); exit(1); }
    fprintf(stderr, "starting game at %08x\n", g_img.entry);
    PUSH32(c, GUEST_RETURN_SENTINEL);
    entry(c);
    fprintf(stderr, "game returned %d\n", (int)c->eax);
    exit((int)c->eax);
}

#if defined(__EMSCRIPTEN__) && !defined(RC_NODE_BUILD)
#include <emscripten/wasmfs.h>
#endif
#ifdef __ANDROID__
#include <SDL.h>   /* renames main to SDL_main, which SDLActivity calls */
const char *android_game_dir(void);   /* android/host/window_android.c */
#endif

int main(int argc, char **argv) {
#if defined(__EMSCRIPTEN__) && !defined(RC_NODE_BUILD)
    /* browser: the page copied the player's game folder into OPFS; reach it through /opfs */
    wasmfs_create_directory("/opfs", 0777, wasmfs_create_opfs_backend());
#endif
    const char *game = argc > 1 ? argv[1] : "game";
#ifdef __ANDROID__
    if (argc <= 1 && !(game = android_game_dir())) return 1;   /* app-specific storage: .../files/game */
#endif
    if (!realpath(game, g_game_root)) { perror(game); return 1; }
    /* RC_USERDATA: another folder for saves and settings (e.g. a fresh copy per test run) */
    if (getenv("RC_USERDATA")) snprintf(g_user_root, sizeof g_user_root, "%s", getenv("RC_USERDATA"));
    else snprintf(g_user_root, sizeof g_user_root, "%s/../userdata", g_game_root);
    mkdir(g_user_root, 0755);
    {   /* the game looks for its install folder by name (FindFirstFileA "c:/ubisoft/pop3"); mirror it in userdata */
        char d[1100]; snprintf(d, sizeof d, "%s", g_user_root);
        for (const char *p = GUEST_ROOT + 2; *p; ) {
            size_t l = strlen(d); d[l++] = '/';
            while (*p == '\\') p++;
            while (*p && *p != '\\') d[l++] = *p++;
            d[l] = 0; mkdir(d, 0755);
        }
    }
    g_log = getenv("RC_LOG") != NULL;
    extern int g_log_d3d;
    g_log_d3d = getenv("RC_LOG_D3D") ? atoi(getenv("RC_LOG_D3D")) : 0;

    extern const ImportTable *pop3_imports(void);
    static ImportHandler all[1024];
    uint32_t n = 0;
    const ImportTable *tables[] = { &kernel32_imports, &user32_imports, &gdi32_imports, &advapi32_imports,
                                    &winmm_imports, &directx_imports, pop3_imports() };
    for (size_t t = 0; t < sizeof tables / sizeof tables[0]; t++)
        for (uint32_t i = 0; i < tables[t]->count; i++) all[n++] = tables[t]->list[i];
    set_import_handlers(all, n);

    runtime_init(&g_cpu);
    vclock_init();
    char exe[1100];
    snprintf(exe, sizeof exe, "%s/%s", g_game_root, getenv("RC_EXE") ? getenv("RC_EXE") : PROFILE_EXE);
    if (load_image(exe, &g_img)) return 1;
    register_all_functions();
    fprintf(stderr, "loaded %s: %u imports, %u of them implemented\n", exe, g_img.num_imports, n);
    if (g_image_packed) {
        const char *why = upx_prepare();
        if (why) {
            fprintf(stderr, "%s\n", why);
#ifdef __ANDROID__
            SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, PROFILE_TITLE, why, NULL);
#endif
            return 1;
        }
    }

    /* The window and its event loop must live on the main thread (macOS); the game gets its own
       thread. RC_HEADLESS=1 runs without a window (no rendering, 60 Hz pacing). */
    int headless = getenv("RC_HEADLESS") != NULL;
    extern int window_create(const char *title);
    extern void window_event_loop(void);
#ifdef __EMSCRIPTEN__
    /* browser: the page owns the canvas and replays our frames; this thread just waits for the game */
    if (!headless) g_render = &g_render_gpu;
    headless = 1;
#else
    if (!headless && window_create(PROFILE_TITLE)) g_render = &g_render_gpu;
    else headless = 1;
#endif

    /* Recompiled code nests one host frame per guest call, so give the game a generous host stack. */
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setstacksize(&at, 256u << 20);
    pthread_t th;
    pthread_create(&th, &at, game_main, NULL);
    if (getenv("RC_POLL_ADDR")) { pthread_t pt; pthread_create(&pt, NULL, poller, (void *)(uintptr_t)strtoul(getenv("RC_POLL_ADDR"), NULL, 16)); }
    if (getenv("RC_WATCHDOG")) {
        pthread_t wd;
        pthread_create(&wd, NULL, watchdog, (void *)(intptr_t)atoi(getenv("RC_WATCHDOG")));
    }
    if (headless) pthread_join(th, NULL);
    else window_event_loop();  /* never returns; closing the window exits */
    return 0;
}
