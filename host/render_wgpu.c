/* Render executor: owns every WebGPU object and executes the recorder's commands (render_cmd.h).
 *
 * Natively (wgpu-native, Metal) commands run as they are recorded and frames go to the SDL window.
 * In the browser (Dawn via Emscripten) this runs on the page's thread: the game thread queues command
 * chunks, and the page pumps them, presenting at most one frame per animation frame; a readback
 * pauses the pump until the GPU copy is mapped.
 *
 * One WGSL "uber shader" implements Direct3D 8's fixed-function stages; ps.1.1 pixel shaders are
 * translated to WGSL (pshader.c). Pipelines are cached by render state; textures are RGBA8 by id. */
#include "render.h"
#include "render_cmd.h"

#include <math.h>
#include <webgpu/webgpu.h>
#ifndef __EMSCRIPTEN__
#include <webgpu/wgpu.h>   /* wgpu-native extensions: border addressing, timestamp period */
#endif
#include <unistd.h>
#include <sys/stat.h>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

extern int g_window_w, g_window_h;
#if defined(__ANDROID__)
/* android/host/window_android.c: the app's ANativeWindow, replaced when Android recreates the surface */
extern void *g_android_window;
unsigned android_present_begin(void);
void android_present_end(void);
static unsigned g_android_gen;
#elif !defined(__EMSCRIPTEN__)
extern void *g_metal_layer;
#endif
static WGPUCompositeAlphaMode g_surf_alpha = WGPUCompositeAlphaMode_Opaque;

/* ------------------------------------------------------------------ shader */

static const char *WGSL_COMMON =
"struct U {\n"
"  mvp: mat4x4<f32>,\n"
"  vp: vec4<f32>,        // D3D viewport x, y, w, h in back-buffer pixels\n"
"  flags: vec4<u32>,     // x: 1 = pre-transformed, 2 = clip space (vertex shader), y: alpha func (0 = test off), z: specular enable, w: lighting\n"
"  st0: vec4<u32>,       // colorop, colorarg1, colorarg2, alphaop\n"
"  st0b: vec4<u32>,      // alphaarg1, alphaarg2, texcoord index, colorarg0 | alphaarg0 << 8\n"
"  st1: vec4<u32>,\n"
"  st1b: vec4<u32>,\n"
"  tfactor: vec4<f32>,\n"
"  misc: vec4<f32>,      // x: alpha ref (0..1)\n"
"  mat_diffuse: vec4<f32>,\n"
"  ambient: vec4<f32>,\n"
"  psc: array<vec4<f32>, 8>,   // pixel shader constants c0-c7\n"
"  bem: array<vec4<f32>, 4>,   // bump-environment matrices per stage (00, 01, 10, 11)\n"
"  bel: array<vec4<f32>, 4>,   // bump luminance scale, offset per stage\n"
"};\n"
"@group(0) @binding(0) var<uniform> u: U;\n"
"@group(1) @binding(0) var tex0: texture_2d<f32>;\n"
"@group(1) @binding(1) var smp0: sampler;\n"
"@group(1) @binding(2) var tex1: texture_2d<f32>;\n"
"@group(1) @binding(3) var smp1: sampler;\n"
"@group(1) @binding(4) var tex2: texture_2d<f32>;\n"
"@group(1) @binding(5) var smp2: sampler;\n"
"@group(1) @binding(6) var tex3: texture_2d<f32>;\n"
"@group(1) @binding(7) var smp3: sampler;\n"
"struct VI { @location(0) pos: vec4<f32>, @location(1) diffuse: u32, @location(2) specular: u32,\n"
"            @location(3) tc0: vec4<f32>, @location(4) tc1: vec4<f32>, @location(5) tc2: vec4<f32>, @location(6) tc3: vec4<f32> };\n"
"struct VO { @builtin(position) pos: vec4<f32>, @location(0) diffuse: vec4<f32>, @location(1) specular: vec4<f32>,\n"
"            @location(2) tc0: vec4<f32>, @location(3) tc1: vec4<f32>, @location(4) tc2: vec4<f32>, @location(5) tc3: vec4<f32> };\n"
"fn d3dcolor(c: u32) -> vec4<f32> { return unpack4x8unorm(c).zyxw; }\n"
"@vertex fn vs(v: VI) -> VO {\n"
"  var o: VO;\n"
"  if (u.flags.x == 1u) {\n"
"    let w = select(1.0 / v.pos.w, 1.0, v.pos.w == 0.0);\n"
"    let nx = (v.pos.x + 0.5 - u.vp.x) / u.vp.z * 2.0 - 1.0;\n"
"    let ny = 1.0 - (v.pos.y + 0.5 - u.vp.y) / u.vp.w * 2.0;\n"
"    o.pos = vec4<f32>(nx * w, ny * w, v.pos.z * w, w);\n"
"  } else {\n"
"    var p = u.mvp * vec4<f32>(v.pos.xyz, 1.0);\n"
"    if (u.flags.x == 2u) { p = v.pos; }   // vertex shader output: already in clip space\n"
"    p.x = p.x + p.w / u.vp.z;\n"
"    p.y = p.y - p.w / u.vp.w;\n"
"    o.pos = p;\n"
"  }\n"
"  o.diffuse = d3dcolor(v.diffuse);\n"
"  if (u.flags.w == 1u) { o.diffuse = clamp(o.diffuse * u.mat_diffuse + u.ambient, vec4<f32>(0.0), vec4<f32>(1.0)); }\n"
"  o.specular = d3dcolor(v.specular);\n"
"  o.tc0 = v.tc0; o.tc1 = v.tc1; o.tc2 = v.tc2; o.tc3 = v.tc3;\n"
"  return o;\n"
"}\n"
"// ps 1.x tex: D3DTTFF_PROJECTED stages divide by a coordinate (u.misc.y: 3 bits per stage, the count 2-4; 0 = not projected)\n"
"fn ptc(tc: vec4<f32>, s: u32) -> vec2<f32> {\n"
"  let c = (u32(u.misc.y) >> (3u * s)) & 7u;\n"
"  if (c == 0u) { return tc.xy; }\n"
"  return tc.xy / select(select(tc.w, tc.z, c == 3u), tc.y, c == 2u);\n"
"}\n"
"fn alpha_ok(a: f32) -> bool {\n"
"  let r = u.misc.x;\n"
"  switch (u.flags.y) {\n"
"    case 1u: { return false; } case 2u: { return a < r; } case 3u: { return abs(a - r) < 0.002; }\n"
"    case 4u: { return a <= r; } case 5u: { return a > r; } case 6u: { return abs(a - r) >= 0.002; }\n"
"    case 7u: { return a >= r; } default: { return true; }\n"
"  }\n"
"}\n";

/* fixed-function texture stages */
static const char *WGSL_FF =
"fn arg(a: u32, cur: vec4<f32>, tex: vec4<f32>, dif: vec4<f32>, spc: vec4<f32>) -> vec4<f32> {\n"
"  var v = cur;\n"
"  switch (a & 7u) { case 0u: { v = dif; } case 2u: { v = tex; } case 3u: { v = u.tfactor; } case 4u: { v = spc; } default: {} }\n"
"  if ((a & 0x10u) != 0u) { v = vec4<f32>(1.0) - v; }\n"
"  if ((a & 0x20u) != 0u) { v = vec4<f32>(v.a); }\n"
"  return v;\n"
"}\n"
"fn top(o: u32, a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>, cur: vec4<f32>, tex: vec4<f32>, dif: vec4<f32>) -> vec4<f32> {\n"
"  switch (o) {\n"
"    case 2u: { return a1; }  case 3u: { return a2; }\n"
"    case 4u: { return a1 * a2; }  case 5u: { return a1 * a2 * 2.0; }  case 6u: { return a1 * a2 * 4.0; }\n"
"    case 7u: { return a1 + a2; }  case 8u: { return a1 + a2 - 0.5; }  case 9u: { return (a1 + a2 - 0.5) * 2.0; }\n"
"    case 10u: { return a1 - a2; }  case 11u: { return a1 + a2 - a1 * a2; }\n"
"    case 12u: { return mix(a2, a1, dif.a); }  case 13u: { return mix(a2, a1, tex.a); }\n"
"    case 14u: { return mix(a2, a1, u.tfactor.a); }  case 15u: { return a1 + a2 * (1.0 - tex.a); }\n"
"    case 16u: { return mix(a2, a1, cur.a); }\n"
"    case 18u: { return vec4<f32>(a1.rgb + a1.a * a2.rgb, a1.a); }\n"
"    case 19u: { return vec4<f32>(a1.rgb * a2.rgb + vec3<f32>(a1.a), a1.a); }\n"
"    case 20u: { return vec4<f32>((1.0 - a1.a) * a2.rgb + a1.rgb, a1.a); }\n"
"    case 21u: { return vec4<f32>((vec3<f32>(1.0) - a1.rgb) * a2.rgb + vec3<f32>(a1.a), a1.a); }\n"
"    case 24u: { let d = saturate(dot(a1.rgb - 0.5, a2.rgb - 0.5) * 4.0); return vec4<f32>(d); }\n"
"    case 25u: { return a0 + a1 * a2; }\n"
"    case 26u: { return a0 * a1 + (vec4<f32>(1.0) - a0) * a2; }\n"
"    default: { return a1; }\n"
"  }\n"
"}\n"
"fn stage(s: vec4<u32>, sb: vec4<u32>, cur: vec4<f32>, tex: vec4<f32>, dif: vec4<f32>, spc: vec4<f32>) -> vec4<f32> {\n"
"  let c0 = arg(sb.w & 255u, cur, tex, dif, spc);\n"
"  let c = top(s.x, c0, arg(s.y, cur, tex, dif, spc), arg(s.z, cur, tex, dif, spc), cur, tex, dif);\n"
"  var a = cur.a;\n"
"  if (s.w > 1u) {\n"
"    let a0 = arg(sb.w >> 8u, cur, tex, dif, spc);\n"
"    a = top(s.w, a0, arg(sb.x, cur, tex, dif, spc), arg(sb.y, cur, tex, dif, spc), cur, tex, dif).a;\n"
"  }\n"
"  return saturate(vec4<f32>(c.rgb, a));\n"
"}\n"
"@fragment fn fs(i: VO) -> @location(0) vec4<f32> {\n"
"  var cur = i.diffuse;\n"
"  let uv0 = select(i.tc0.xy, i.tc1.xy, u.st0b.z == 1u);\n"
"  let uv1 = select(i.tc0.xy, i.tc1.xy, u.st1b.z == 1u);\n"
"  let tx0 = textureSample(tex0, smp0, uv0);\n"
"  let tx1 = textureSample(tex1, smp1, uv1);\n"
"  if (u.st0.x != 1u) {\n"
"    cur = stage(u.st0, u.st0b, cur, tx0, i.diffuse, i.specular);\n"
"    if (u.st1.x != 1u) { cur = stage(u.st1, u.st1b, cur, tx1, i.diffuse, i.specular); }\n"
"  }\n"
"  if (u.flags.z == 1u) { cur = vec4<f32>(saturate(cur.rgb + i.specular.rgb), cur.a); }\n"
"  if (!alpha_ok(cur.a)) { discard; }\n"
"  return cur;\n"
"}\n";

/* ------------------------------------------------------------------ state */

static void release_view(WGPUTextureView view);   /* drops cached bind groups using it, then releases it */

typedef struct { WGPUTexture tex; WGPUTextureView view; uint32_t used_in_batch, w, h, levels, fmt; size_t bytes; } GTex;
int g_gpu_bc;               /* device has texture-compression-bc */
size_t g_gpu_tex_bytes;     /* GPU memory in game textures, for the [mem] log */
/* Per-command counts and host time, for the [gpu] line next to [fps] (RC_LOG_FPS=1) */
enum { ST_DRAW, ST_UPLOAD_TEX, ST_UPLOAD_SURF, ST_READBACK, ST_PRESENT, ST_OTHER, ST_N };
static uint32_t g_pc_hits, g_pc_waits, g_pc_shared, g_pc_built, g_pc_warmed;   /* pipelines: from the warm-up thread (waited for), the same as one made earlier, built by the game thread, warmed up */
static struct { uint32_t n[ST_N]; double t[ST_N]; uint32_t submits, passes, passes_load; uint64_t upload_bytes; double acquire_t, submit_t, swap_t, pace_t, pt_wait_t; } g_st;
void gpu_stats_log(double secs, uint32_t frames) {
    static const char *names[ST_N] = { "draw", "tex upload", "surf upload", "readback", "present", "other" };
    if (!frames) frames = 1;
    char line[512];
    int o = snprintf(line, sizeof line, "[gpu] per frame:");
    for (int i = 0; i < ST_N; i++)
        if (g_st.n[i]) o += snprintf(line + o, sizeof line - (size_t)o, " %s %.1fx %.2fms,", names[i],
                                     (double)g_st.n[i] / frames, g_st.t[i] * 1000 / frames);
    fprintf(stderr, "%s passes %.1f (%.1f loading color), submits %.1f, uploads %.0f KB | present: submit %.2fms acquire %.2fms swap %.2fms pace %.2fms, game waits %.2fms "
            "(frame %.1fms)\n", line, (double)g_st.passes / frames, (double)g_st.passes_load / frames, (double)g_st.submits / frames, g_st.upload_bytes / 1024.0 / frames,
            g_st.submit_t * 1000 / frames, g_st.acquire_t * 1000 / frames, g_st.swap_t * 1000 / frames,
            g_st.pace_t * 1000 / frames, g_st.pt_wait_t * 1000 / frames, secs * 1000 / frames);
    if (g_pc_hits || g_pc_built || g_pc_shared)
        fprintf(stderr, "[gpu] new pipelines: %u built here, %u from the warm-up cache (%u waited for), %u shared\n", g_pc_built, g_pc_hits,
                g_pc_waits, g_pc_shared);
    g_pc_hits = g_pc_waits = g_pc_built = g_pc_shared = 0;
    memset(&g_st, 0, sizeof g_st);
}
typedef struct { WGPUTexture tex; WGPUTextureView view; uint32_t w, h; int depth, gpu_dirty; } GSurf;

static WGPUInstance g_inst;
static WGPUAdapter g_adapter;
static WGPUDevice g_dev;
static WGPUQueue g_queue;
static WGPUSurface g_surface;
static WGPUTextureFormat g_surf_fmt;
extern int g_game_aspect_w, g_game_aspect_h;   /* the game's back buffer shape (host/core.c): 4:3 by default, 16:9 at 1080p */
static int g_asp_w = 4, g_asp_h = 3;
static int g_surf_w, g_surf_h;   /* back buffer: the largest 4:3 area that fits the window */
static int g_win_w, g_win_h;     /* window (swap chain) size; the back buffer is centred in it */
static WGPUTexture g_depth;
static WGPUTextureView g_depth_view;
static WGPUTexture g_bb;          /* offscreen back buffer: always rendered, copied to the window when visible */
static WGPUTextureView g_bb_view;
static WGPUShaderModule g_shader;   /* shared header + fixed-function fragment stage */
static WGPUBindGroupLayout g_bgl_u, g_bgl_t;
static WGPUPipelineLayout g_pl;
static WGPUBuffer g_vbuf, g_ubuf;
static uint32_t g_vpos, g_upos;
/* CPU copies of the vertex, uniform and vertex-shader-constant data of the batch being recorded: each
   buffer is written to the GPU once per submit instead of once per draw (two writes per draw cost more
   than everything else in a frame, and each wgpuQueueWriteBuffer takes wgpu's queue lock, which the
   present thread holds while the driver blocks in present) */
static uint8_t *g_vstage, *g_ustage, *g_cstage;
static int g_cfg_window_w, g_cfg_window_h;   /* g_window_w/h the swap chain was configured for */
#define VBUF_SIZE (32u << 20)
#define UBUF_SIZE (8u << 20)
#define USTRIDE 512u
/* vertex shader constants (96 vec4) of draws whose vertex shader runs on the GPU: @group(2) */
#define CBUF_SIZE (8u << 20)
#define CSTRIDE 1536u
static WGPUBindGroupLayout g_bgl_c;
static WGPUPipelineLayout g_pl_vs;
static WGPUBuffer g_cbuf;
static WGPUBindGroup g_bg_c;
static uint32_t g_cpos;
/* contents of cached vertex buffers created since the last submit (exec_upload_vb) */
typedef struct { WGPUBuffer buf; uint8_t *data; uint32_t bytes; } VbPending;
static VbPending *g_pend;
static size_t g_npend, g_cappend;

/* Present thread (native): the game thread hands a finished frame over and goes on with the next one,
 * while the present thread copies the back buffer into the window and presents it (vkQueuePresentKHR
 * blocks there for ~20 ms on Mali). States: 0 idle, 1 frame handed over, 2 copy submitted (presenting).
 * The game thread's next submit waits for state != 1, so the copy reads the back buffer before anything
 * new is drawn into it; reconfiguring the surface waits for state 0. g_qlock: Vulkan queues need
 * external synchronization, so submits and presents never run at once. RC_PRESENT_THREAD=0: inline. */
#ifndef __EMSCRIPTEN__
#include <pthread.h>
static pthread_mutex_t g_qlock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_pt_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_pt_cond = PTHREAD_COND_INITIALIZER;
static int g_pt_on, g_pt_state;
static int g_pt_reconfig;   /* set by the present thread: 1 new Android surface, 2 reconfigure the surface */
static void pt_wait(int idle) {
    if (!g_pt_on) return;
    pthread_mutex_lock(&g_pt_lock);
    while (idle ? g_pt_state != 0 : g_pt_state == 1) pthread_cond_wait(&g_pt_cond, &g_pt_lock);
    pthread_mutex_unlock(&g_pt_lock);
}
static void pt_set(int state) {
    pthread_mutex_lock(&g_pt_lock);
    g_pt_state = state;
    pthread_cond_broadcast(&g_pt_cond);
    pthread_mutex_unlock(&g_pt_lock);
}
static void queue_submit(WGPUCommandBuffer cb) {
    pthread_mutex_lock(&g_qlock);
    wgpuQueueSubmit(g_queue, 1, &cb);
    pthread_mutex_unlock(&g_qlock);
}
static void start_present_thread(void);
static void pipe_cache_start(void);   /* pipeline warm-up cache (pipelines section) */
#else
static void pt_wait(int idle) { (void)idle; }
static void queue_submit(WGPUCommandBuffer cb) { wgpuQueueSubmit(g_queue, 1, &cb); }
#endif
static WGPUBindGroup g_bg_u;
static GTex g_white, g_black;
static WGPUSampler g_samplers[4096];   /* indexed by sampler key: never evicted while in use */
/* RC_GPU_TIMES=1: every 2 s one frame's render passes get GPU timestamps, logged as [gputime] (what each pass costs
   the GPU: the post-process chain, shadow maps, the scene) */
static int g_ts_on, g_ts_capture;
#define TS_MAX 64
static uint32_t g_ts_n;
static struct { uint32_t w, h, load, rt; } g_ts_info[TS_MAX];
static WGPUQuerySet g_ts_set;
static WGPUBuffer g_ts_resolve, g_ts_read;
static int g_gpu_border;   /* WGPUNativeFeature_AddressModeClampToBorder: D3DTADDRESS_BORDER sampled as a border, not edge-clamped */

static WGPUCommandEncoder g_enc;
static WGPURenderPassEncoder g_pass;
static uint32_t g_batch = 1;
static int g_frame_open;

static WGPUStringView sv(const char *s) { return (WGPUStringView){ s, WGPU_STRLEN }; }

/* ids (shared counter for textures and surfaces) -> GTex / GSurf */
#define OBJ_CAP 16384
static struct { uint32_t id; void *p; } g_objs[OBJ_CAP];
static void *obj_get(uint32_t id) {
    for (uint32_t i = 0, h = id * 2654435761u; i < OBJ_CAP; i++) {
        uint32_t k = (h + i) & (OBJ_CAP - 1);
        if (g_objs[k].id == id) return g_objs[k].p;
        if (!g_objs[k].id) return NULL;
    }
    return NULL;
}
static void obj_put(uint32_t id, void *p) {
    for (uint32_t i = 0, h = id * 2654435761u; i < OBJ_CAP; i++) {
        uint32_t k = (h + i) & (OBJ_CAP - 1);
        if (g_objs[k].id == id || !g_objs[k].id || g_objs[k].id == 0xFFFFFFFFu) { g_objs[k].id = id; g_objs[k].p = p; return; }
    }
    fprintf(stderr, "[wgpu] object table full\n");
}
static void obj_del(uint32_t id) {
    for (uint32_t i = 0, h = id * 2654435761u; i < OBJ_CAP; i++) {
        uint32_t k = (h + i) & (OBJ_CAP - 1);
        if (g_objs[k].id == id) { g_objs[k].id = 0xFFFFFFFFu; g_objs[k].p = NULL; return; }   /* tombstone */
        if (!g_objs[k].id) return;
    }
}

/* ------------------------------------------------------------------ init */

static void on_adapter(WGPURequestAdapterStatus st, WGPUAdapter a, WGPUStringView msg, void *u1, void *u2) {
    (void)u2;
    if (st != WGPURequestAdapterStatus_Success) fprintf(stderr, "[wgpu] adapter: %.*s\n", (int)msg.length, msg.data);
#ifndef __EMSCRIPTEN__
    else {   /* the GPU and driver, for crash reports */
        WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
        if (wgpuAdapterGetInfo(a, &info) == WGPUStatus_Success) {
            fprintf(stderr, "[wgpu] GPU: %.*s, driver %.*s (%.*s)\n", (int)info.device.length, info.device.data,
                    (int)info.description.length, info.description.data, (int)info.architecture.length, info.architecture.data);
            wgpuAdapterInfoFreeMembers(info);
        }
    }
#endif
    *(WGPUAdapter *)u1 = a;
}
static void on_device(WGPURequestDeviceStatus st, WGPUDevice d, WGPUStringView msg, void *u1, void *u2) {
    (void)u2;
    if (st != WGPURequestDeviceStatus_Success) fprintf(stderr, "[wgpu] device: %.*s\n", (int)msg.length, msg.data);
    *(WGPUDevice *)u1 = d;
}
static void on_error(WGPUDevice const *d, WGPUErrorType t, WGPUStringView msg, void *u1, void *u2) {
    (void)d; (void)u1; (void)u2;
    static int count;
    if (count++ < 20) fprintf(stderr, "[wgpu] error %d: %.*s\n", (int)t, (int)msg.length, msg.data);
}

static void make_depth(void) {
    if (g_bb) { release_view(g_bb_view); wgpuTextureRelease(g_bb); }
    WGPUTextureDescriptor bd = WGPU_TEXTURE_DESCRIPTOR_INIT;
    bd.size = (WGPUExtent3D){ (uint32_t)g_surf_w, (uint32_t)g_surf_h, 1 };
    bd.format = g_surf_fmt;
    bd.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
    g_bb = wgpuDeviceCreateTexture(g_dev, &bd);
    g_bb_view = wgpuTextureCreateView(g_bb, NULL);
    if (g_depth) { release_view(g_depth_view); wgpuTextureRelease(g_depth); }
    WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
    td.size = (WGPUExtent3D){ (uint32_t)g_surf_w, (uint32_t)g_surf_h, 1 };
    td.format = WGPUTextureFormat_Depth24PlusStencil8;
    td.usage = WGPUTextureUsage_RenderAttachment;
    g_depth = wgpuDeviceCreateTexture(g_dev, &td);
    g_depth_view = wgpuTextureCreateView(g_depth, NULL);
}

static void configure_surface(void) {
    g_cfg_window_w = g_window_w;
    g_cfg_window_h = g_window_h;
    g_win_w = g_window_w > 0 ? g_window_w : 1024;
    g_win_h = g_window_h > 0 ? g_window_h : 768;
#ifdef __ANDROID__
    /* Phone GPUs are slow and screens dense: render a smaller swap chain and let the display hardware
       scale it up, which costs nothing. RC_RENDER_HEIGHT sets its height (default 720; 0 = native). */
    static int render_h = -1;
    if (render_h < 0) {
        const char *e = getenv("RC_RENDER_HEIGHT");
        render_h = e ? atoi(e) : 540;   /* matches the game resolution (host/core.c video_size) */
        if (render_h < 0) render_h = 0;
    }
    if (render_h >= 240 && render_h < g_win_h) {
        g_win_w = (int)((int64_t)g_win_w * render_h / g_win_h) & ~1;
        g_win_h = render_h;
    }
#endif
    /* render at the largest area with the game's aspect ratio that fits, letterboxed (no stretching) */
    g_asp_w = g_game_aspect_w; g_asp_h = g_game_aspect_h;
    if ((long)g_win_w * g_asp_h > (long)g_win_h * g_asp_w) { g_surf_h = g_win_h; g_surf_w = g_win_h * g_asp_w / g_asp_h; }
    else { g_surf_w = g_win_w; g_surf_h = g_win_w * g_asp_h / g_asp_w; }
    WGPUSurfaceConfiguration sc = WGPU_SURFACE_CONFIGURATION_INIT;
    sc.device = g_dev;
    sc.format = g_surf_fmt;
    sc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst;  /* back buffer is copied in */
    sc.width = (uint32_t)g_win_w;
    sc.height = (uint32_t)g_win_h;
    sc.presentMode = WGPUPresentMode_Fifo;   /* vsync, like D3DPRESENT_INTERVAL_ONE */
#ifdef __EMSCRIPTEN__
    sc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst;
#endif
    sc.alphaMode = g_surf_alpha;
    wgpuSurfaceConfigure(g_surface, &sc);
    make_depth();
}

static void make_solid(GTex *g, uint8_t v) {
    WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
    td.size = (WGPUExtent3D){ 1, 1, 1 };
    td.format = WGPUTextureFormat_RGBA8Unorm;
    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    g->tex = wgpuDeviceCreateTexture(g_dev, &td);
    g->view = wgpuTextureCreateView(g->tex, NULL);
    uint8_t px[4] = { v, v, v, 255 };
    WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    dst.texture = g->tex;
    WGPUTexelCopyBufferLayout lay = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
    lay.bytesPerRow = 4; lay.rowsPerImage = 1;
    WGPUExtent3D ex = { 1, 1, 1 };
    wgpuQueueWriteTexture(g_queue, &dst, px, 4, &lay, &ex);
}

static WGPUShaderModule make_module(const char *fragment) {
    size_t n = strlen(WGSL_COMMON) + strlen(fragment) + 1;
    char *code = malloc(n);
    snprintf(code, n, "%s%s", WGSL_COMMON, fragment);
    WGPUShaderSourceWGSL src = WGPU_SHADER_SOURCE_WGSL_INIT;
    src.code = sv(code);
    WGPUShaderModuleDescriptor smd = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    smd.nextInChain = &src.chain;
    extern double g_hx[]; extern uint32_t g_hxb[];
    double t_m = host_seconds();
    WGPUShaderModule m = wgpuDeviceCreateShaderModule(g_dev, &smd);
    g_hx[4] += host_seconds() - t_m; g_hxb[4]++;
    free(code);
    return m;
}

/* everything after the device exists: surface format, back buffer, shaders, layouts, buffers */
static void create_resources(void) {
    WGPUSurfaceCapabilities caps = WGPU_SURFACE_CAPABILITIES_INIT;
    wgpuSurfaceGetCapabilities(g_surface, g_adapter, &caps);
    g_surf_fmt = caps.formatCount ? caps.formats[0] : WGPUTextureFormat_BGRA8Unorm;
    for (size_t i = 0; i < caps.formatCount; i++)   /* prefer a non-sRGB format: D3D8 blends in gamma space */
        if (caps.formats[i] == WGPUTextureFormat_BGRA8Unorm) g_surf_fmt = caps.formats[i];
#ifdef __ANDROID__
    if (g_surf_fmt != WGPUTextureFormat_BGRA8Unorm)
        for (size_t i = 0; i < caps.formatCount; i++)
            if (caps.formats[i] == WGPUTextureFormat_RGBA8Unorm) g_surf_fmt = caps.formats[i];
#endif
    /* Opaque where offered; some Android drivers (Mali on the Galaxy A17) only offer Inherit */
    if (caps.alphaModeCount) g_surf_alpha = caps.alphaModes[0];
    for (size_t i = 0; i < caps.alphaModeCount; i++)
        if (caps.alphaModes[i] == WGPUCompositeAlphaMode_Opaque) g_surf_alpha = caps.alphaModes[i];
    fprintf(stderr, "[wgpu] surface: format %d, alpha mode %d\n", (int)g_surf_fmt, (int)g_surf_alpha);
    configure_surface();

    g_shader = make_module(WGSL_FF);

    WGPUBindGroupLayoutEntry ue = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    ue.binding = 0;
    ue.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
    ue.buffer.type = WGPUBufferBindingType_Uniform;
    ue.buffer.hasDynamicOffset = 1;
    ue.buffer.minBindingSize = sizeof(Uniforms);
    WGPUBindGroupLayoutDescriptor bd = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bd.entryCount = 1; bd.entries = &ue;
    g_bgl_u = wgpuDeviceCreateBindGroupLayout(g_dev, &bd);

    WGPUBindGroupLayoutEntry te[8];
    for (int i = 0; i < 8; i++) {
        te[i] = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        te[i].binding = (uint32_t)i;
        te[i].visibility = WGPUShaderStage_Fragment;
        if (i % 2 == 0) {
            te[i].texture.sampleType = WGPUTextureSampleType_Float;
            te[i].texture.viewDimension = WGPUTextureViewDimension_2D;
        } else {
            te[i].sampler.type = WGPUSamplerBindingType_Filtering;
        }
    }
    bd.entryCount = 8; bd.entries = te;
    g_bgl_t = wgpuDeviceCreateBindGroupLayout(g_dev, &bd);
    WGPUBindGroupLayout ls[2] = { g_bgl_u, g_bgl_t };
    WGPUPipelineLayoutDescriptor pld = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pld.bindGroupLayoutCount = 2; pld.bindGroupLayouts = ls;
    g_pl = wgpuDeviceCreatePipelineLayout(g_dev, &pld);

    WGPUBufferDescriptor bdesc = WGPU_BUFFER_DESCRIPTOR_INIT;
    bdesc.size = VBUF_SIZE; bdesc.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
    g_vbuf = wgpuDeviceCreateBuffer(g_dev, &bdesc);
    bdesc.size = UBUF_SIZE; bdesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    g_ubuf = wgpuDeviceCreateBuffer(g_dev, &bdesc);
    WGPUBindGroupEntry be = WGPU_BIND_GROUP_ENTRY_INIT;
    be.binding = 0; be.buffer = g_ubuf; be.size = sizeof(Uniforms);
    WGPUBindGroupDescriptor bgd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bgd.layout = g_bgl_u; bgd.entryCount = 1; bgd.entries = &be;
    g_bg_u = wgpuDeviceCreateBindGroup(g_dev, &bgd);

    WGPUBindGroupLayoutEntry ce = ue;
    ce.visibility = WGPUShaderStage_Vertex;
    ce.buffer.minBindingSize = CSTRIDE;
    bd.entryCount = 1; bd.entries = &ce;
    g_bgl_c = wgpuDeviceCreateBindGroupLayout(g_dev, &bd);
    WGPUBindGroupLayout lc[3] = { g_bgl_u, g_bgl_t, g_bgl_c };
    pld.bindGroupLayoutCount = 3; pld.bindGroupLayouts = lc;
    g_pl_vs = wgpuDeviceCreatePipelineLayout(g_dev, &pld);
    bdesc.size = CBUF_SIZE; bdesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    g_cbuf = wgpuDeviceCreateBuffer(g_dev, &bdesc);
    g_vstage = malloc(VBUF_SIZE);   /* only the pages a frame uses are ever touched */
    g_ustage = malloc(UBUF_SIZE);
    g_cstage = malloc(CBUF_SIZE);
    be.buffer = g_cbuf; be.size = CSTRIDE;
    bgd.layout = g_bgl_c;
    g_bg_c = wgpuDeviceCreateBindGroup(g_dev, &bgd);

    make_solid(&g_white, 255);
    make_solid(&g_black, 0);   /* pixel shaders sampling an empty stage read (0, 0, 0, 1) */
    fprintf(stderr, "[wgpu] renderer ready: rendering at %dx%d in a %dx%d window\n", g_surf_w, g_surf_h, g_win_w, g_win_h);
#ifndef __EMSCRIPTEN__
    start_present_thread();
    pipe_cache_start();
#endif
}


#ifdef __ANDROID__
static void android_create_surface(void) {
    WGPUSurfaceSourceAndroidNativeWindow aw = WGPU_SURFACE_SOURCE_ANDROID_NATIVE_WINDOW_INIT;
    aw.window = g_android_window;
    WGPUSurfaceDescriptor sd = WGPU_SURFACE_DESCRIPTOR_INIT;
    sd.nextInChain = &aw.chain;
    g_surface = wgpuInstanceCreateSurface(g_inst, &sd);
}
#endif

#ifndef __EMSCRIPTEN__
int gpu_init_native(void) {
    if (g_dev) return 1;
#ifdef __ANDROID__
    g_inst = wgpuCreateInstance(NULL);
    g_android_gen = android_present_begin();   /* waits for the first surface */
    android_create_surface();
    android_present_end();
#else
    if (!g_metal_layer) { fprintf(stderr, "[wgpu] no window\n"); return 0; }
    g_inst = wgpuCreateInstance(NULL);
    WGPUSurfaceSourceMetalLayer ml = WGPU_SURFACE_SOURCE_METAL_LAYER_INIT;
    ml.layer = g_metal_layer;
    WGPUSurfaceDescriptor sd = WGPU_SURFACE_DESCRIPTOR_INIT;
    sd.nextInChain = &ml.chain;
    g_surface = wgpuInstanceCreateSurface(g_inst, &sd);
#endif

    WGPURequestAdapterOptions ao = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    ao.compatibleSurface = g_surface;
    ao.powerPreference = WGPUPowerPreference_HighPerformance;
    WGPURequestAdapterCallbackInfo aci = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    aci.mode = WGPUCallbackMode_AllowProcessEvents;
    aci.callback = on_adapter;
    aci.userdata1 = &g_adapter;
    wgpuInstanceRequestAdapter(g_inst, &ao, aci);
    while (!g_adapter) wgpuInstanceProcessEvents(g_inst);

    WGPUDeviceDescriptor dd = WGPU_DEVICE_DESCRIPTOR_INIT;
    dd.uncapturedErrorCallbackInfo.callback = on_error;
    static const WGPUFeatureName bc = WGPUFeatureName_TextureCompressionBC;
    g_gpu_bc = wgpuAdapterHasFeature(g_adapter, bc) && !getenv("RC_NO_BC");   /* DXT textures stay compressed */
    static WGPUFeatureName feats[3];
    uint32_t nf = 0;
    if (g_gpu_bc) feats[nf++] = bc;
    const WGPUFeatureName border = (WGPUFeatureName)WGPUNativeFeature_AddressModeClampToBorder;
    g_gpu_border = wgpuAdapterHasFeature(g_adapter, border) && !getenv("RC_NO_BORDER");
    if (g_gpu_border) feats[nf++] = border;
    g_ts_on = getenv("RC_GPU_TIMES") && wgpuAdapterHasFeature(g_adapter, WGPUFeatureName_TimestampQuery);
    if (g_ts_on) feats[nf++] = WGPUFeatureName_TimestampQuery;
    if (getenv("RC_GPU_TIMES") && !g_ts_on) fprintf(stderr, "[gputime] the GPU has no timestamp queries\n");
    dd.requiredFeatureCount = nf; dd.requiredFeatures = nf ? feats : NULL;
    fprintf(stderr, "[wgpu] BC texture compression: %s, border addressing: %s\n", g_gpu_bc ? "yes" : "no (DXT textures are converted to RGBA8)",
            g_gpu_border ? "yes" : "no (edge clamp)");
    WGPURequestDeviceCallbackInfo dci = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    dci.mode = WGPUCallbackMode_AllowProcessEvents;
    dci.callback = on_device;
    dci.userdata1 = &g_dev;
    wgpuAdapterRequestDevice(g_adapter, &dd, dci);
    while (!g_dev) wgpuInstanceProcessEvents(g_inst);
    g_queue = wgpuDeviceGetQueue(g_dev);
    create_resources();
    return 1;
}
#endif


#ifndef __EMSCRIPTEN__
#include <zlib.h>
static void write_png(const char *path, const uint8_t *bgra, uint32_t w, uint32_t h, uint32_t stride) {
    size_t raw_len = (size_t)(w * 3 + 1) * h;
    uint8_t *raw = malloc(raw_len);
    for (uint32_t y = 0; y < h; y++) {
        uint8_t *r = raw + (size_t)y * (w * 3 + 1);
        r[0] = 0;
        for (uint32_t x = 0; x < w; x++) {
            const uint8_t *p = bgra + (size_t)y * stride + 4 * x;
            int is_bgra = g_surf_fmt == WGPUTextureFormat_BGRA8Unorm || g_surf_fmt == WGPUTextureFormat_BGRA8UnormSrgb;
            r[1 + 3 * x] = is_bgra ? p[2] : p[0]; r[2 + 3 * x] = p[1]; r[3 + 3 * x] = is_bgra ? p[0] : p[2];
        }
    }
    uLongf zlen = compressBound((uLong)raw_len);
    uint8_t *z = malloc(zlen);
    compress2(z, &zlen, raw, (uLong)raw_len, 6);
    FILE *f = fopen(path, "wb");
    if (!f) { free(raw); free(z); return; }
    static const uint8_t sig[8] = { 0x89, 'P', 'N', 'G', 13, 10, 26, 10 };
    fwrite(sig, 1, 8, f);
    uint8_t ihdr[13] = { (uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w,
                         (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h, 8, 2, 0, 0, 0 };
    const struct { const char *t; const uint8_t *d; uint32_t n; } chunks[3] = { { "IHDR", ihdr, 13 }, { "IDAT", z, (uint32_t)zlen }, { "IEND", NULL, 0 } };
    for (int i = 0; i < 3; i++) {
        uint8_t len[4] = { (uint8_t)(chunks[i].n >> 24), (uint8_t)(chunks[i].n >> 16), (uint8_t)(chunks[i].n >> 8), (uint8_t)chunks[i].n };
        fwrite(len, 1, 4, f);
        fwrite(chunks[i].t, 1, 4, f);
        if (chunks[i].n) fwrite(chunks[i].d, 1, chunks[i].n, f);
        uLong crc = crc32(0, (const Bytef *)chunks[i].t, 4);
        if (chunks[i].n) crc = crc32(crc, chunks[i].d, chunks[i].n);
        uint8_t cb[4] = { (uint8_t)(crc >> 24), (uint8_t)(crc >> 16), (uint8_t)(crc >> 8), (uint8_t)crc };
        fwrite(cb, 1, 4, f);
    }
    fclose(f);
    free(raw); free(z);
}

#endif
static int g_map_done;
static void on_mapped(WGPUMapAsyncStatus st, WGPUStringView msg, void *u1, void *u2) {
    (void)st; (void)msg; (void)u1; (void)u2;
    g_map_done = 1;
}

/* Reads the back buffer back and writes it as a PNG (work recorded so far is submitted first). */
#ifndef __EMSCRIPTEN__
static void q_drain(void);
static void capture_tex(WGPUTexture tex, uint32_t w, uint32_t h, const char *path) {
    q_drain();
    pt_wait(1);   /* the wait below polls the device: keep the present thread out of it */
    uint32_t stride = (w * 4 + 255) & ~255u;
    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.size = (uint64_t)stride * h;
    bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(g_dev, &bd);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_dev, NULL);
    WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    src.texture = tex;
    WGPUTexelCopyBufferInfo dst = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    dst.buffer = buf;
    dst.layout.bytesPerRow = stride;
    dst.layout.rowsPerImage = h;
    WGPUExtent3D ex = { w, h, 1 };
    wgpuCommandEncoderCopyTextureToBuffer(enc, &src, &dst, &ex);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    queue_submit(cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    WGPUBufferMapCallbackInfo mci = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    mci.mode = WGPUCallbackMode_AllowProcessEvents;
    mci.callback = on_mapped;
    g_map_done = 0;
    wgpuBufferMapAsync(buf, WGPUMapMode_Read, 0, bd.size, mci);
    while (!g_map_done) wgpuInstanceProcessEvents(g_inst);
    const uint8_t *data = wgpuBufferGetConstMappedRange(buf, 0, bd.size);
    write_png(path, data, w, h, stride);
    fprintf(stderr, "[wgpu] screenshot %s\n", path);
    wgpuBufferUnmap(buf);
    wgpuBufferDestroy(buf);
    wgpuBufferRelease(buf);
}
static void capture_png(const char *path) { capture_tex(g_bb, (uint32_t)g_surf_w, (uint32_t)g_surf_h, path); }

#endif

static WGPUAddressMode addr_mode(uint32_t m) {
    switch (m) { case 2: return WGPUAddressMode_MirrorRepeat;
                 case 4: if (g_gpu_border) return (WGPUAddressMode)0x00000004 /* WGPUNativeAddressMode_ClampToBorder */; return WGPUAddressMode_ClampToEdge;
                 case 3: case 5: return WGPUAddressMode_ClampToEdge;
                 default: return WGPUAddressMode_Repeat; }
}
/* sampler key: bits 0-9 filters/address modes (render_rec.c sampler_key), bits 10-11 the border colour class
   (0 transparent black, 1 opaque black, 2 opaque white): Vulkan only has those three */
static WGPUSampler sampler_for(uint32_t key) {
    key &= 4095;
    uint32_t mag = key & 1, min = (key >> 1) & 1, mip = (key >> 2) & 3, au = (key >> 4) & 7, av = (key >> 7) & 7, bcl = (key >> 10) & 3;
    if (g_samplers[key]) return g_samplers[key];
    WGPUSamplerDescriptor sd = WGPU_SAMPLER_DESCRIPTOR_INIT;
    sd.magFilter = mag ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
    sd.minFilter = min ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
    sd.mipmapFilter = mip == 2 ? WGPUMipmapFilterMode_Linear : WGPUMipmapFilterMode_Nearest;
    sd.lodMaxClamp = mip ? 32.f : 0.f;
    sd.addressModeU = addr_mode(au);
    sd.addressModeV = addr_mode(av);
#ifndef __EMSCRIPTEN__
    WGPUSamplerDescriptorExtras ex;
    if (g_gpu_border && (au == 4 || av == 4)) {
        memset(&ex, 0, sizeof ex);
        ex.chain.sType = (WGPUSType)WGPUSType_SamplerDescriptorExtras;
        ex.samplerBorderColor = bcl == 2 ? WGPUSamplerBorderColor_OpaqueWhite : bcl == 1 ? WGPUSamplerBorderColor_OpaqueBlack
                                                                                  : WGPUSamplerBorderColor_TransparentBlack;
        sd.nextInChain = &ex.chain;
        fprintf(stderr, "[wgpu] border sampler key %03x colour class %u\n", key, bcl);
    }
#endif
    g_samplers[key] = wgpuDeviceCreateSampler(g_dev, &sd);
    return g_samplers[key];
}

/* ------------------------------------------------------------------ pipeline warm-up cache
 *
 * Creating a pipeline the first time costs 25-56 ms on Mali (naga's SPIR-V output plus the driver's
 * compile), which showed as a stutter whenever new players or objects appeared. wgpu-native has no
 * pipeline-cache API, so every pipeline's recipe is kept instead: userdata/gpu_pipelines.bin holds the
 * WGSL of each translated shader ('S' records) and each pipeline's state with its shaders named by the
 * hash of their WGSL ('P' records), appended as they are first created. At the next launch a background
 * thread recreates them all in first-use order, and pipeline_for takes the finished ones (or waits for
 * one being built). Shader modules are shared by WGSL hash, so the game thread doesn't parse them again.
 * RC_PIPE_CACHE=0 turns it off. */

static WGPUShaderModule make_module(const char *fragment);

static uint64_t fnv64(const void *p, size_t n, uint64_t h) {
    for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t *)p)[i]) * 0x100000001b3ull;
    return h;
}
#define FNV64 0xcbf29ce484222325ull

/* a pipeline independent of this launch's shader handles: k.ps / k.vs are just 0 or 1 */
typedef struct { PKey k; uint64_t psh, vsh; } WKey;
static WKey wkey(const PKey *k, uint64_t psh, uint64_t vsh) {
    WKey w;
    memset(&w, 0, sizeof w);
    w.k = *k; w.k.ps = psh != 0; w.k.vs = vsh != 0; w.psh = psh; w.vsh = vsh;
    return w;
}

#ifndef __EMSCRIPTEN__
static pthread_mutex_t g_wlock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_wcond = PTHREAD_COND_INITIALIZER;
#define WLOCK() pthread_mutex_lock(&g_wlock)
#define WUNLOCK() pthread_mutex_unlock(&g_wlock)
#else
#define WLOCK() ((void)0)
#define WUNLOCK() ((void)0)
#endif

/* shader modules by the hash of their WGSL */
#define NMODS 4096
static struct { uint64_t h; WGPUShaderModule m; } g_mods[NMODS];
static WGPUShaderModule module_cached(const char *wgsl, uint64_t h) {
    WLOCK();
    uint32_t i = (uint32_t)h % NMODS, n = 0;
    while (g_mods[i].m && g_mods[i].h != h && n++ < NMODS) i = (i + 1) % NMODS;
    WGPUShaderModule m = g_mods[i].h == h ? g_mods[i].m : NULL;
    WUNLOCK();
    if (m) return m;
    m = make_module(wgsl);
    WLOCK();
    if (g_mods[i].m && g_mods[i].h == h) { wgpuShaderModuleRelease(m); m = g_mods[i].m; }   /* the other thread won */
    else if (!g_mods[i].m) { g_mods[i].h = h; g_mods[i].m = m; }
    WUNLOCK();
    return m;
}

/* the records already in the file (shader hashes and pipeline-key hashes), so nothing is written twice */
#define NKNOWN 16384
static uint64_t g_known[NKNOWN];
static int known_add(uint64_t h) {   /* 1 if it wasn't known yet */
    if (!h) h = 1;
    uint32_t i = (uint32_t)h % NKNOWN;
    for (uint32_t n = 0; n < NKNOWN; n++, i = (i + 1) % NKNOWN) {
        if (g_known[i] == h) return 0;
        if (!g_known[i]) { g_known[i] = h; return 1; }
    }
    return 0;
}

/* pipelines from the file: p is NULL while being built; taken once pipeline_for owns it (or builds it) */
#define NWARM 4096
typedef struct { WKey w; WGPURenderPipeline p; int used, taken, pending; } WEntry;   /* pending: queued for the async builder */
static WEntry g_warm[NWARM];
static WEntry *warm_slot(const WKey *w) {
    uint64_t h = fnv64(w, sizeof *w, FNV64);
    uint32_t i = (uint32_t)h % NWARM;
    for (uint32_t n = 0; n < NWARM; n++, i = (i + 1) % NWARM)
        if (!g_warm[i].used || !memcmp(&g_warm[i].w, w, sizeof *w)) return &g_warm[i];
    return NULL;
}

#ifndef __EMSCRIPTEN__
extern char g_user_root[];
static FILE *g_pcf;   /* the cache file, open for appending (NULL: cache off) */
static uint8_t *g_pc_data;   /* the file as read at startup: the warm-up thread's work */
static size_t g_pc_len;
#define PC_MAGIC 0x31435052u   /* "RPC1" */

static void pc_write(const void *rec, size_t n) {
    if (!g_pcf) return;
    fwrite(rec, 1, n, g_pcf);
    fflush(g_pcf);
}
static void pc_shader(uint64_t h, int nin, const char *wgsl) {
    WLOCK();
    int fresh = g_pcf && known_add(h);
    WUNLOCK();
    if (!fresh) return;
    uint32_t len = (uint32_t)strlen(wgsl);
    uint8_t hd[17];
    hd[0] = 'S'; memcpy(hd + 1, &h, 8); memcpy(hd + 9, &nin, 4); memcpy(hd + 13, &len, 4);
    WLOCK();
    pc_write(hd, sizeof hd);
    pc_write(wgsl, len);
    WUNLOCK();
}
static void pc_pipeline(const WKey *w) {
    uint8_t rec[1 + sizeof *w];
    rec[0] = 'P'; memcpy(rec + 1, w, sizeof *w);
    WLOCK();
    if (g_pcf && known_add(fnv64(w, sizeof *w, FNV64))) pc_write(rec, sizeof rec);
    WUNLOCK();
}
#else
static void pc_shader(uint64_t h, int nin, const char *wgsl) { (void)h; (void)nin; (void)wgsl; }
static void pc_pipeline(const WKey *w) { (void)w; }
#endif

/* ------------------------------------------------------------------ pixel shaders */

typedef struct { WGPUShaderModule module; uint64_t hash; } PsProgram;
/* Fixed-function draws get a fragment shader made for their texture-stage setup instead of the uber shader:
   the uber shader samples both textures on every pixel and contains a discard (alpha test) even when the test
   is off, and on Mali a shader that can discard turns off early depth testing and hidden-pixel removal, so
   every layer of overdraw was fully shaded. A specialised one samples only the textures its stages read and
   has a discard only with the alpha test on. Pipelines carry it as key.ps = FF_TAG | table index.
   RC_FF_GENERIC=1: the uber shader for everything. */
#define FF_TAG 0x80000000u
#define FF_MAX 2048
typedef struct { uint32_t d[18]; PsProgram prog; int used; } FfEntry;
static FfEntry g_ff[FF_MAX];
static int ff_tex_used(const uint32_t *st, const uint32_t *sb) {
    #define T(a) (((a) & 7u) == 2u)
    if (T(st[1]) || T(st[2]) || T(sb[3] & 255) || st[0] == 13 || st[0] == 15) return 1;
    if (st[3] > 1 && (T(sb[0]) || T(sb[1]) || T(sb[3] >> 8) || st[3] == 13 || st[3] == 15)) return 1;
    return 0;
    #undef T
}
static PsProgram *ff_lookup(uint32_t tag) { return g_ff[tag & ~FF_TAG].used ? &g_ff[tag & ~FF_TAG].prog : NULL; }
static uint64_t fnv64(const void *p, size_t n, uint64_t h);
static WGPUShaderModule module_cached(const char *wgsl, uint64_t h);
static void pc_shader(uint64_t h, int nin, const char *wgsl);
static uint32_t ff_tag(const Uniforms *u) {
    static int off = -1;
    if (off < 0) off = getenv("RC_FF_GENERIC") != NULL;
    if (off) return 0;
    uint32_t d[18];
    memset(d, 0, sizeof d);
    memcpy(d, u->st0, 16); memcpy(d + 4, u->st0b, 16);
    if (u->st0[0] != 1 && u->st1[0] != 1) { memcpy(d + 8, u->st1, 16); memcpy(d + 12, u->st1b, 16); }
    else d[8] = 1;
    if (u->st0[0] == 1) memset(d + 1, 0, 28);   /* stage 0 off: nothing else of the stages matters */
    d[16] = u->flags[1] == 8 ? 0 : u->flags[1];  /* ALWAYS = no test */
    d[17] = u->flags[2];
    uint32_t i = (uint32_t)fnv64(d, sizeof d, FNV64) % FF_MAX;
    for (uint32_t n = 0; n < FF_MAX; n++, i = (i + 1) % FF_MAX) {
        FfEntry *e = &g_ff[i];
        if (e->used && !memcmp(e->d, d, sizeof d)) return FF_TAG | i;
        if (e->used) continue;
        /* new variant: the uber shader's helper functions, then a fragment stage written out for this setup */
        const char *fsp = strstr(WGSL_FF, "@fragment");
        size_t pre = (size_t)(fsp - WGSL_FF);
        char *w = malloc(pre + 2048);
        memcpy(w, WGSL_FF, pre);
        int o = (int)pre;
        o += sprintf(w + o, "@fragment fn fs(i: VO) -> @location(0) vec4<f32> {\n  var cur = i.diffuse;\n");
        for (int s = 0; s < 2 && d[8 * s] != 1; s++) {
            const uint32_t *st = d + 8 * s, *sb = d + 8 * s + 4;
            if (ff_tex_used(st, sb))
                o += sprintf(w + o, "  let tx%d = textureSample(tex%d, smp%d, i.%s.xy);\n", s, s, s, sb[2] == 1 ? "tc1" : "tc0");
            else o += sprintf(w + o, "  let tx%d = vec4<f32>(1.0);\n", s);
            o += sprintf(w + o, "  cur = stage(vec4<u32>(%uu, %uu, %uu, %uu), vec4<u32>(%uu, %uu, %uu, %uu), cur, tx%d, i.diffuse, i.specular);\n",
                         st[0], st[1], st[2], st[3], sb[0], sb[1], sb[2], sb[3], s);
        }
        if (d[17] == 1) o += sprintf(w + o, "  cur = vec4<f32>(saturate(cur.rgb + i.specular.rgb), cur.a);\n");
        if (d[16]) o += sprintf(w + o, "  if (!alpha_ok(cur.a)) { discard; }\n");
        o += sprintf(w + o, "  return cur;\n}\n");
        e->prog.hash = fnv64(w, (size_t)o, FNV64) | 1;
        e->prog.module = module_cached(w, e->prog.hash);
        pc_shader(e->prog.hash, -1, w);
        free(w);
        memcpy(e->d, d, sizeof d);
        e->used = 1;
        return FF_TAG | i;
    }
    return 0;
}

char *ps_to_wgsl(const uint32_t *code, uint32_t n, char *why, size_t why_n);

/* NULL when the shader can't be translated: such draws use the fixed-function stage instead */
static PsProgram *ps_program(uint32_t handle) {
    Shader *sh = shader_get(handle);
    if (!sh || !sh->pixel) return NULL;
    if (sh->backend) return sh->backend == (void *)1 ? NULL : sh->backend;
    char why[96] = "";
    char *fs = ps_to_wgsl(sh->code, sh->code_len, why, sizeof why);
    if (!fs) {
        fprintf(stderr, "[wgpu] pixel shader %x not translated: %s\n", handle, why);
        sh->backend = (void *)1;
        return NULL;
    }
    PsProgram *p = calloc(1, sizeof *p);
    p->hash = fnv64(fs, strlen(fs), FNV64) | 1;
    p->module = module_cached(fs, p->hash);
    pc_shader(p->hash, -1, fs);
    if (getenv("RC_DUMP_PS")) {
        fprintf(stderr, "[ps %x] tokens:", handle);
        for (uint32_t k = 0; k < sh->code_len; k++) fprintf(stderr, " %08x", sh->code[k]);
        fprintf(stderr, "\n%s", fs);
    }
    free(fs);
    sh->backend = p;
    return p;
}

/* ------------------------------------------------------------------ vertex shaders */

#include "vshader.h"

typedef struct { WGPUShaderModule module; uint64_t hash; int nin; } VsGpu;

/* the vertex stage of a vs.1.1 shader (entry point "vsp"); NULL if it can't be translated */
static VsGpu *vs_gpu(uint32_t handle) {
    Shader *sh = shader_get(handle);
    if (!sh || sh->pixel) return NULL;
    if (sh->gpu) return sh->gpu == (void *)1 ? NULL : sh->gpu;
    char *vs = vs_to_wgsl(sh);
    if (!vs) { fprintf(stderr, "[wgpu] vertex shader %x not translated\n", handle); sh->gpu = (void *)1; return NULL; }
    if (getenv("RC_DUMP_VS_WGSL")) fprintf(stderr, "[vs %x]\n%s", handle, vs);
    VsGpu *g = calloc(1, sizeof *g);
    g->hash = fnv64(vs, strlen(vs), FNV64) | 1;
    g->nin = vs_gpu_inputs(sh);
    g->module = module_cached(vs, g->hash);
    pc_shader(g->hash, g->nin, vs);
    free(vs);
    sh->gpu = g;
    return g;
}

/* ------------------------------------------------------------------ pipelines */

typedef struct { PKey k; WGPURenderPipeline p; } PEntry;
#define NPIPES 2048
static PEntry g_pipes[NPIPES];

static WGPUBlendFactor bf(uint32_t b) {
    switch (b) {
    case 1: return WGPUBlendFactor_Zero; case 2: return WGPUBlendFactor_One;
    case 3: return WGPUBlendFactor_Src; case 4: return WGPUBlendFactor_OneMinusSrc;
    case 5: return WGPUBlendFactor_SrcAlpha; case 6: return WGPUBlendFactor_OneMinusSrcAlpha;
    case 7: return WGPUBlendFactor_DstAlpha; case 8: return WGPUBlendFactor_OneMinusDstAlpha;
    case 9: return WGPUBlendFactor_Dst; case 10: return WGPUBlendFactor_OneMinusDst;
    case 11: return WGPUBlendFactor_SrcAlphaSaturated;
    case 12: return WGPUBlendFactor_OneMinusSrcAlpha;  /* BOTHSRCALPHA (approximation) */
    case 13: return WGPUBlendFactor_SrcAlpha;          /* BOTHINVSRCALPHA (approximation) */
    default: return WGPUBlendFactor_One;
    }
}
static WGPUCompareFunction cmpf(uint32_t f) {
    switch (f) {
    case 1: return WGPUCompareFunction_Never; case 2: return WGPUCompareFunction_Less;
    case 3: return WGPUCompareFunction_Equal; case 4: return WGPUCompareFunction_LessEqual;
    case 5: return WGPUCompareFunction_Greater; case 6: return WGPUCompareFunction_NotEqual;
    case 7: return WGPUCompareFunction_GreaterEqual; default: return WGPUCompareFunction_Always;   /* 8, and 0 = unset */
    }
}
static WGPUStencilOperation sop(uint32_t o) {   /* D3DSTENCILOP (0: unset = keep) */
    switch (o) {
    case 2: return WGPUStencilOperation_Zero; case 3: return WGPUStencilOperation_Replace;
    case 4: return WGPUStencilOperation_IncrementClamp; case 5: return WGPUStencilOperation_DecrementClamp;
    case 6: return WGPUStencilOperation_Invert; case 7: return WGPUStencilOperation_IncrementWrap;
    case 8: return WGPUStencilOperation_DecrementWrap; default: return WGPUStencilOperation_Keep;
    }
}
static WGPUBlendOperation bop(uint32_t o) {
    switch (o) { case 2: return WGPUBlendOperation_Subtract; case 3: return WGPUBlendOperation_ReverseSubtract;
                 case 4: return WGPUBlendOperation_Min; case 5: return WGPUBlendOperation_Max;
                 default: return WGPUBlendOperation_Add; }
}

/* fm: the fragment stage's module (a pixel shader's, else the fixed-function one, which also has the
   fixed-function vertex stage); vm: the game's vertex shader with nin inputs, or NULL */
static WGPURenderPipeline make_pipeline(const PKey *k, WGPUShaderModule fm, WGPUShaderModule vm, int nin) {
    WGPUVertexAttribute at[7] = {
        { NULL, WGPUVertexFormat_Float32x4, 0, 0 }, { NULL, WGPUVertexFormat_Uint32, 16, 1 },
        { NULL, WGPUVertexFormat_Uint32, 20, 2 }, { NULL, WGPUVertexFormat_Float32x4, 24, 3 },
        { NULL, WGPUVertexFormat_Float32x4, 40, 4 }, { NULL, WGPUVertexFormat_Float32x4, 56, 5 },
        { NULL, WGPUVertexFormat_Float32x4, 72, 6 },
    };
    WGPUVertexBufferLayout vbl = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
    vbl.arrayStride = sizeof(CVert); vbl.attributeCount = 7; vbl.attributes = at;
    vbl.stepMode = WGPUVertexStepMode_Vertex;
    WGPUBlendState bs = WGPU_BLEND_STATE_INIT;
    bs.color.srcFactor = bf(k->src); bs.color.dstFactor = bf(k->dst); bs.color.operation = bop(k->op);
    bs.alpha = bs.color;
    WGPUColorTargetState ct = WGPU_COLOR_TARGET_STATE_INIT;
    ct.format = g_surf_fmt;
    ct.blend = k->blend ? &bs : NULL;
    ct.writeMask = (k->cmask & 1 ? WGPUColorWriteMask_Red : 0) | (k->cmask & 2 ? WGPUColorWriteMask_Green : 0) |
                   (k->cmask & 4 ? WGPUColorWriteMask_Blue : 0) | (k->cmask & 8 ? WGPUColorWriteMask_Alpha : 0);
    WGPUFragmentState fs = WGPU_FRAGMENT_STATE_INIT;
    fs.module = fm; fs.entryPoint = sv("fs"); fs.targetCount = 1; fs.targets = &ct;
    WGPUDepthStencilState ds = WGPU_DEPTH_STENCIL_STATE_INIT;
    ds.format = WGPUTextureFormat_Depth24PlusStencil8;
    ds.depthWriteEnabled = k->zen && k->zwrite ? WGPUOptionalBool_True : WGPUOptionalBool_False;
    ds.depthCompare = k->zen ? cmpf(k->zfunc) : WGPUCompareFunction_Always;
    if (k->sten & 1) {
        uint32_t t = k->sten, t2 = k->sten2;
        ds.stencilFront.compare = cmpf((t >> 4) & 15);
        ds.stencilFront.failOp = sop((t >> 8) & 15); ds.stencilFront.depthFailOp = sop((t >> 12) & 15); ds.stencilFront.passOp = sop((t >> 16) & 15);
        if ((t >> 20) & 1) {   /* TWOSIDEDSTENCILMODE: the CCW ops for counter-clockwise (back, with front face CW) triangles */
            ds.stencilBack.compare = cmpf((t >> 24) & 15);
            ds.stencilBack.failOp = sop(t2 & 15); ds.stencilBack.depthFailOp = sop((t2 >> 4) & 15); ds.stencilBack.passOp = sop((t2 >> 8) & 15);
        } else ds.stencilBack = ds.stencilFront;
        ds.stencilReadMask = (t2 >> 12) & 255; ds.stencilWriteMask = (t2 >> 20) & 255;
    }
    WGPURenderPipelineDescriptor pd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    pd.layout = g_pl;
    pd.vertex.module = fm; pd.vertex.entryPoint = sv("vs");
    pd.vertex.bufferCount = 1; pd.vertex.buffers = &vbl;
    WGPUVertexAttribute vat[16];
    if (vm) {   /* the game's vertex shader: one vec4 per input register it reads */
        for (int a = 0; a < nin && a < 16; a++)
            vat[a] = (WGPUVertexAttribute){ NULL, WGPUVertexFormat_Float32x4, 16u * (uint32_t)a, (uint32_t)a };
        vbl.arrayStride = 16u * (uint32_t)(nin > 0 ? nin : 1);
        vbl.attributeCount = (size_t)(nin > 0 ? nin : 0); vbl.attributes = vat;
        pd.layout = g_pl_vs;
        pd.vertex.module = vm; pd.vertex.entryPoint = sv("vsp");
    }
    pd.primitive.topology = (WGPUPrimitiveTopology)k->topo;
    pd.primitive.frontFace = WGPUFrontFace_CW;
    pd.primitive.cullMode = k->cull == 2 ? WGPUCullMode_Front : k->cull == 3 ? WGPUCullMode_Back : WGPUCullMode_None;
    pd.depthStencil = &ds;
    pd.fragment = &fs;
    pd.multisample.count = 1; pd.multisample.mask = 0xFFFFFFFFu;
    return wgpuDeviceCreateRenderPipeline(g_dev, &pd);
}

/* g_warm maps every pipeline made this session (and those from the file) by WKey: the game creates
   identical shaders under new handles for each player and object, so most new PKeys are a pipeline that
   already exists. Returns it (waiting if the warm-up thread is building it), else NULL after marking the
   slot taken, so the warm-up thread leaves it to the caller, who hands the result to warm_put. */
static WGPURenderPipeline warm_take(const WKey *w) {
    WGPURenderPipeline p = NULL;
    WLOCK();
    WEntry *e = warm_slot(w);
    if (e && e->used) {
#ifndef __EMSCRIPTEN__
        if (!e->taken) {
            if (!e->p) g_pc_waits++;
            while (!e->p && !e->taken) pthread_cond_wait(&g_wcond, &g_wlock);
            if (e->p) g_pc_hits++;
        } else
#endif
        if (e->p) g_pc_shared++;
        p = e->p;
    } else if (e) {
        e->used = 1; e->w = *w;
    }
    if (e) e->taken = 1;
    WUNLOCK();
    return p;
}
static void warm_put(const WKey *w, WGPURenderPipeline p) {
    WLOCK();
    WEntry *e = warm_slot(w);
    if (e && (!e->used || !e->p)) { e->used = e->taken = 1; e->w = *w; e->p = p; }
    WUNLOCK();
}

#ifndef __EMSCRIPTEN__
/* A pipeline nothing has built yet (a surface type seen for the first time) takes ~40 ms on Mali, a hitch on
   entering new places. It is built on a worker thread instead, and draws that need it are skipped for the
   few frames that takes: RC_ASYNC_PIPELINES=1 (off by default: things pop in). By default they are built in place,
   once ever: userdata/gpu_pipelines.bin keeps every pipeline seen and warms them all up at start. */
typedef struct { WKey w; PKey k; WGPUShaderModule fm, vm; int nin; } PJob;
#define NPJOB 512
static PJob g_pjob[NPJOB];
static uint32_t g_pjob_head, g_pjob_tail;
static pthread_cond_t g_pjob_cond = PTHREAD_COND_INITIALIZER;
static void *pjob_thread(void *arg) {
    (void)arg;
    for (;;) {
        WLOCK();
        while (g_pjob_head == g_pjob_tail) pthread_cond_wait(&g_pjob_cond, &g_wlock);
        PJob j = g_pjob[g_pjob_tail % NPJOB];
        WUNLOCK();
        WGPURenderPipeline p = make_pipeline(&j.k, j.fm, j.vm, j.nin);
        WLOCK();
        g_pjob_tail++;
        WEntry *e = warm_slot(&j.w);
        if (e) { e->p = p; e->pending = 0; }
        pthread_cond_broadcast(&g_wcond);
        WUNLOCK();
    }
    return NULL;
}
/* the pipeline if it is ready, else NULL after queuing it (once) */
static WGPURenderPipeline warm_async(const WKey *w, const PKey *k, WGPUShaderModule fm, WGPUShaderModule vm, int nin) {
    static int started;
    if (!started) {
        started = 1;
        pthread_t t;
        if (pthread_create(&t, NULL, pjob_thread, NULL) == 0) pthread_detach(t);
    }
    WGPURenderPipeline p = NULL;
    int queued = 0;
    WLOCK();
    WEntry *e = warm_slot(w);
    if (e && e->used && e->p) { p = e->p; if (!e->taken) g_pc_hits++; else g_pc_shared++; e->taken = 1; }
    else if (e && (!e->used || (e->taken && !e->pending)) && g_pjob_head - g_pjob_tail < NPJOB) {
        /* new, or the warm-up thread gave it up: queue it (one still warming up from the file is left to finish) */
        e->used = e->taken = e->pending = 1; e->w = *w;
        PJob *j = &g_pjob[g_pjob_head++ % NPJOB];
        j->w = *w; j->k = *k; j->fm = fm; j->vm = vm; j->nin = nin;
        pthread_cond_signal(&g_pjob_cond);
        queued = 1;
        g_pc_built++;
    }
    WUNLOCK();
    if (queued) pc_pipeline(w);
    return p;
}
#endif

static WGPURenderPipeline pipeline_for(const PKey *k) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < sizeof *k / 4; i++) h = (h ^ ((const uint32_t *)k)[i]) * 16777619u;
    for (uint32_t i = 0; i < NPIPES; i++) {
        PEntry *e = &g_pipes[(h + i) % NPIPES];
        if (e->p && !memcmp(&e->k, k, sizeof *k)) return e->p;
        if (!e->p) {
            PsProgram *ps = !k->ps ? NULL : (k->ps & FF_TAG) ? ff_lookup(k->ps) : ps_program(k->ps);
            VsGpu *vs = NULL;
            if (k->vs && !(vs = vs_gpu(k->vs))) return NULL;
            WKey w = wkey(k, ps ? ps->hash : 0, vs ? vs->hash : 0);
#ifndef __EMSCRIPTEN__
            static int sync = -1;
            if (sync < 0) sync = getenv("RC_ASYNC_PIPELINES") == NULL;   /* default: built in place, no pop-in; the file cache warms them at start */
            if (!sync) {
                WGPURenderPipeline pa = warm_async(&w, k, ps ? ps->module : g_shader, vs ? vs->module : NULL, vs ? vs->nin : 0);
                if (!pa) return NULL;   /* being built: skip the draw this frame */
                e->k = *k;
                e->p = pa;
                return pa;
            }
#endif
            WGPURenderPipeline p = warm_take(&w);
            if (!p) {
                extern double g_hx[]; extern uint32_t g_hxb[];
                double t_p = host_seconds();
                p = make_pipeline(k, ps ? ps->module : g_shader, vs ? vs->module : NULL, vs ? vs->nin : 0);
                g_hx[3] += host_seconds() - t_p; g_hxb[3]++;
                warm_put(&w, p);
                pc_pipeline(&w);
                g_pc_built++;
            }
            e->k = *k;
            e->p = p;
            return e->p;
        }
    }
    static int warned;
    if (!warned++) fprintf(stderr, "[wgpu] pipeline cache full\n");
    return NULL;
}

#ifndef __EMSCRIPTEN__
/* the 'S' record with hash h in the startup file: its WGSL (not NUL-terminated) and input count */
static const uint8_t *pc_find_shader(uint64_t h, uint32_t *len, int *nin) {
    for (size_t o = 16; o < g_pc_len;) {
        uint8_t t = g_pc_data[o];
        if (t == 'P') { o += 1 + sizeof(WKey); continue; }
        uint64_t sh; uint32_t n;
        memcpy(&sh, g_pc_data + o + 1, 8); memcpy(nin, g_pc_data + o + 9, 4); memcpy(&n, g_pc_data + o + 13, 4);
        if (sh == h) { *len = n; return g_pc_data + o + 17; }
        o += 17 + n;
    }
    return NULL;
}
static WGPUShaderModule pc_module(uint64_t h, int *nin) {
    uint32_t len;
    const uint8_t *s = pc_find_shader(h, &len, nin);
    if (!s) return NULL;
    char *z = malloc(len + 1);
    memcpy(z, s, len); z[len] = 0;
    WGPUShaderModule m = module_cached(z, h);
    free(z);
    return m;
}

static void *warm_thread(void *arg) {
    (void)arg;
    double t0 = host_seconds();
    for (size_t o = 16; o < g_pc_len;) {
        if (g_pc_data[o] == 'S') { uint32_t n; memcpy(&n, g_pc_data + o + 13, 4); o += 17 + n; continue; }
        WKey w;
        memcpy(&w, g_pc_data + o + 1, sizeof w);
        o += 1 + sizeof w;
        WLOCK();
        WEntry *e = warm_slot(&w);
        int mine = e && !e->used;
        if (mine) { e->used = 1; e->w = w; }
        WUNLOCK();
        if (!mine) continue;   /* the game got there first */
        int nin = 0, unused;
        WGPUShaderModule fm = w.psh ? pc_module(w.psh, &unused) : g_shader;
        WGPUShaderModule vm = w.vsh ? pc_module(w.vsh, &nin) : NULL;
        WGPURenderPipeline p = fm && (vm || !w.vsh) ? make_pipeline(&w.k, fm, vm, nin) : NULL;
        WLOCK();
        if (p) { e->p = p; g_pc_warmed++; }
        else e->taken = 1;   /* a shader is missing: whoever waits builds it */
        pthread_cond_broadcast(&g_wcond);
        WUNLOCK();
    }
    fprintf(stderr, "[wgpu] pipeline warm-up: %u pipelines in %.1f s\n", g_pc_warmed, host_seconds() - t0);
    free(g_pc_data);
    g_pc_data = NULL;
    return NULL;
}

/* open userdata/gpu_pipelines.bin: keep a valid one to warm up from and append to, else start it over */
static void pipe_cache_start(void) {
    const char *e = getenv("RC_PIPE_CACHE");
    if (e && !atoi(e)) return;
    char path[4200];
    snprintf(path, sizeof path, "%s/gpu_pipelines.bin", g_user_root);
    /* the file is only valid for this surface format and these shared shader parts */
    uint32_t hdr[4] = { PC_MAGIC, (uint32_t)g_surf_fmt, 0, (uint32_t)sizeof(WKey) };
    uint64_t sh = fnv64(WGSL_COMMON, strlen(WGSL_COMMON), FNV64);
    hdr[2] = (uint32_t)fnv64(WGSL_FF, strlen(WGSL_FF), sh);
    size_t good = 0;
    FILE *f = fopen(path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (n >= 16) {
            g_pc_data = malloc((size_t)n);
            g_pc_len = fread(g_pc_data, 1, (size_t)n, f);
        }
        fclose(f);
    }
    if (g_pc_data && g_pc_len >= 16 && !memcmp(g_pc_data, hdr, 16)) {
        /* index what is there; a record cut short by the app being killed ends the file */
        size_t o = 16;
        uint32_t npipes = 0;
        while (o < g_pc_len) {
            uint8_t t = g_pc_data[o];
            if (t == 'P' && o + 1 + sizeof(WKey) <= g_pc_len) {
                known_add(fnv64(g_pc_data + o + 1, sizeof(WKey), FNV64));
                o += 1 + sizeof(WKey); npipes++;
            } else if (t == 'S' && o + 17 <= g_pc_len) {
                uint64_t h; uint32_t len;
                memcpy(&h, g_pc_data + o + 1, 8); memcpy(&len, g_pc_data + o + 13, 4);
                if (o + 17 + len > g_pc_len) break;
                known_add(h);
                o += 17 + len;
            } else break;
        }
        good = o;
        g_pc_len = o;
        fprintf(stderr, "[wgpu] pipeline cache: %u pipelines in %s\n", npipes, path);
    } else {
        free(g_pc_data);
        g_pc_data = NULL; g_pc_len = 0;
    }
    if (good) {
        if (truncate(path, (off_t)good) != 0) good = 0;
    }
    g_pcf = fopen(path, good ? "ab" : "wb");
    if (!g_pcf) { fprintf(stderr, "[wgpu] pipeline cache: can't write %s\n", path); return; }
    if (!good) pc_write(hdr, sizeof hdr);
    if (g_pc_data) {
        pthread_t t;
        if (pthread_create(&t, NULL, warm_thread, NULL) == 0) pthread_detach(t);
        else { free(g_pc_data); g_pc_data = NULL; }
    }
}
#endif

/* ------------------------------------------------------------------ frames and passes */

static WGPUTextureView g_tgt_color, g_tgt_depth;   /* attachments of the current pass target (NULL: back buffer) */
/* The current depth attachment is a scratch buffer: 1 the game set no depth surface for this target, so its contents
   mean nothing and passes clear it and discard it instead of loading and storing it; 2 the game's depth surface is
   another size (Direct3D 9 allows a larger one), kept loaded and stored because the scene relies on it across passes. On a
   tile-based GPU (Mali) every loaded/stored full-screen attachment is memory bandwidth; the post-process passes
   (glow chain, StretchRect copies) are all of this kind. RC_SCRATCH_KEEP=1: load and store as before. */
static int g_tgt_depth_scratch;
static uint32_t g_tgt_w, g_tgt_h, g_tgt_lw = 640, g_tgt_lh = 480;   /* GPU size and Direct3D size */
static GSurf *g_tgt_gs;                                 /* NULL when drawing to the back buffer */

static void end_pass(void) {
    if (g_pass) { wgpuRenderPassEncoderEnd(g_pass); wgpuRenderPassEncoderRelease(g_pass); g_pass = NULL; }
}

/* Textures the game freed. Releasing a WebGPU texture only drops a reference: its GPU memory goes when the
   browser garbage-collects it, and screens that recreate textures every frame (team select) run the GPU out
   of memory first (Firefox). destroy() frees it at once, but draws recorded into the open command encoder
   may still use it, so the destroy waits until that work has been submitted. */
/* Bind groups of draws, cached by their 4 texture views and 4 sampler keys. A WebGPU bind group can't be
   destroyed, only dropped for the browser's garbage collector, and a new one per draw (hundreds a frame)
   outran it on Firefox/D3D12, where they hold descriptor-heap slots: "Out of Memory", then invalid bind
   groups. Entries using a view are dropped when that view is released (pointers can be reused). */
#define BG_CACHE 4096
#define BG_PROBES 8
typedef struct { WGPUTextureView v[4]; uint32_t s[4]; WGPUBindGroup bg; uint32_t stamp; } BgEntry;
static uint32_t g_bg_freed;   /* bind groups released so far: a new one may reuse a freed one's address (exec_draw's state cache) */
static uint32_t g_pass_gen;   /* render passes begun (exec_draw's state cache) */
static BgEntry g_bgc[BG_CACHE];
static uint32_t g_bg_stamp;
uint32_t g_gpu_bind_groups_made;   /* for the [mem] log */
static uint32_t bg_hash(WGPUTextureView const v[4], const uint32_t s[4]) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 4; i++) { h = (h ^ (uint64_t)(uintptr_t)v[i]) * 1099511628211ull; h = (h ^ s[i]) * 1099511628211ull; }
    return (uint32_t)(h ^ (h >> 32));
}
static void bg_forget_view(WGPUTextureView view) {
    for (int i = 0; i < BG_CACHE; i++) {
        BgEntry *e = &g_bgc[i];
        if (e->bg && (e->v[0] == view || e->v[1] == view || e->v[2] == view || e->v[3] == view)) {
            wgpuBindGroupRelease(e->bg);   /* recorded commands keep their own reference */
            g_bg_freed++;
            e->bg = NULL;
        }
    }
}
static void release_view(WGPUTextureView view) { bg_forget_view(view); wgpuTextureViewRelease(view); }

static WGPUTexture *g_retired;
static size_t g_nretired, g_capretired;
static void retire_texture(WGPUTexture t) {
    if (g_nretired == g_capretired) {
        g_capretired = g_capretired ? g_capretired * 2 : 64;
        g_retired = realloc(g_retired, g_capretired * sizeof *g_retired);
    }
    g_retired[g_nretired++] = t;
}
static void destroy_retired(void) {
    for (size_t i = 0; i < g_nretired; i++) { wgpuTextureDestroy(g_retired[i]); wgpuTextureRelease(g_retired[i]); }
    g_nretired = 0;
}

static void dxt_flush(void);   /* DXT decoding (below) */
static WGPUCommandEncoder g_dxt_enc;

#ifndef __EMSCRIPTEN__
/* Submit thread: finishing a command encoder (wgpu validates and translates every recorded command there) and
 * submitting it took ~3.5 ms of each 20 ms frame on the game thread (Galaxy A17). submit() now hands the encoder,
 * with copies of the batch's staged data, to this thread and goes on with the next batch; the thread runs the
 * jobs in order (finish, buffer writes, DXT decodes, submit, retired textures), and a present job hands the frame
 * to the present thread. Anything else that touches the queue in an order-sensitive way first waits for it to
 * be idle (q_drain). At most a few jobs wait, so the game stays within a frame of the GPU work.
 * RC_ASYNC_SUBMIT=1 turns it on (off by default until it is tested on more phones). */
typedef struct QJob {
    WGPUCommandEncoder enc, dxt;
    uint8_t *v, *u, *c;
    size_t vn, un, cn;
    VbPending *pend;
    size_t npend;
    WGPUTexture *ret;
    size_t nret;
    int present;
    struct QJob *next;
} QJob;
static pthread_mutex_t g_qj_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_qj_cond = PTHREAD_COND_INITIALIZER;
static QJob *g_qj_head, *g_qj_tail;
static int g_qj_n, g_qj_on;
static void qjob_run(QJob *j) {
    if (j->enc) {
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(j->enc, NULL);
        pt_wait(0);   /* the present thread's copy must read the back buffer before these draws */
        if (j->vn) wgpuQueueWriteBuffer(g_queue, g_vbuf, 0, j->v, j->vn);
        if (j->un) wgpuQueueWriteBuffer(g_queue, g_ubuf, 0, j->u, j->un);
        if (j->cn) wgpuQueueWriteBuffer(g_queue, g_cbuf, 0, j->c, j->cn);
        for (size_t i = 0; i < j->npend; i++) {
            wgpuQueueWriteBuffer(g_queue, j->pend[i].buf, 0, j->pend[i].data, j->pend[i].bytes);
            wgpuBufferRelease(j->pend[i].buf);
            free(j->pend[i].data);
        }
        if (j->dxt) {
            WGPUCommandBuffer dcb = wgpuCommandEncoderFinish(j->dxt, NULL);
            queue_submit(dcb);
            wgpuCommandBufferRelease(dcb);
            wgpuCommandEncoderRelease(j->dxt);
        }
        queue_submit(cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(j->enc);
    }
    for (size_t i = 0; i < j->nret; i++) { wgpuTextureDestroy(j->ret[i]); wgpuTextureRelease(j->ret[i]); }
    if (j->present) { pt_wait(1); pt_set(1); }   /* the previous frame is presented: hand this one over */
    free(j->v); free(j->u); free(j->c); free(j->pend); free(j->ret);
    free(j);
}
static void *qjob_thread(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_qj_lock);
        while (!g_qj_head) pthread_cond_wait(&g_qj_cond, &g_qj_lock);
        QJob *j = g_qj_head;
        g_qj_head = j->next;
        if (!g_qj_head) g_qj_tail = NULL;
        pthread_mutex_unlock(&g_qj_lock);
        qjob_run(j);
        pthread_mutex_lock(&g_qj_lock);
        g_qj_n--;
        pthread_cond_broadcast(&g_qj_cond);
        pthread_mutex_unlock(&g_qj_lock);
    }
    return NULL;
}
static void q_push(QJob *j) {
    pthread_mutex_lock(&g_qj_lock);
    while (g_qj_n >= 4) pthread_cond_wait(&g_qj_cond, &g_qj_lock);   /* stay close to the GPU work */
    if (g_qj_tail) g_qj_tail->next = j; else g_qj_head = j;
    g_qj_tail = j;
    g_qj_n++;
    pthread_cond_broadcast(&g_qj_cond);
    pthread_mutex_unlock(&g_qj_lock);
}
/* everything handed over so far is submitted (before queue work that must come after it) */
static void q_drain(void) {
    if (!g_qj_on) return;
    pthread_mutex_lock(&g_qj_lock);
    while (g_qj_n) pthread_cond_wait(&g_qj_cond, &g_qj_lock);
    pthread_mutex_unlock(&g_qj_lock);
}
static void *memdup_(const void *p, size_t n) { void *d = n ? malloc(n) : NULL; if (n) memcpy(d, p, n); return d; }
static void q_start(void) {
    const char *e = getenv("RC_ASYNC_SUBMIT");
    if (!e || !atoi(e)) return;   /* opt-in (RC_ASYNC_SUBMIT=1) until tested on more phones */
    pthread_t t;
    if (pthread_create(&t, NULL, qjob_thread, NULL)) return;
    pthread_detach(t);
    g_qj_on = 1;
    fprintf(stderr, "[wgpu] submitting on a separate thread\n");
}
#else
static void q_drain(void) {}
#endif

static void submit(void) {
    end_pass();
#ifndef __EMSCRIPTEN__
    if (g_qj_on) {
        QJob *j = calloc(1, sizeof *j);
        if (g_enc) {
            j->enc = g_enc; g_enc = NULL;
            j->dxt = g_dxt_enc; g_dxt_enc = NULL;
            j->vn = g_vpos; j->v = memdup_(g_vstage, g_vpos);
            j->un = g_upos; j->u = memdup_(g_ustage, g_upos);
            j->cn = g_cpos; j->c = memdup_(g_cstage, g_cpos);
            j->npend = g_npend; j->pend = memdup_(g_pend, g_npend * sizeof *g_pend);
            g_npend = 0;
            g_st.submits++;
        }
        j->nret = g_nretired; j->ret = memdup_(g_retired, g_nretired * sizeof *g_retired);
        g_nretired = 0;
        q_push(j);
        g_vpos = g_upos = g_cpos = 0;
        g_batch++;
        return;
    }
#endif
    if (g_enc) {
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(g_enc, NULL);
        double t_w = host_seconds();
        pt_wait(0);   /* the present thread's copy must read the back buffer before these draws */
        g_st.pt_wait_t += host_seconds() - t_w;
        { extern double g_hx[]; g_hx[5] += host_seconds() - t_w; }
        /* the batch's data, written ahead of the commands that read it (queue order) */
        if (g_vpos) wgpuQueueWriteBuffer(g_queue, g_vbuf, 0, g_vstage, g_vpos);
        if (g_upos) wgpuQueueWriteBuffer(g_queue, g_ubuf, 0, g_ustage, g_upos);
        if (g_cpos) wgpuQueueWriteBuffer(g_queue, g_cbuf, 0, g_cstage, g_cpos);
        for (size_t i = 0; i < g_npend; i++) {
            wgpuQueueWriteBuffer(g_queue, g_pend[i].buf, 0, g_pend[i].data, g_pend[i].bytes);
            wgpuBufferRelease(g_pend[i].buf);
            free(g_pend[i].data);
        }
        g_npend = 0;
        dxt_flush();   /* textures this batch draws with */
        queue_submit(cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(g_enc);
        g_enc = NULL;
        g_st.submits++;
    }
    destroy_retired();   /* nothing recorded can use them any more */
    g_vpos = g_upos = g_cpos = 0;
    g_batch++;
}

static void begin_frame(void) {
    if (g_frame_open) return;
    /* (compared with the window size the surface was configured for: on Android the swap chain is smaller
       than the window, RC_RENDER_HEIGHT) */
    if (g_window_w != g_cfg_window_w || g_window_h != g_cfg_window_h || g_asp_w != g_game_aspect_w || g_asp_h != g_game_aspect_h) {
        end_pass();
        q_drain();
        pt_wait(1);   /* not while the present thread uses the surface */
        configure_surface();
        g_tgt_color = g_tgt_depth = NULL;
        g_tgt_w = (uint32_t)g_surf_w; g_tgt_h = (uint32_t)g_surf_h;
    }
    g_frame_open = 1;
}

/* Opens a render pass on the current target; clear flags: 1 color, 2 depth, 4 stencil. */
static int g_scratch_clear;   /* the next pass runs on a scratch depth buffer: start it cleared (its old contents are another target's) */
static void begin_pass(uint32_t clear, uint32_t color, float z, uint32_t stencil) {
    end_pass();
    if (g_scratch_clear && getenv("RC_SCRATCH_CLEAR")) { clear |= 2 | 4; z = 1.f; stencil = 0; }
    g_scratch_clear = 0;
    if (!g_enc) g_enc = wgpuDeviceCreateCommandEncoder(g_dev, NULL);
    WGPURenderPassColorAttachment ca = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    ca.view = g_tgt_color ? g_tgt_color : g_bb_view;
    ca.loadOp = (clear & 1) ? WGPULoadOp_Clear : WGPULoadOp_Load;
    ca.storeOp = WGPUStoreOp_Store;
    ca.clearValue = (WGPUColor){ ((color >> 16) & 255) / 255.0, ((color >> 8) & 255) / 255.0, (color & 255) / 255.0,
                                 (color >> 24) / 255.0 };
    WGPURenderPassDepthStencilAttachment da = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
    da.view = g_tgt_depth ? g_tgt_depth : g_depth_view;
    da.depthLoadOp = (clear & 2) ? WGPULoadOp_Clear : WGPULoadOp_Load;
    da.depthStoreOp = WGPUStoreOp_Store;
    da.depthClearValue = z;
    da.stencilLoadOp = (clear & 4) ? WGPULoadOp_Clear : WGPULoadOp_Load;
    da.stencilStoreOp = WGPUStoreOp_Store;
    da.stencilClearValue = stencil;
    static int keep = -1;
    if (keep < 0) keep = getenv("RC_SCRATCH_KEEP") != NULL;
    if (g_tgt_depth && g_tgt_depth_scratch == 1 && !keep) {   /* not 2: the game's own depth of another size, whose contents
                                                                    can carry over when a pass on that target is split */
        if (!(clear & 2)) da.depthClearValue = 1.f;
        if (!(clear & 4)) da.stencilClearValue = 0;
        da.depthLoadOp = da.stencilLoadOp = WGPULoadOp_Clear;
        da.depthStoreOp = da.stencilStoreOp = WGPUStoreOp_Discard;
    }
    WGPURenderPassDescriptor rp = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    rp.colorAttachmentCount = 1; rp.colorAttachments = &ca;
    rp.depthStencilAttachment = &da;
    WGPUPassTimestampWrites tw = WGPU_PASS_TIMESTAMP_WRITES_INIT;
    if (g_ts_capture && g_ts_n < TS_MAX) {
        tw.querySet = g_ts_set; tw.beginningOfPassWriteIndex = 2 * g_ts_n; tw.endOfPassWriteIndex = 2 * g_ts_n + 1;
        rp.timestampWrites = &tw;
        g_ts_info[g_ts_n].w = g_tgt_w; g_ts_info[g_ts_n].h = g_tgt_h;
        g_ts_info[g_ts_n].load = !(clear & 1); g_ts_info[g_ts_n].rt = g_tgt_color && g_tgt_color != g_bb_view;
        g_ts_n++;
    }
    g_st.passes++;
    if (!(clear & 1)) g_st.passes_load++;
    g_pass = wgpuCommandEncoderBeginRenderPass(g_enc, &rp);
    g_pass_gen++;   /* a fresh pass has no state bound: exec_draw's cache of what is set starts over */
}

/* ------------------------------------------------------------------ render targets */

static GSurf *gsurf(uint32_t id, uint32_t w, uint32_t h, int depth) {
    GSurf *g = obj_get(id);
    if (g) return g;
    g = calloc(1, sizeof *g);
    g->w = w ? w : 1; g->h = h ? h : 1; g->depth = depth;
    WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
    td.size = (WGPUExtent3D){ g->w, g->h, 1 };
    td.format = depth ? WGPUTextureFormat_Depth24PlusStencil8 : g_surf_fmt;
    td.usage = WGPUTextureUsage_RenderAttachment |
               (depth ? 0 : WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst);
    g->tex = wgpuDeviceCreateTexture(g_dev, &td);
    g->view = wgpuTextureCreateView(g->tex, NULL);
    obj_put(id, g);
    return g;
}

/* depth buffer for targets whose depth surface doesn't match their size (WebGPU needs equal sizes) */
static WGPUTextureView scratch_depth(uint32_t w, uint32_t h) {
    static struct { uint32_t w, h; WGPUTexture t; WGPUTextureView v; } cache[8];
    for (int i = 0; i < 8; i++) if (cache[i].t && cache[i].w == w && cache[i].h == h) return cache[i].v;
    for (int i = 0; i < 8; i++)
        if (!cache[i].t) {
            WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
            td.size = (WGPUExtent3D){ w, h, 1 };
            td.format = WGPUTextureFormat_Depth24PlusStencil8;
            td.usage = WGPUTextureUsage_RenderAttachment;
            cache[i].t = wgpuDeviceCreateTexture(g_dev, &td);
            cache[i].v = wgpuTextureCreateView(cache[i].t, NULL);
            cache[i].w = w; cache[i].h = h;
            return cache[i].v;
        }
    return cache[0].v;
}

static void exec_target(const CmdTarget *c) {
    WGPUTextureView color, depth;
    GSurf *gs = NULL;
    uint32_t w, h;
    if (c->color_id) { gs = gsurf(c->color_id, c->color_w, c->color_h, 0); color = gs->view; w = gs->w; h = gs->h; }
    else { color = g_bb_view; w = (uint32_t)g_surf_w; h = (uint32_t)g_surf_h; }
    int scratch = 0;
    if (!c->color_id && !c->depth_id) depth = g_depth_view;
    else if (c->depth_id) {
        GSurf *gd = gsurf(c->depth_id, c->depth_w, c->depth_h, 1);
        if (gd->w == w && gd->h == h) depth = gd->view; else { depth = scratch_depth(w, h); scratch = 2; }
    } else { depth = scratch_depth(w, h); scratch = 1; }
    g_tgt_depth_scratch = scratch;
    if (color != (g_tgt_color ? g_tgt_color : g_bb_view) || depth != (g_tgt_depth ? g_tgt_depth : g_depth_view)) {
        end_pass();
        g_tgt_color = color; g_tgt_depth = depth;
        if (c->color_id && !c->depth_id) g_scratch_clear = 1;
    }
    g_tgt_w = w; g_tgt_h = h; g_tgt_lw = c->lw; g_tgt_lh = c->lh; g_tgt_gs = gs;
}

static void use_backbuffer(void) {
    if (g_tgt_color && g_tgt_color != g_bb_view) end_pass();
    g_tgt_color = g_tgt_depth = NULL;
    g_tgt_gs = NULL;
    g_tgt_w = (uint32_t)g_surf_w; g_tgt_h = (uint32_t)g_surf_h;
}

/* one BGRA8 pixel -> a Direct3D format; returns bytes written (0: unsupported) */
static uint32_t pack_pixel(uint32_t fmt, const uint8_t *bgra, uint8_t *o) {
    uint32_t b = bgra[0], g = bgra[1], r = bgra[2], a = bgra[3];
    switch (fmt) {
    case FMT_A8R8G8B8: case FMT_X8R8G8B8: memcpy(o, bgra, 4); if (fmt == FMT_X8R8G8B8) o[3] = 255; return 4;
    case FMT_R5G6B5: { uint16_t v = (uint16_t)((r >> 3) << 11 | (g >> 2) << 5 | b >> 3); memcpy(o, &v, 2); return 2; }
    case FMT_X1R5G5B5: case FMT_A1R5G5B5: {
        uint16_t v = (uint16_t)((a >= 128 ? 0x8000 : 0) | (r >> 3) << 10 | (g >> 3) << 5 | b >> 3); memcpy(o, &v, 2); return 2; }
    case FMT_A4R4G4B4: { uint16_t v = (uint16_t)((a >> 4) << 12 | (r >> 4) << 8 | (g >> 4) << 4 | b >> 4); memcpy(o, &v, 2); return 2; }
    default: return 0;
    }
}

/* mapped BGRA pixels -> guest memory in the surface's format (nearest scaling: the back buffer is scaled) */
static void store_pixels(const CmdReadback *c, const uint8_t *px, uint32_t gw, uint32_t gh, uint32_t stride) {
    uint8_t tmp[4];
    if (!pack_pixel(c->fmt, (const uint8_t[4]){ 0, 0, 0, 0 }, tmp)) {
        static int warned;
        if (!warned++) fprintf(stderr, "[wgpu] render target readback: format %x not supported\n", c->fmt);
        return;
    }
    for (uint32_t y = 0; y < c->h_; y++) {
        const uint8_t *row = px + (size_t)(y * gh / c->h_) * stride;
        uint8_t *o = HP(c->guest_mem + y * c->pitch);
        for (uint32_t x = 0; x < c->w; x++) o += pack_pixel(c->fmt, row + 4 * (x * gw / c->w), o);
    }
}

/* Starts copying a readback source into a mappable buffer; returns the buffer (or NULL: nothing to read). */
static WGPUBuffer start_readback(const CmdReadback *c, uint32_t *gw, uint32_t *gh, uint32_t *stride) {
    GSurf *gs = c->id ? obj_get(c->id) : NULL;
    if (c->id && (!gs || gs->depth || !gs->gpu_dirty)) return NULL;   /* CPU copy is current */
    WGPUTexture tex = c->id ? gs->tex : g_bb;
    *gw = c->id ? gs->w : (uint32_t)g_surf_w;
    *gh = c->id ? gs->h : (uint32_t)g_surf_h;
    *stride = (*gw * 4 + 255) & ~255u;
    submit();
    if (g_frame_open) begin_pass(0, 0, 1.f, 0);   /* keep drawing into the same target afterwards */
    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.size = (uint64_t)*stride * *gh;
    bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(g_dev, &bd);
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_dev, NULL);
    WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    src.texture = tex;
    WGPUTexelCopyBufferInfo dst = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    dst.buffer = buf; dst.layout.bytesPerRow = *stride; dst.layout.rowsPerImage = *gh;
    WGPUExtent3D ex = { *gw, *gh, 1 };
    wgpuCommandEncoderCopyTextureToBuffer(enc, &src, &dst, &ex);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    queue_submit(cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    if (gs) gs->gpu_dirty = 0;
    return buf;
}

static void finish_readback(const CmdReadback *c) {
    __atomic_store_n((volatile int *)(uintptr_t)c->done_ptr, 1, __ATOMIC_RELEASE);
}

/* ------------------------------------------------------------------ commands */

/* Converted vertices of unchanging Direct3D vertex buffers (render_rec.c's cache), by id. Releasing one
   only drops a reference: draws already recorded keep it alive until they have run. */
int g_gpu_vb_cache = 1;   /* tells the recorder these commands are handled */
typedef struct { WGPUBuffer buf; uint32_t bytes; } GVb;
size_t g_gpu_vb_bytes;

static void exec_upload_vb(const CmdUploadVb *c) {
    GVb *g = calloc(1, sizeof *g);
    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.size = (c->bytes + 3) & ~3u;
    bd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
    g->buf = wgpuDeviceCreateBuffer(g_dev, &bd);
    g->bytes = c->bytes;
    /* written at the next submit, ahead of the draws that read it (like the staging buffers: a queue write
       now could wait for the present thread) */
    if (g_npend == g_cappend) {
        g_cappend = g_cappend ? g_cappend * 2 : 64;
        g_pend = realloc(g_pend, g_cappend * sizeof *g_pend);
    }
    VbPending *pe = &g_pend[g_npend++];
    pe->buf = g->buf;
    wgpuBufferAddRef(g->buf);   /* in case it is destroyed before the submit */
    pe->bytes = (uint32_t)bd.size;
    pe->data = malloc(pe->bytes);
    memcpy(pe->data, c + 1, c->bytes);
    memset(pe->data + c->bytes, 0, pe->bytes - c->bytes);
    g_st.upload_bytes += c->bytes;
    g_gpu_vb_bytes += c->bytes;
    obj_put(c->id, g);
}

static void exec_destroy_vb(uint32_t id) {
    GVb *g = obj_get(id);
    if (!g) return;
    wgpuBufferRelease(g->buf);
    g_gpu_vb_bytes -= g->bytes;
    free(g);
    obj_del(id);
}

/* ------------------------------------------------------------------ DXT decoding on the GPU
 * Mali has no BC texture sampling, so DXT textures used to be decoded on the CPU at load (~20 ms per batch,
 * plus 4x the upload). Their blocks are uploaded as stored instead and a compute shader writes the RGBA8
 * texture, one invocation per 4x4 block (src: w, h, kind 1/2/3 = DXT1/3/5, pad, then the blocks). The work goes
 * in its own command buffer, submitted just ahead of the batch that draws with it (submit). RC_CPU_DXT=1: CPU. */
static const char *WGSL_DXT =
"@group(0) @binding(0) var<storage, read> src: array<u32>;\n"
"@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;\n"
"fn c565(c: u32) -> vec3<f32> { return vec3<f32>(f32((c >> 11u) & 31u) / 31.0, f32((c >> 5u) & 63u) / 63.0, f32(c & 31u) / 31.0); }\n"
"@compute @workgroup_size(8, 8)\n"
"fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
"  let w = src[0]; let h = src[1]; let fmt = src[2];\n"
"  let bw = (w + 3u) / 4u; let bh = (h + 3u) / 4u;\n"
"  if (id.x >= bw || id.y >= bh) { return; }\n"
"  let base = 4u + (id.y * bw + id.x) * select(4u, 2u, fmt == 1u);\n"
"  var a0 = 0u; var a1 = 0u; var cw = base;\n"
"  if (fmt != 1u) { a0 = src[base]; a1 = src[base + 1u]; cw = base + 2u; }\n"
"  let cols = src[cw]; let idx = src[cw + 1u];\n"
"  let c0 = cols & 0xffffu; let c1 = cols >> 16u;\n"
"  let p0 = c565(c0); let p1 = c565(c1);\n"
"  var pal: array<vec4<f32>, 4>;\n"
"  pal[0] = vec4<f32>(p0, 1.0); pal[1] = vec4<f32>(p1, 1.0);\n"
"  if (c0 > c1 || fmt != 1u) { pal[2] = vec4<f32>((2.0 * p0 + p1) / 3.0, 1.0); pal[3] = vec4<f32>((p0 + 2.0 * p1) / 3.0, 1.0); }\n"
"  else { pal[2] = vec4<f32>((p0 + p1) * 0.5, 1.0); pal[3] = vec4<f32>(0.0); }\n"
"  let f0 = f32(a0 & 255u); let f1 = f32((a0 >> 8u) & 255u);\n"
"  for (var i = 0u; i < 16u; i++) {\n"
"    let x = id.x * 4u + (i & 3u); let y = id.y * 4u + (i >> 2u);\n"
"    if (x >= w || y >= h) { continue; }\n"
"    var c = pal[(idx >> (2u * i)) & 3u];\n"
"    if (fmt == 2u) {\n"
"      c.a = f32((select(a0, a1, i >= 8u) >> (4u * (i & 7u))) & 15u) / 15.0;\n"
"    } else if (fmt == 3u) {\n"
"      let bit = 16u + 3u * i;\n"
"      var ai = 0u;\n"
"      if (bit + 3u <= 32u) { ai = (a0 >> bit) & 7u; } else if (bit >= 32u) { ai = (a1 >> (bit - 32u)) & 7u; }\n"
"      else { ai = ((a0 >> bit) | (a1 << (32u - bit))) & 7u; }\n"
"      var av = 0.0;\n"
"      if (ai == 0u) { av = f0; } else if (ai == 1u) { av = f1; }\n"
"      else if (f0 > f1) { av = (f32(8u - ai) * f0 + f32(ai - 1u) * f1) / 7.0; }\n"
"      else if (ai == 6u) { av = 0.0; } else if (ai == 7u) { av = 255.0; }\n"
"      else { av = (f32(6u - ai) * f0 + f32(ai - 1u) * f1) / 5.0; }\n"
"      c.a = av / 255.0;\n"
"    }\n"
"    textureStore(dst, vec2<i32>(i32(x), i32(y)), c);\n"
"  }\n"
"}\n";
static WGPUComputePipeline g_dxt_pipe;
static WGPUBindGroupLayout g_dxt_bgl;
static void dxt_init(void) {
    WGPUShaderSourceWGSL src = WGPU_SHADER_SOURCE_WGSL_INIT;
    src.code = sv(WGSL_DXT);
    WGPUShaderModuleDescriptor smd = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    smd.nextInChain = &src.chain;
    WGPUShaderModule m = wgpuDeviceCreateShaderModule(g_dev, &smd);
    WGPUBindGroupLayoutEntry e[2] = { WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT };
    e[0].binding = 0; e[0].visibility = WGPUShaderStage_Compute; e[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    e[1].binding = 1; e[1].visibility = WGPUShaderStage_Compute; e[1].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
    e[1].storageTexture.format = WGPUTextureFormat_RGBA8Unorm; e[1].storageTexture.viewDimension = WGPUTextureViewDimension_2D;
    WGPUBindGroupLayoutDescriptor bd = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bd.entryCount = 2; bd.entries = e;
    g_dxt_bgl = wgpuDeviceCreateBindGroupLayout(g_dev, &bd);
    WGPUPipelineLayoutDescriptor pld = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &g_dxt_bgl;
    WGPUComputePipelineDescriptor cd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    cd.layout = wgpuDeviceCreatePipelineLayout(g_dev, &pld);
    cd.compute.module = m;
    cd.compute.entryPoint = sv("main");
    g_dxt_pipe = wgpuDeviceCreateComputePipeline(g_dev, &cd);
    fprintf(stderr, "[wgpu] DXT textures decoded on the GPU%s\n", g_dxt_pipe ? "" : ": pipeline failed");
}
/* one mip level: its blocks (as stored) -> level l of tex */
static void dxt_decode(WGPUTexture tex, uint32_t l, uint32_t w, uint32_t h, uint32_t fmt, const uint8_t *blocks, size_t n) {
    if (!g_dxt_pipe) dxt_init();
    if (!g_dxt_pipe) return;
    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.size = (16 + n + 3) & ~(size_t)3;
    bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    WGPUBuffer buf = wgpuDeviceCreateBuffer(g_dev, &bd);
    uint32_t head[4] = { w, h, fmt == TEXFMT_DXT1_DEC ? 1u : fmt == TEXFMT_DXT3_DEC ? 2u : 3u, 0 };
    wgpuQueueWriteBuffer(g_queue, buf, 0, head, 16);
    wgpuQueueWriteBuffer(g_queue, buf, 16, blocks, n & ~(size_t)3);
    WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    vd.baseMipLevel = l; vd.mipLevelCount = 1; vd.dimension = WGPUTextureViewDimension_2D;
    WGPUTextureView view = wgpuTextureCreateView(tex, &vd);
    WGPUBindGroupEntry be[2] = { WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT };
    be[0].binding = 0; be[0].buffer = buf; be[0].size = bd.size;
    be[1].binding = 1; be[1].textureView = view;
    WGPUBindGroupDescriptor bgd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bgd.layout = g_dxt_bgl; bgd.entryCount = 2; bgd.entries = be;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_dev, &bgd);
    if (!g_dxt_enc) g_dxt_enc = wgpuDeviceCreateCommandEncoder(g_dev, NULL);
    WGPUComputePassEncoder cp = wgpuCommandEncoderBeginComputePass(g_dxt_enc, NULL);
    wgpuComputePassEncoderSetPipeline(cp, g_dxt_pipe);
    wgpuComputePassEncoderSetBindGroup(cp, 0, bg, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(cp, ((w + 3) / 4 + 7) / 8, ((h + 3) / 4 + 7) / 8, 1);
    wgpuComputePassEncoderEnd(cp);
    wgpuComputePassEncoderRelease(cp);
    wgpuBindGroupRelease(bg); wgpuTextureViewRelease(view); wgpuBufferRelease(buf);   /* kept alive by the commands */
}
/* the decodes go to the queue ahead of the batch that draws with them */
static void dxt_flush(void) {
    if (!g_dxt_enc) return;
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(g_dxt_enc, NULL);
    queue_submit(cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(g_dxt_enc);
    g_dxt_enc = NULL;
}

static void exec_upload_tex(const CmdUploadTex *c) {
    GTex *g = obj_get(c->id);
    static const WGPUTextureFormat formats[4] = { WGPUTextureFormat_RGBA8Unorm, WGPUTextureFormat_BC1RGBAUnorm,
                                                  WGPUTextureFormat_BC2RGBAUnorm, WGPUTextureFormat_BC3RGBAUnorm };
    uint32_t bs = TEXFMT_BLOCK_BYTES(c->fmt);
    int dec = c->fmt >= TEXFMT_DXT1_DEC;   /* blocks decoded into an RGBA8 texture (dxt_decode) */
    int fresh = 0;   /* made here: nothing submitted yet can have drawn with it */
    if (g && (g->w != c->w[0] || g->h != c->h_[0] || g->levels != c->levels || g->fmt != c->fmt)) {
        release_view(g->view); retire_texture(g->tex); g_gpu_tex_bytes -= g->bytes; free(g); obj_del(c->id); g = NULL;
    }
    if (!g) {
        g = calloc(1, sizeof *g);
        fresh = 1;
        WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
        td.size = (WGPUExtent3D){ c->w[0], c->h_[0], 1 };
        td.format = dec ? WGPUTextureFormat_RGBA8Unorm : formats[c->fmt & 3];
        td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst | (dec ? WGPUTextureUsage_StorageBinding : 0);
        td.mipLevelCount = c->levels;
        g->tex = wgpuDeviceCreateTexture(g_dev, &td);
        g->view = wgpuTextureCreateView(g->tex, NULL);
        g->w = c->w[0]; g->h = c->h_[0]; g->levels = c->levels; g->fmt = c->fmt;
        for (uint32_t l = 0; l < c->levels; l++)
            g->bytes += c->fmt && !dec ? (size_t)((c->w[l] + 3) / 4) * ((c->h_[l] + 3) / 4) * bs : (size_t)c->w[l] * c->h_[l] * 4;
        g_gpu_tex_bytes += g->bytes;
        obj_put(c->id, g);
    } else if (g->used_in_batch == g_batch) {
        /* already drawn with in the current batch: flush that work first, so the earlier draws see
           the old contents (Direct3D semantics) */
        submit();
        if (g_frame_open) begin_pass(0, 0, 1.f, 0);
    }
    const uint8_t *px = (const uint8_t *)(c + 1);
    for (uint32_t l = 0; l < c->levels; l++) {
        WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        dst.texture = g->tex; dst.mipLevel = l;
        WGPUTexelCopyBufferLayout lay = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
        WGPUExtent3D ex = { c->w[l], c->h_[l], 1 };
        size_t n;
        if (c->fmt) {   /* whole blocks: small mips are copied at their rounded-up (physical) size */
            uint32_t bw = (c->w[l] + 3) / 4, bh = (c->h_[l] + 3) / 4;
            lay.bytesPerRow = bw * bs; lay.rowsPerImage = bh;
            ex = (WGPUExtent3D){ bw * 4, bh * 4, 1 };
            n = (size_t)bw * bh * bs;
        } else {
            lay.bytesPerRow = c->w[l] * 4; lay.rowsPerImage = c->h_[l];
            n = (size_t)c->w[l] * c->h_[l] * 4;
        }
        if (dec) dxt_decode(g->tex, l, c->w[l], c->h_[l], c->fmt, px, n);
        else { if (!l && !fresh) q_drain(); wgpuQueueWriteTexture(g_queue, &dst, px, n, &lay, &ex); }
        g_st.upload_bytes += n;
        px += n;
    }
}

static void exec_destroy(uint32_t id, int surface) {
    void *p = obj_get(id);
    if (!p) return;
    if (surface) {
        GSurf *g = p;
        if (g->view == g_tgt_color || g->view == g_tgt_depth) use_backbuffer();
        release_view(g->view); retire_texture(g->tex);
    } else {
        GTex *g = p;
        release_view(g->view); retire_texture(g->tex);
        g_gpu_tex_bytes -= g->bytes;
    }
    free(p);
    obj_del(id);
}

static void exec_upload_surf(const CmdUploadSurf *c) {
    q_drain();   /* earlier batches drawn with its old contents go first */
    GSurf *gs = gsurf(c->id, c->w, c->h_, 0);
    uint8_t *px = malloc((size_t)gs->w * gs->h * 4);
    memcpy(px, c + 1, (size_t)gs->w * gs->h * 4);
    if (g_surf_fmt == WGPUTextureFormat_BGRA8Unorm)
        for (size_t k = 0; k < (size_t)gs->w * gs->h; k++) { uint8_t t = px[4 * k]; px[4 * k] = px[4 * k + 2]; px[4 * k + 2] = t; }
    submit();   /* keep ordering with draws already recorded */
    if (g_frame_open) begin_pass(0, 0, 1.f, 0);
    WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    dst.texture = gs->tex;
    WGPUTexelCopyBufferLayout lay = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
    lay.bytesPerRow = gs->w * 4; lay.rowsPerImage = gs->h;
    WGPUExtent3D ex = { gs->w, gs->h, 1 };
    wgpuQueueWriteTexture(g_queue, &dst, px, (size_t)gs->w * gs->h * 4, &lay, &ex);
    free(px);
    gs->gpu_dirty = 0;
}

static void exec_clear(const CmdClear *c) {
    begin_frame();
    begin_pass(c->flags, c->color, c->z, c->stencil);
    if (g_tgt_gs) g_tgt_gs->gpu_dirty = 1;
}

static void exec_draw(const CmdDraw *c) {
    {   /* RC_LOG_QUADS=1: log full-screen-style quads (4 or 6 vertices) */
        static int q = -1; static int n;
        if (q < 0) q = getenv("RC_LOG_QUADS") != NULL;
        if (q && c->nverts <= 6 && n < 400 && !c->key.vs && !c->vb_id) {
            const CVert *v = (const CVert *)(c + 1);
            n++;
            fprintf(stderr, "[quad] flags0 %u vp %g,%g,%g,%g kinds %u,%u ids %u,%u ps %u verts (%g,%g,%g,%g) (%g,%g) (%g,%g) tc0 (%g,%g) (%g,%g)\n", c->u.flags[0], c->u.vp[0], c->u.vp[1], c->u.vp[2], c->u.vp[3],
                    c->slot_kind[0], c->slot_kind[1], c->slot_id[0], c->slot_id[1], c->key.ps, v[0].x, v[0].y, v[0].z, v[0].w, v[1].x, v[1].y, v[2].x, v[2].y, v[0].tc[0][0], v[0].tc[0][1], v[2].tc[0][0], v[2].tc[0][1]);
        }
    }
    begin_frame();
    if (!g_pass) begin_pass(0, 0, 1.f, 0);
    /* vertex shader on the GPU: its constants, then its input registers per vertex (render_cmd.h) */
    const float *vs_consts = c->key.vs ? (const float *)(c + 1) : NULL;
    const void *verts = c->key.vs ? (const void *)(vs_consts + 96 * 4) : (const void *)(c + 1);
    /* vb_id: verts are uint32 indices into a cached vertex buffer */
    GVb *gvb = c->vb_id ? obj_get(c->vb_id) : NULL;
    if (c->vb_id && !gvb) return;
    uint32_t n = c->nverts, bytes = n * (gvb ? 4u : c->key.vs ? c->vs_stride : (uint32_t)sizeof(CVert));
    uint32_t ibytes = gvb ? 0 : c->nidx * 4;   /* indices into this draw's own vertices follow them */
    {   /* debugging: RC_SUBMIT_EVERY=N submits the frame's work every N draws (as the per-draw capture does) */
        static int every = -1, n;
        if (every < 0) every = getenv("RC_SUBMIT_EVERY") ? atoi(getenv("RC_SUBMIT_EVERY")) : 0;
        if (every && ++n >= every) { n = 0; submit(); begin_pass(0, 0, 1.f, 0); }
    }
    if (g_vpos + bytes + ibytes > VBUF_SIZE || g_upos + USTRIDE > UBUF_SIZE || g_cpos + CSTRIDE > CBUF_SIZE) {
        submit(); begin_pass(0, 0, 1.f, 0);
    }

    PsProgram *ps = c->key.ps ? ps_program(c->key.ps) : NULL;
    WGPUTextureView views[4];
    for (int s = 0; s < 4; s++) {
        views[s] = ps ? g_black.view : g_white.view;
        if (c->slot_kind[s] == TS_TEX) {
            GTex *g = obj_get(c->slot_id[s]);
            if (g) { g->used_in_batch = g_batch; views[s] = g->view; }
        } else if (c->slot_kind[s] == TS_BACKBUF) {
            /* the back buffer is sampled (through a copy of it) only while drawing into something else */
            if (g_tgt_color && g_tgt_color != g_bb_view) {
                static WGPUTexture copy; static WGPUTextureView copy_view; static uint32_t cw, ch;
                if (!copy || cw != (uint32_t)g_surf_w || ch != (uint32_t)g_surf_h) {
                    if (copy) { release_view(copy_view); wgpuTextureRelease(copy); }
                    WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
                    cw = (uint32_t)g_surf_w; ch = (uint32_t)g_surf_h;
                    td.size = (WGPUExtent3D){ cw, ch, 1 }; td.format = g_surf_fmt;
                    td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
                    copy = wgpuDeviceCreateTexture(g_dev, &td);
                    copy_view = wgpuTextureCreateView(copy, NULL);
                }
                end_pass();
                WGPUTexelCopyTextureInfo cs = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT, cd = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
                cs.texture = g_bb; cd.texture = copy;
                WGPUExtent3D cex = { cw, ch, 1 };
                wgpuCommandEncoderCopyTextureToTexture(g_enc, &cs, &cd, &cex);
                begin_pass(0, 0, 1.f, 0);
                views[s] = copy_view;
            } else views[s] = g_black.view;
        } else if (c->slot_kind[s] == TS_RTSURF) {
            GSurf *gs = obj_get(c->slot_id[s]);
            /* a render-target texture is never sampled while it is being drawn into */
            views[s] = gs && !gs->depth && gs->view != g_tgt_color ? gs->view : g_black.view;
        }
    }
    memcpy(g_vstage + g_vpos, verts, bytes + ibytes);
    memcpy(g_ustage + g_upos, &c->u, sizeof c->u);

    WGPUBindGroupEntry te[8];
    for (int i = 0; i < 4; i++) {
        te[2 * i] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
        te[2 * i].binding = (uint32_t)(2 * i); te[2 * i].textureView = views[i];
        te[2 * i + 1] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
        te[2 * i + 1].binding = (uint32_t)(2 * i + 1); te[2 * i + 1].sampler = sampler_for(c->samp_key[i]);
    }
    WGPUBindGroup bg = NULL;
    uint32_t hsh = bg_hash(views, c->samp_key);
    BgEntry *victim = NULL;
    for (int p = 0; p < BG_PROBES && !bg; p++) {
        BgEntry *e = &g_bgc[(hsh + (uint32_t)p) % BG_CACHE];
        if (e->bg && !memcmp(e->v, views, sizeof e->v) && !memcmp(e->s, c->samp_key, sizeof e->s)) { bg = e->bg; e->stamp = ++g_bg_stamp; }
        else if (!victim || !e->bg || (victim->bg && e->stamp < victim->stamp)) victim = e;
    }
    if (!bg) {
        WGPUBindGroupDescriptor bgd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bgd.layout = g_bgl_t; bgd.entryCount = 8; bgd.entries = te;
        bg = wgpuDeviceCreateBindGroup(g_dev, &bgd);
        g_gpu_bind_groups_made++;
        if (victim->bg) { wgpuBindGroupRelease(victim->bg); g_bg_freed++; }
        memcpy(victim->v, views, sizeof victim->v);
        memcpy(victim->s, c->samp_key, sizeof victim->s);
        victim->bg = bg;
        victim->stamp = ++g_bg_stamp;
    }

    PKey k = c->key;
    if (!ps) k.ps = ff_tag(&c->u);   /* fixed function (or an untranslatable pixel shader): its own fragment shader */
    WGPURenderPipeline p = pipeline_for(&k);
    if (!p) return;   /* untranslatable vertex shader, or the cache is full */

    float vx, vy, vw, vh;
    if (c->full_viewport) { vx = vy = 0; vw = (float)g_tgt_w; vh = (float)g_tgt_h; }
    else {
        float sx = (float)g_tgt_w / (float)g_tgt_lw, sy = (float)g_tgt_h / (float)g_tgt_lh;
        vx = c->vp[0] * sx; vy = c->vp[1] * sy; vw = c->vp[2] * sx; vh = c->vp[3] * sy;
        if (vx + vw > g_tgt_w) vw = g_tgt_w - vx;
        if (vy + vh > g_tgt_h) vh = g_tgt_h - vy;
    }
    if (vw <= 0 || vh <= 0) return;
    /* RC_SKIP_DRAWS=1 (profiling): everything but the draw itself, to tell GPU cost from the rest;
       =2 skips only draws whose vertex shader runs on the GPU */
    static int skip = -1;
    if (skip < 0) skip = getenv("RC_SKIP_DRAWS") ? atoi(getenv("RC_SKIP_DRAWS")) : 0;
    if (skip == 1 || (skip == 2 && c->key.vs)) return;
    /* State already set in this pass isn't set again: each encoder call costs wgpu validation and tracking on the
       CPU, and consecutive draws mostly share viewport, pipeline and textures (RC_NO_STATE_CACHE=1 sets everything). */
    static struct { uint32_t gen; float vp[6]; WGPURenderPipeline pipe; uint32_t sref; int sref_ok; WGPUBindGroup bg1;
                    uint32_t bg_freed; } S;
    static int nocache = -1;
    if (nocache < 0) nocache = getenv("RC_NO_STATE_CACHE") != NULL;
    if (S.gen != g_pass_gen || nocache) { memset(&S, 0, sizeof S); S.gen = g_pass_gen; S.vp[2] = -1; }
    if (S.vp[0] != vx || S.vp[1] != vy || S.vp[2] != vw || S.vp[3] != vh || S.vp[4] != c->minz || S.vp[5] != c->maxz) {
        wgpuRenderPassEncoderSetViewport(g_pass, vx, vy, vw, vh, c->minz, c->maxz);
        S.vp[0] = vx; S.vp[1] = vy; S.vp[2] = vw; S.vp[3] = vh; S.vp[4] = c->minz; S.vp[5] = c->maxz;
    }
    if (S.pipe != p) { wgpuRenderPassEncoderSetPipeline(g_pass, p); S.pipe = p; }
    if ((k.sten & 1) && (!S.sref_ok || S.sref != c->stencil_ref)) {
        wgpuRenderPassEncoderSetStencilReference(g_pass, c->stencil_ref);
        S.sref = c->stencil_ref; S.sref_ok = 1;
    }
    uint32_t off = g_upos;
    wgpuRenderPassEncoderSetBindGroup(g_pass, 0, g_bg_u, 1, &off);   /* this draw's uniforms: a new offset each time */
    if (S.bg1 != bg || S.bg_freed != g_bg_freed) {   /* owned by the cache */
        wgpuRenderPassEncoderSetBindGroup(g_pass, 1, bg, 0, NULL); S.bg1 = bg; S.bg_freed = g_bg_freed;
    }
    if (vs_consts) {
        memcpy(g_cstage + g_cpos, vs_consts, CSTRIDE);
        uint32_t coff = g_cpos;
        wgpuRenderPassEncoderSetBindGroup(g_pass, 2, g_bg_c, 1, &coff);
        g_cpos += CSTRIDE;
    }
    if (gvb) {
        wgpuRenderPassEncoderSetVertexBuffer(g_pass, 0, gvb->buf, 0, gvb->bytes);
        wgpuRenderPassEncoderSetIndexBuffer(g_pass, g_vbuf, WGPUIndexFormat_Uint32, g_vpos, bytes);
        wgpuRenderPassEncoderDrawIndexed(g_pass, n, 1, 0, 0, 0);
    } else if (ibytes) {
        wgpuRenderPassEncoderSetVertexBuffer(g_pass, 0, g_vbuf, g_vpos, bytes);
        wgpuRenderPassEncoderSetIndexBuffer(g_pass, g_vbuf, WGPUIndexFormat_Uint32, g_vpos + bytes, ibytes);
        wgpuRenderPassEncoderDrawIndexed(g_pass, c->nidx, 1, 0, 0, 0);
    } else {
        wgpuRenderPassEncoderSetVertexBuffer(g_pass, 0, g_vbuf, g_vpos, bytes);
        wgpuRenderPassEncoderDraw(g_pass, n, 1, 0, 0);
    }
    if (g_tgt_gs) g_tgt_gs->gpu_dirty = 1;
    g_vpos += (bytes + ibytes + 255) & ~255u;
    g_upos += USTRIDE;
#ifndef __EMSCRIPTEN__
    if (c->capture_index >= 0) {
        /* frame capture: the back buffer after every draw, as /tmp/rc_cap/NNNN.png */
        mkdir("/tmp/rc_cap", 0755);
        char path[64];
        snprintf(path, sizeof path, "/tmp/rc_cap/%04d.png", c->capture_index);
        submit();
        capture_png(path);
        begin_pass(0, 0, 1.f, 0);
    }
#endif
}

/* ------------------------------------------------------------------ present */

#ifndef __EMSCRIPTEN__
static void pace_60hz(void) {
    static double next;
    double now = host_seconds();
    if (next == 0 || now - next > 0.1) next = now;
    next += 1.0 / 60.0;
    if (next > now) usleep((useconds_t)((next - now) * 1e6));
}
#endif

/* Copies the back buffer into the window and presents it: on the present thread, or inline. */
static void present_to_window(void) {
#ifdef __ANDROID__
    /* blocks while the app is in the background; a new Android surface needs a new wgpu surface */
    unsigned gen = android_present_begin();
    if (gen != g_android_gen) {
        if (g_pt_on) { g_pt_reconfig = 1; android_present_end(); return; }   /* the game thread swaps it */
        g_android_gen = gen;
        wgpuSurfaceRelease(g_surface);
        android_create_surface();
        configure_surface();
    }
#endif
    WGPUSurfaceTexture cur = WGPU_SURFACE_TEXTURE_INIT;
    double t_acq = host_seconds();
    wgpuSurfaceGetCurrentTexture(g_surface, &cur);
    g_st.acquire_t += host_seconds() - t_acq;
    if (cur.status == WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal ||
        cur.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_dev, NULL);
        if (g_win_w != g_surf_w || g_win_h != g_surf_h) {   /* black bars around the 4:3 picture */
            WGPUTextureView v = wgpuTextureCreateView(cur.texture, NULL);
            WGPURenderPassColorAttachment ca = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
            ca.view = v; ca.loadOp = WGPULoadOp_Clear; ca.storeOp = WGPUStoreOp_Store;
            ca.clearValue = (WGPUColor){ 0, 0, 0, 1 };
            WGPURenderPassDescriptor rp = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
            rp.colorAttachmentCount = 1; rp.colorAttachments = &ca;
            WGPURenderPassEncoder pe = wgpuCommandEncoderBeginRenderPass(enc, &rp);
            wgpuRenderPassEncoderEnd(pe);
            wgpuRenderPassEncoderRelease(pe);
            wgpuTextureViewRelease(v);
        }
        WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT, dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        src.texture = g_bb;
        dst.texture = cur.texture;
        dst.origin = (WGPUOrigin3D){ (uint32_t)(g_win_w - g_surf_w) / 2, (uint32_t)(g_win_h - g_surf_h) / 2, 0 };
        WGPUExtent3D ex = { (uint32_t)g_surf_w, (uint32_t)g_surf_h, 1 };
        wgpuCommandEncoderCopyTextureToTexture(enc, &src, &dst, &ex);
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
        queue_submit(cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(enc);
#ifndef __EMSCRIPTEN__
        if (g_pt_on) pt_set(2);   /* the back buffer is copied: the game thread may draw into it again */
        double t_swap = host_seconds();
        pthread_mutex_lock(&g_qlock);
        wgpuSurfacePresent(g_surface);   /* FIFO: waits for vsync */
        pthread_mutex_unlock(&g_qlock);
        g_st.swap_t += host_seconds() - t_swap;
        wgpuTextureRelease(cur.texture);
        /* RC_PACE60: fixed 60 Hz even on fast displays, so scripted input (RC_INPUT) is repeatable. Always
           on Android, where 90/120 Hz panels ignore the frame-rate hint: keep the game at 60 */
        static int pace = -1;
#ifdef __ANDROID__
        if (pace < 0) pace = 1;
#endif
        if (pace < 0) pace = getenv("RC_PACE60") != NULL;
        double t_pace = host_seconds();
        if (pace) pace_60hz();
        g_st.pace_t += host_seconds() - t_pace;
#else
        wgpuTextureRelease(cur.texture);   /* the browser presents the canvas when this task ends */
#endif
    } else {
        if (cur.texture) wgpuTextureRelease(cur.texture);
#ifndef __EMSCRIPTEN__
        if (cur.status != (WGPUSurfaceGetCurrentTextureStatus)0x00030001) {   /* not just occluded */
            if (g_pt_on) g_pt_reconfig = 2; else configure_surface();
        }
        pace_60hz();
#endif
    }
#ifdef __ANDROID__
    android_present_end();
#endif
}

#ifndef __EMSCRIPTEN__
static void *present_thread(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_pt_lock);
        while (g_pt_state != 1) pthread_cond_wait(&g_pt_cond, &g_pt_lock);
        pthread_mutex_unlock(&g_pt_lock);
        present_to_window();
        pt_set(0);
    }
    return NULL;
}

/* game thread, with the present thread idle: what it asked for (surfaces belong to the game thread,
   since configuring one recreates the back buffer it draws into) */
static void pt_reconfigure(void) {
    int r = g_pt_reconfig;
    g_pt_reconfig = 0;
#ifdef __ANDROID__
    if (r == 1) {
        g_android_gen = android_present_begin();
        wgpuSurfaceRelease(g_surface);
        android_create_surface();
        configure_surface();
        android_present_end();
        return;
    }
#endif
    if (r) configure_surface();
}
#endif

#ifndef __EMSCRIPTEN__
static void ts_frame_done(void) {
    if (!g_ts_on) return;
    static double last;
    if (g_ts_capture) {   /* this frame was measured: read its timestamps back (stalls this frame) */
        q_drain();
        g_ts_capture = 0;
        if (!g_ts_n) return;
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_dev, NULL);
        wgpuCommandEncoderResolveQuerySet(enc, g_ts_set, 0, 2 * g_ts_n, g_ts_resolve, 0);
        wgpuCommandEncoderCopyBufferToBuffer(enc, g_ts_resolve, 0, g_ts_read, 0, 16 * g_ts_n);
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
        pt_wait(1);   /* the wait below polls the device: keep the present thread out of it */
        queue_submit(cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(enc);
        WGPUBufferMapCallbackInfo mci = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        mci.mode = WGPUCallbackMode_AllowProcessEvents;
        mci.callback = on_mapped;
        g_map_done = 0;
        wgpuBufferMapAsync(g_ts_read, WGPUMapMode_Read, 0, 16 * g_ts_n, mci);
        while (!g_map_done) wgpuInstanceProcessEvents(g_inst);
        const uint64_t *t = wgpuBufferGetConstMappedRange(g_ts_read, 0, 16 * g_ts_n);
        double period = wgpuQueueGetTimestampPeriod(g_queue) * 1e-6;   /* ticks -> ms */
        double total = 0;
        char line[4096]; int o = 0;
        for (uint32_t i = 0; i < g_ts_n; i++) {
            double ms = t[2 * i + 1] > t[2 * i] ? (double)(t[2 * i + 1] - t[2 * i]) * period : 0;
            total += ms;
            if (o < (int)sizeof line - 64)
                o += snprintf(line + o, sizeof line - (size_t)o, " %ux%u%s%s %.2f", g_ts_info[i].w, g_ts_info[i].h,
                              g_ts_info[i].rt ? "rt" : "bb", g_ts_info[i].load ? "L" : "C", ms);
        }
        double span = t[2 * g_ts_n - 1] > t[0] ? (double)(t[2 * g_ts_n - 1] - t[0]) * period : 0;
        fprintf(stderr, "[gputime] %u passes, %.2f ms in passes, %.2f ms first to last:%s\n", g_ts_n, total, span, line);
        wgpuBufferUnmap(g_ts_read);
        return;
    }
    if (host_seconds() - last < 2.0) return;
    last = host_seconds();
    if (!g_ts_set) {
        WGPUQuerySetDescriptor qd = WGPU_QUERY_SET_DESCRIPTOR_INIT;
        qd.type = WGPUQueryType_Timestamp; qd.count = 2 * TS_MAX;
        g_ts_set = wgpuDeviceCreateQuerySet(g_dev, &qd);
        WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
        bd.size = 16 * TS_MAX; bd.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc;
        g_ts_resolve = wgpuDeviceCreateBuffer(g_dev, &bd);
        bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        g_ts_read = wgpuDeviceCreateBuffer(g_dev, &bd);
    }
    g_ts_capture = 1;   /* the next frame's passes */
    g_ts_n = 0;
}
#endif

/* where the game's picture is in the window, as fractions (x, y, w, h): touch input maps through it */
void render_picture_rect(float r[4]) {
    if (g_win_w <= 0 || g_win_h <= 0) { r[0] = r[1] = r[2] = r[3] = 0; return; }
    r[0] = (float)(g_win_w - g_surf_w) / 2.f / (float)g_win_w; r[1] = (float)(g_win_h - g_surf_h) / 2.f / (float)g_win_h;
    r[2] = (float)g_surf_w / (float)g_win_w; r[3] = (float)g_surf_h / (float)g_win_h;
}

static void exec_present(const CmdPresent *c) {
    begin_frame();
    use_backbuffer();
    if (!g_pass && !g_enc) begin_pass(1, 0, 1.f, 0);   /* nothing drawn: present black */
    double t_sub = host_seconds();
    submit();
    g_st.submit_t += host_seconds() - t_sub;
#ifndef __EMSCRIPTEN__
    ts_frame_done();
#endif
#ifndef __EMSCRIPTEN__
    if (c->screenshot) {
        char path[256]; snprintf(path, sizeof path, "/tmp/rc_frame_%u.png", c->frame); capture_png(path);
        if (getenv("RC_DUMP_RT")) {   /* also write that render target (a render-target surface id) */
            uint32_t id = (uint32_t)atoi(getenv("RC_DUMP_RT"));
            GSurf *gs = obj_get(id);
            if (gs && !gs->depth) { snprintf(path, sizeof path, "/tmp/rc_rt%u_%u.png", id, c->frame); capture_tex(gs->tex, gs->w, gs->h, path); }
        }
    }
#else
    (void)c;
#endif
    g_frame_open = 0;
#ifndef __EMSCRIPTEN__
    if (g_pt_on && g_qj_on) {   /* the submit thread hands it over after the frame's submit */
        if (g_pt_reconfig) { q_drain(); pt_wait(1); pt_reconfigure(); }
        QJob *j = calloc(1, sizeof *j);
        j->present = 1;
        q_push(j);
        return;
    }
    if (g_pt_on) {
        double t_w = host_seconds();
        pt_wait(1);   /* the previous frame is presented (it overlapped this frame's game work) */
        g_st.pt_wait_t += host_seconds() - t_w;
        { extern double g_hx[]; g_hx[5] += host_seconds() - t_w; }
        if (g_pt_reconfig) pt_reconfigure();
        pt_set(1);
        return;
    }
#endif
    present_to_window();
}

#ifndef __EMSCRIPTEN__
static void start_present_thread(void) {
    const char *e = getenv("RC_PRESENT_THREAD");
    if (e && !atoi(e)) return;
    pthread_t t;
    if (pthread_create(&t, NULL, present_thread, NULL)) return;
    pthread_detach(t);
    g_pt_on = 1;
    fprintf(stderr, "[wgpu] presenting on a separate thread\n");
    q_start();
}
#endif

/* ------------------------------------------------------------------ native: commands run immediately */

#ifndef __EMSCRIPTEN__
static void gpu_exec_cmd(const CmdHdr *h);
void gpu_exec(const CmdHdr *h) {
    int st = h->type == CMD_DRAW ? ST_DRAW : h->type == CMD_UPLOAD_TEX ? ST_UPLOAD_TEX : h->type == CMD_UPLOAD_SURF ? ST_UPLOAD_SURF
           : h->type == CMD_READBACK ? ST_READBACK : h->type == CMD_PRESENT ? ST_PRESENT : ST_OTHER;
    double t0 = host_seconds();
    gpu_exec_cmd(h);
    g_st.n[st]++;
    double dt_ = host_seconds() - t0;
    g_st.t[st] += dt_;
    extern double g_hx[];
    if (st == ST_UPLOAD_TEX) g_hx[2] += dt_;
}
static void gpu_exec_cmd(const CmdHdr *h) {
    {   /* RC_LOG_CMDS=1: a one-line summary of every non-draw command (and the draw count per frame in the Present line) */
        static int log = -1; static unsigned draws;
        if (log < 0) log = getenv("RC_LOG_CMDS") != NULL;
        if (log) {
            if (h->type == CMD_DRAW) draws++;
            else if (h->type == CMD_PRESENT) { fprintf(stderr, "[cmd] present (%u draws)\n", draws); draws = 0; }
            else if (h->type == CMD_TARGET) { const CmdTarget *t = (const CmdTarget *)h; fprintf(stderr, "[cmd] (%u draws so far) target color %u %ux%u depth %u logical %ux%u\n", draws, t->color_id, t->color_w, t->color_h, t->depth_id, t->lw, t->lh); }
            else if (h->type == CMD_CLEAR) { const CmdClear *t = (const CmdClear *)h; fprintf(stderr, "[cmd] clear flags %x color %08x\n", t->flags, t->color); }
            else if (h->type == CMD_UPLOAD_SURF) { const CmdUploadSurf *t = (const CmdUploadSurf *)h; fprintf(stderr, "[cmd] upload surface %u %ux%u\n", t->id, t->w, t->h_); }
            else if (h->type == CMD_UPLOAD_TEX) { const CmdUploadTex *t = (const CmdUploadTex *)h; if (t->w[0] >= 256) fprintf(stderr, "[cmd] upload texture %u %ux%u\n", t->id, t->w[0], t->h_[0]); }
        }
    }
    switch (h->type) {
    case CMD_TARGET: exec_target((const CmdTarget *)h); break;
    case CMD_CLEAR: exec_clear((const CmdClear *)h); break;
    case CMD_DRAW: exec_draw((const CmdDraw *)h); break;
    case CMD_UPLOAD_TEX: exec_upload_tex((const CmdUploadTex *)h); break;
    case CMD_DESTROY_TEX: exec_destroy(((const CmdDestroy *)h)->id, 0); break;
    case CMD_UPLOAD_SURF: exec_upload_surf((const CmdUploadSurf *)h); break;
    case CMD_DESTROY_SURF: exec_destroy(((const CmdDestroy *)h)->id, 1); break;
    case CMD_UPLOAD_VB: exec_upload_vb((const CmdUploadVb *)h); break;
    case CMD_DESTROY_VB: exec_destroy_vb(((const CmdDestroy *)h)->id); break;
    case CMD_PRESENT: exec_present((const CmdPresent *)h); break;
    case CMD_READBACK: {
        const CmdReadback *c = (const CmdReadback *)h;
        uint32_t gw, gh, stride;
        q_drain();
        pt_wait(1);   /* the wait below polls the device: keep the present thread out of it */
        WGPUBuffer buf = start_readback(c, &gw, &gh, &stride);
        if (buf) {
            WGPUBufferMapCallbackInfo mci = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
            mci.mode = WGPUCallbackMode_AllowProcessEvents;
            mci.callback = on_mapped;
            g_map_done = 0;
            wgpuBufferMapAsync(buf, WGPUMapMode_Read, 0, (uint64_t)stride * gh, mci);
            while (!g_map_done) wgpuInstanceProcessEvents(g_inst);
            store_pixels(c, wgpuBufferGetConstMappedRange(buf, 0, (uint64_t)stride * gh), gw, gh, stride);
            wgpuBufferUnmap(buf);
            wgpuBufferDestroy(buf);   /* free the staging memory now, not at garbage collection */
            wgpuBufferRelease(buf);
        }
        finish_readback(c);
        break;
    }
    }
}
#endif

/* ------------------------------------------------------------------ browser: queued commands */

#ifdef __EMSCRIPTEN__
#include <pthread.h>

typedef struct Chunk { uint8_t *buf; size_t len, off; int ends_frame; struct Chunk *next; } Chunk;
static pthread_mutex_t g_q_lock = PTHREAD_MUTEX_INITIALIZER;
static Chunk *g_q_head, *g_q_tail;
static int g_q_frames, g_ready;
static WGPUBuffer g_rb_buf;
static CmdReadback g_rb_cmd;
static uint32_t g_rb_gw, g_rb_gh, g_rb_stride;
static int g_rb_pending;

/* game thread: hand over a finished chunk of commands (takes ownership of buf) */
void web_queue_chunk(uint8_t *buf, size_t len, int ends_frame) {
    Chunk *ch = calloc(1, sizeof *ch);
    ch->buf = buf; ch->len = len; ch->ends_frame = ends_frame;
    pthread_mutex_lock(&g_q_lock);
    if (g_q_tail) g_q_tail->next = ch; else g_q_head = ch;
    g_q_tail = ch;
    if (ends_frame) g_q_frames++;
    pthread_mutex_unlock(&g_q_lock);
}
int web_queued_frames(void) { return __atomic_load_n(&g_q_frames, __ATOMIC_ACQUIRE); }
int web_gpu_ready(void) { return __atomic_load_n(&g_ready, __ATOMIC_ACQUIRE); }

static void on_rb_mapped(WGPUMapAsyncStatus st, WGPUStringView msg, void *u1, void *u2) {
    (void)msg; (void)u1; (void)u2;
    uint64_t size = (uint64_t)g_rb_stride * g_rb_gh;
    if (st == WGPUMapAsyncStatus_Success)
        store_pixels(&g_rb_cmd, wgpuBufferGetConstMappedRange(g_rb_buf, 0, size), g_rb_gw, g_rb_gh, g_rb_stride);
    wgpuBufferUnmap(g_rb_buf);
    wgpuBufferDestroy(g_rb_buf);
    wgpuBufferRelease(g_rb_buf);
    finish_readback(&g_rb_cmd);
    g_rb_pending = 0;
}

/* runs a chunk from where it stopped; returns 1 when it paused for a readback */
static int run_chunk(Chunk *ch) {
    while (ch->off < ch->len) {
        const CmdHdr *h = (const CmdHdr *)(ch->buf + ch->off);
        ch->off += h->size;
        switch (h->type) {
        case CMD_TARGET: exec_target((const CmdTarget *)h); break;
        case CMD_CLEAR: exec_clear((const CmdClear *)h); break;
        case CMD_DRAW: exec_draw((const CmdDraw *)h); break;
        case CMD_UPLOAD_TEX: exec_upload_tex((const CmdUploadTex *)h); break;
        case CMD_DESTROY_TEX: exec_destroy(((const CmdDestroy *)h)->id, 0); break;
        case CMD_UPLOAD_SURF: exec_upload_surf((const CmdUploadSurf *)h); break;
        case CMD_DESTROY_SURF: exec_destroy(((const CmdDestroy *)h)->id, 1); break;
        case CMD_UPLOAD_VB: exec_upload_vb((const CmdUploadVb *)h); break;
        case CMD_DESTROY_VB: exec_destroy_vb(((const CmdDestroy *)h)->id); break;
        case CMD_PRESENT: exec_present((const CmdPresent *)h); break;
        case CMD_READBACK: {
            const CmdReadback *c = (const CmdReadback *)h;
            WGPUBuffer buf = start_readback(c, &g_rb_gw, &g_rb_gh, &g_rb_stride);
            if (!buf) { finish_readback(c); break; }
            g_rb_cmd = *c;
            g_rb_buf = buf;
            g_rb_pending = 1;
            WGPUBufferMapCallbackInfo mci = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
            mci.mode = WGPUCallbackMode_AllowSpontaneous;
            mci.callback = on_rb_mapped;
            wgpuBufferMapAsync(buf, WGPUMapMode_Read, 0, (uint64_t)g_rb_stride * g_rb_gh, mci);
            return 1;
        }
        }
    }
    return 0;
}

/* page thread: run queued commands. A chunk that ends a frame (and so presents) only runs when
   allow_present is set - the page passes 1 from requestAnimationFrame. Returns frames presented. */
EMSCRIPTEN_KEEPALIVE int rc_gpu_pump(int allow_present) {
    int presented = 0;
    if (!g_ready) return 0;
    while (!g_rb_pending) {
        pthread_mutex_lock(&g_q_lock);
        Chunk *ch = g_q_head;
        pthread_mutex_unlock(&g_q_lock);
        if (!ch || (ch->ends_frame && !allow_present)) break;
        if (run_chunk(ch)) break;   /* readback in flight: resume when it is mapped */
        pthread_mutex_lock(&g_q_lock);
        g_q_head = ch->next;
        if (!g_q_head) g_q_tail = NULL;
        if (ch->ends_frame) g_q_frames--;
        pthread_mutex_unlock(&g_q_lock);
        if (ch->ends_frame) { presented++; allow_present = 0; }
        free(ch->buf);
        free(ch);
    }
    return presented;
}

EMSCRIPTEN_KEEPALIVE void rc_set_canvas_size(int w, int h) { g_window_w = w; g_window_h = h; }

static void on_web_device(WGPURequestDeviceStatus st, WGPUDevice d, WGPUStringView msg, void *u1, void *u2) {
    (void)u1; (void)u2;
    if (st != WGPURequestDeviceStatus_Success) { fprintf(stderr, "[wgpu] device: %.*s\n", (int)msg.length, msg.data); return; }
    g_dev = d;
    g_queue = wgpuDeviceGetQueue(g_dev);
    create_resources();
    __atomic_store_n(&g_ready, 1, __ATOMIC_RELEASE);
}

static void request_web_adapter(int attempt);
static void on_web_adapter(WGPURequestAdapterStatus st, WGPUAdapter a, WGPUStringView msg, void *u1, void *u2) {
    (void)u2;
    int attempt = (int)(intptr_t)u1;
    if (st != WGPURequestAdapterStatus_Success) {
        fprintf(stderr, "[wgpu] adapter (attempt %d): %.*s\n", attempt, (int)msg.length, msg.data);
        if (attempt < 2) request_web_adapter(attempt + 1);   /* default preference, then the software fallback */
        return;
    }
    if (attempt == 2) fprintf(stderr, "[wgpu] using the browser's software (fallback) adapter: expect it to be slow\n");
    g_adapter = a;
    WGPUDeviceDescriptor dd = WGPU_DEVICE_DESCRIPTOR_INIT;
    dd.uncapturedErrorCallbackInfo.callback = on_error;
    static const WGPUFeatureName bc = WGPUFeatureName_TextureCompressionBC;
    g_gpu_bc = wgpuAdapterHasFeature(g_adapter, bc) && !getenv("RC_NO_BC");   /* DXT textures stay compressed */
    if (g_gpu_bc) { dd.requiredFeatureCount = 1; dd.requiredFeatures = &bc; }
    fprintf(stderr, "[wgpu] BC texture compression: %s\n", g_gpu_bc ? "yes" : "no (DXT textures are converted to RGBA8)");
    WGPURequestDeviceCallbackInfo dci = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    dci.mode = WGPUCallbackMode_AllowSpontaneous;
    dci.callback = on_web_device;
    wgpuAdapterRequestDevice(g_adapter, &dd, dci);
}

/* page thread: create the device for <canvas id="canvas"> (asynchronous; the game waits for it) */
EMSCRIPTEN_KEEPALIVE void rc_gpu_init(int w, int h) {
    g_window_w = w; g_window_h = h;
    g_inst = wgpuCreateInstance(NULL);
    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector cs = WGPU_EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR_INIT;
    cs.selector = sv("#canvas");
    WGPUSurfaceDescriptor sd = WGPU_SURFACE_DESCRIPTOR_INIT;
    sd.nextInChain = &cs.chain;
    g_surface = wgpuInstanceCreateSurface(g_inst, &sd);
    request_web_adapter(0);
}
/* attempt 0: high-performance GPU; 1: any GPU; 2: the browser's software fallback adapter, if it has one */
static void request_web_adapter(int attempt) {
    WGPURequestAdapterOptions ao = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    if (attempt == 0) ao.powerPreference = WGPUPowerPreference_HighPerformance;
    if (attempt == 2) ao.forceFallbackAdapter = 1;
    WGPURequestAdapterCallbackInfo aci = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    aci.mode = WGPUCallbackMode_AllowSpontaneous;
    aci.callback = on_web_adapter;
    aci.userdata1 = (void *)(intptr_t)attempt;
    wgpuInstanceRequestAdapter(g_inst, &ao, aci);
}
#endif
