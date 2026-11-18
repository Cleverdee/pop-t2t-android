# libpop3.so (Prince of Persia: The Two Thrones): the recompiled game (out_all/*.c from translator/x86toc.py), runtime/, host/ with the
# Android window layer in place of host/window.c, and SDL2 built from source as libSDL2.so.
cmake_minimum_required(VERSION 3.22)
project(pop3 C)

set(RC_REPO "" CACHE PATH "repository root")
set(SDL2_SRC "" CACHE PATH "SDL2 source tree")
set(RC_OPT "-O2" CACHE STRING "optimisation flags for the game code")
option(RC_LTO "link-time optimisation (slow to link)" OFF)
set(WGPU "$ENV{RC_WGPU}" CACHE PATH "wgpu-native for aarch64-linux-android")
file(TO_CMAKE_PATH "${RC_REPO}" RC_REPO)   # forward slashes: RC_PROFILE below becomes a C string
get_filename_component(RC_REPO "${RC_REPO}" REALPATH)
set(GEN "${RC_REPO}/out_all")

if(NOT EXISTS "${GEN}/recomp_stubs.c")
  message(FATAL_ERROR "no generated code in ${GEN}: run android/build.ps1 (it translates the game first)")
endif()
# the packed (GOG) exe's UPX unpacker stub, translated by tools/upx_stub.py (android/build.ps1)
set(UPX_SRC "${RC_REPO}/out_upx/upx_stub.c")
if(NOT EXISTS "${UPX_SRC}")
  message(FATAL_ERROR "no ${UPX_SRC}: run android/build.ps1")
endif()

add_subdirectory("${SDL2_SRC}" "${CMAKE_BINARY_DIR}/SDL2")

file(GLOB GEN_SRC CONFIGURE_DEPENDS "${GEN}/*.c")
file(GLOB HOST_SRC CONFIGURE_DEPENDS "${RC_REPO}/host/*.c")
# Mac window (Metal/objc), browser glue and the signal-based watchpoints (needs API 33) stay out
list(REMOVE_ITEM HOST_SRC "${RC_REPO}/host/window.c" "${RC_REPO}/host/web.c" "${RC_REPO}/host/watch.c")

add_library(wgpu_native STATIC IMPORTED)
set_target_properties(wgpu_native PROPERTIES IMPORTED_LOCATION "${WGPU}/lib/libwgpu_native.a")

add_library(pop3 SHARED
  ${GEN_SRC} "${UPX_SRC}"
  "${RC_REPO}/runtime/runtime.c" "${RC_REPO}/runtime/interp.c"
  ${HOST_SRC}
  "${RC_REPO}/android/host/window_android.c" "${RC_REPO}/android/host/pop3_android_stubs.c")

# FFmpeg with only the Bink demuxer and decoders (host/bink.c), prebuilt for arm64: built by third_party/build_ffmpeg.sh
set(FFMPEG "${RC_REPO}/third_party/ffmpeg-android-arm64")
foreach(lib avformat avcodec avutil)
  add_library(ff_${lib} STATIC IMPORTED)
  set_target_properties(ff_${lib} PROPERTIES IMPORTED_LOCATION "${FFMPEG}/lib/lib${lib}.a")
endforeach()

target_include_directories(pop3 PRIVATE "${RC_REPO}/runtime" "${RC_REPO}/host" "${GEN}" "${WGPU}/include")
set_source_files_properties("${RC_REPO}/host/bink.c" PROPERTIES INCLUDE_DIRECTORIES "${FFMPEG}/include")
target_compile_definitions(pop3 PRIVATE "RC_PROFILE=\"${RC_REPO}/android/profile_pop3.h\"")
# the x87 rounding mode is mirrored into the host FPU at runtime (as in build.sh)
target_compile_options(pop3 PRIVATE -frounding-math -w)
# calls inside libpop3.so bind directly instead of through the PLT (~4% of the game thread in simpleperf)
target_compile_options(pop3 PRIVATE -fno-semantic-interposition)
target_link_options(pop3 PRIVATE -Wl,-Bsymbolic)
separate_arguments(RC_OPT_LIST UNIX_COMMAND "${RC_OPT}")
set_source_files_properties(${GEN_SRC} "${UPX_SRC}" "${RC_REPO}/runtime/runtime.c" "${RC_REPO}/runtime/interp.c"
                            PROPERTIES COMPILE_OPTIONS "${RC_OPT_LIST}")
set_source_files_properties(${HOST_SRC} "${RC_REPO}/android/host/window_android.c" "${RC_REPO}/android/host/pop3_android_stubs.c"
                            PROPERTIES COMPILE_OPTIONS "-O2")
if(RC_LTO)
  target_compile_options(pop3 PRIVATE -flto=thin)
  target_link_options(pop3 PRIVATE -flto=thin -Wl,--thinlto-jobs=all)
endif()

target_link_libraries(pop3 PRIVATE SDL2 wgpu_native ff_avformat ff_avcodec ff_avutil android aaudio log z m dl)
