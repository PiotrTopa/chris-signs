// C entry points into signd's own image code for the Python evaluation of lane L24-S (ctypes): JPEG -> NV12
// exactly as signd_replay does it, and signd's resampler (nv12_region_to_rgb_planar) and glow check, so the
// offline detections are made from the same pixels the C++ pipeline sees. Build: tools/signs_eval/build_pre.sh.
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <vector>

#include <jpeglib.h>

#include "verify.h"
#include "yolo.h"

using namespace roadsense::signs;

namespace {
struct JErr {
    jpeg_error_mgr pub;
    jmp_buf jb;
};
void jerr_exit(j_common_ptr c) { longjmp(reinterpret_cast<JErr *>(c->err)->jb, 1); }

Nv12View view_of(const uint8_t *nv12, int w, int h) {
    Nv12View v;
    v.y = nv12;
    v.uv = nv12 + size_t(w) * h;
    v.width = w, v.height = h, v.y_stride = v.uv_stride = w;
    return v;
}
}  // namespace

extern "C" {

// Decodes a JPEG and converts it to NV12 (signd's rgb_to_nv12). out must hold w*h*3/2 bytes of the expected
// size (*w, *h in: expected size, out: actual). Returns 0 on success.
int pre_jpeg_nv12(const uint8_t *jpg, size_t n, uint8_t *out, int *w, int *h) {
    jpeg_decompress_struct cinfo;
    JErr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jerr_exit;
    std::vector<uint8_t> px;
    if (setjmp(jerr.jb)) {
        jpeg_destroy_decompress(&cinfo);
        return 1;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, jpg, n);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);
    const int W = int(cinfo.output_width), H = int(cinfo.output_height);
    px.resize(size_t(W) * H * 3);
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row = &px[size_t(cinfo.output_scanline) * W * 3];
        jpeg_read_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    if ((W & ~1) != *w || (H & ~1) != *h) {
        *w = W & ~1, *h = H & ~1;
        return 2;
    }
    Nv12Image img;
    rgb_to_nv12(px.data(), W, H, img);
    std::memcpy(out, img.data.data(), img.data.size());
    return 0;
}

// signd's nv12_region_to_rgb_planar on a packed NV12 buffer (strides = w).
void pre_region(const uint8_t *nv12, int w, int h, double sx, double sy, double sw, double sh, int dw, int dh, float *dst,
                int dst_w, int dst_h, int dx, int dy) {
    nv12_region_to_rgb_planar(view_of(nv12, w, h), sx, sy, sw, sh, dw, dh, dst, dst_w, size_t(dst_w) * dst_h, dx, dy);
}

// 32x32 appearance resample of a box (as SignPipeline) into app (3*32*32 floats); returns the glow share.
float pre_appearance(const uint8_t *nv12, int w, int h, float x0, float y0, float x1, float y1, float *app) {
    nv12_region_to_rgb_planar(view_of(nv12, w, h), x0, y0, x1 - x0, y1 - y0, kAppearanceSize, kAppearanceSize, app,
                              kAppearanceSize, size_t(kAppearanceSize) * kAppearanceSize, 0, 0);
    return glow_score(app, kAppearanceSize);
}

int pre_is_round_red(const char *merged) { return is_round_red(merged) ? 1 : 0; }
}
