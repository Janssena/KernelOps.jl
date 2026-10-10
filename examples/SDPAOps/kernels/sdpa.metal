// Scaled dot-product attention, one thread per query: a hand-written Metal kernel standing in for a
// binary from another toolchain (Triton, say). Compile with
//
//     xcrun -sdk macosx metal sdpa.metal -o sdpa.metallib
//
// Column-major (Julia) layout: q is d×n, k is d×m, v is dv×m, o is dv×n; each column is one
// query/key/value. Argument order, Triton style: inputs, the output, then sizes and the scale.
// KernelDispatch binds every argument as a buffer (a scalar too), in order, from index 0.

#include <metal_stdlib>
using namespace metal;

kernel void sdpa_fwd(device const float* q [[buffer(0)]],
                     device const float* k [[buffer(1)]],
                     device const float* v [[buffer(2)]],
                     device float* o       [[buffer(3)]],
                     constant int& n       [[buffer(4)]],
                     constant int& m       [[buffer(5)]],
                     constant int& d       [[buffer(6)]],
                     constant int& dv      [[buffer(7)]],
                     constant float& scale [[buffer(8)]],
                     uint group  [[threadgroup_position_in_grid]],
                     uint lane   [[thread_position_in_threadgroup]],
                     uint tgsize [[threads_per_threadgroup]]) {
    int i = group * tgsize + lane;           // this thread's query, 0-based
    if (i >= n) return;
    device const float* qi = q + i * d;
    device float* oi = o + i * dv;

    float mx = -INFINITY;                    // pass 1: the largest score, for a stable softmax
    for (int j = 0; j < m; ++j) {
        float s = 0.0f;
        for (int t = 0; t < d; ++t) s += qi[t] * k[j * d + t];
        mx = max(mx, s * scale);
    }
    float denom = 0.0f;                      // pass 2: weights and the weighted sum of values
    for (int c = 0; c < dv; ++c) oi[c] = 0.0f;
    for (int j = 0; j < m; ++j) {
        float s = 0.0f;
        for (int t = 0; t < d; ++t) s += qi[t] * k[j * d + t];
        float p = exp(s * scale - mx);
        denom += p;
        for (int c = 0; c < dv; ++c) oi[c] += p * v[j * dv + c];
    }
    for (int c = 0; c < dv; ++c) oi[c] /= denom;
}
