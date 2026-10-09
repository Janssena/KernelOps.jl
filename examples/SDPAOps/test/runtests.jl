# The tutorial example, checked on Metal: both kernels against the host implementation, switching,
# the bindings each binary gets, and Enzyme gradients against Enzyme through the host path.

using SDPAOps, KernelOps, Metal, Enzyme, Random, Test

rng = Xoshiro(1)
d, n, m, dv = 16, 100, 70, 8
q, k, v = randn(rng, Float32, d, n), randn(rng, Float32, d, m), randn(rng, Float32, dv, m)
qd, kd, vd = MtlArray(q), MtlArray(k), MtlArray(v)
ref = SDPA()(q, k, v)                                    # host arrays: the host path

@testset "kernels" begin
    @test current_kernel(:sdpa) === :flash                # the `@default_kernel`
    @test Array(SDPA()(qd, kd, vd)) ≈ ref
    # One call with a particular kernel: its `forward`, no switch.
    @test Array(KernelOps.forward(SDPA(), SDPAOps.SDPAKA(), qd, kd, vd)) ≈ ref
    @test current_kernel(SDPA()) === :flash               # nothing switched
    rec(kern) = map(first, only(KernelOps.record_bindings(() -> KernelOps.forward(SDPA(), kern, qd, kd, vd))).slots)
    @test rec(SDPAOps.FlashAttn()) == (:in, :in, :in, :out, :scalar, :scalar, :scalar, :scalar, :scalar)
    # A KA binary takes the state word and KA's context first.
    @test rec(SDPAOps.SDPAKA()) == (:bytes, :bytes, :out, :in, :in, :in, :scalar, :scalar, :scalar, :scalar, :scalar)
end

w = randn(rng, Float32, dv, n)
loss(q, k, v, w) = sum(SDPA()(q, k, v) .* w)
dq, dk, dv_ = zero(q), zero(k), zero(v)
Enzyme.autodiff(Reverse, loss, Active, Duplicated(q, dq), Duplicated(k, dk), Duplicated(v, dv_), Const(w))

f(q, k, v) = SDPA()(q, k, v)
function gpu_grads(qd, kd, vd, w)
    gq, gk, gv = zero(qd), zero(kd), zero(vd)
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated,
        Duplicated{typeof(qd)}, Duplicated{typeof(kd)}, Duplicated{typeof(vd)})
    args = (Const(f), Duplicated(qd, gq), Duplicated(kd, gk), Duplicated(vd, gv))
    tape, o, dout = fwd(args...)
    dout .= MtlArray(w)
    rev(args..., tape)
    return o, gq, gk, gv
end

# Switched at top level: a thunk holds the kernel selected when it was built.
for kname in (:flash, :ka)
    use_kernel!(:sdpa, kname)
    @eval @testset "Enzyme, $($(QuoteNode(kname)))" begin
        o, gq, gk, gv = gpu_grads(qd, kd, vd, w)
        @test Array(o) ≈ ref
        @test Array(gq) ≈ dq rtol = 1e-4
        @test Array(gk) ≈ dk rtol = 1e-4
        @test Array(gv) ≈ dv_ rtol = 1e-4
    end
end
reset_kernel!(:sdpa)
