# The tutorial example package (examples/SDPAOps) as the integration fixture: a real op with two
# real kernels (a prebuilt `.metallib` and a KernelAbstractions kernel) and a plain-Julia `host` and
# `backward`. The fixture ops in `fixtures.jl` cover what it cannot (a backward binary, registry
# edge cases, binary-level launch features); this file checks that the machinery composes into what
# the tutorials teach.
#
#   host        no device needed: against a column-by-column reference
#   backward    no device needed: against finite differences
#   Metal       both kernels, odd sizes, the lazy variant build and its manifest
#   example     the example's own test file, run as part of this suite
#   Reactant    both kernels traced as custom calls (needs the jax-mps plugin, see mps_client.jl)

using Test, Random, KernelOps, SDPAOps, Enzyme, Reactant
# Metal (and the jax-mps client) only exist on a Mac: elsewhere these sections skip.
@static if Sys.isapple()
    using Metal
end
isdefined(@__MODULE__, :Residual) || include("fixtures.jl")     # also keeps SDPAOps' manifests out of the user's cache
Sys.isapple() && (isdefined(@__MODULE__, :mps_client) || include("mps_client.jl"))

# Independent of `SDPAOps.host`: one query at a time, in Float64.
function sdpa_reference(q, k, v)
    d, n = size(q)
    o = zeros(Float64, size(v, 1), n)
    for i in 1:n
        s = [sum(Float64.(q[:, i]) .* Float64.(k[:, j])) / sqrt(d) for j in 1:size(k, 2)]
        p = exp.(s .- maximum(s))
        o[:, i] = Float64.(v) * (p ./ sum(p))
    end
    return o
end

inputs(rng, d, n, m, dv) = (randn(rng, Float32, d, n), randn(rng, Float32, d, m), randn(rng, Float32, dv, m))

# (d, n, m, dv): `n` below, at, and above a threadgroup of 64; a single query and key.
const SDPA_SIZES = ((16, 100, 70, 8), (4, 1, 1, 1), (8, 64, 64, 4), (32, 129, 5, 16))

@testset "host" begin
    rng = Xoshiro(1)
    for (d, n, m, dv) in SDPA_SIZES
        q, k, v = inputs(rng, d, n, m, dv)
        o = SDPA()(q, k, v)                         # host arrays: the host path
        @test size(o) == (dv, n)
        @test o ≈ sdpa_reference(q, k, v) rtol = 1e-4
    end
end

@testset "backward against finite differences" begin
    rng = Xoshiro(2)
    d, n, m, dv = 4, 3, 5, 2
    q, k, v = (Float64.(x) for x in inputs(rng, d, n, m, dv))
    w = randn(rng, dv, n)
    loss(q, k, v) = sum(KernelOps.host(SDPA(), q, k, v) .* w)
    dq, dk, dv_ = KernelOps.backward(SDPA(), FlashAttn(), (q, k, v), nothing, w)
    function fd(f, x, i; h=1e-6)
        xp, xm = copy(x), copy(x)
        xp[i] += h
        xm[i] -= h
        return (f(xp) - f(xm)) / 2h
    end
    for i in eachindex(q)
        @test dq[i] ≈ fd(x -> loss(x, k, v), q, i) atol = 1e-6
    end
    for i in eachindex(k)
        @test dk[i] ≈ fd(x -> loss(q, x, v), k, i) atol = 1e-6
    end
    for i in eachindex(v)
        @test dv_[i] ≈ fd(x -> loss(q, k, x), v, i) atol = 1e-6
    end
    # The gradients do not depend on which kernel produced the forward.
    @test KernelOps.backward(SDPA(), SDPAKA(), (q, k, v), nothing, w) == (dq, dk, dv_)
end

if !Sys.isapple() || !Metal.functional()
    @warn "Metal not functional; skipping SDPAOps Metal tests" maxlog = 1
else
    @testset "Metal" begin
        rng = Xoshiro(3)
        @testset "$kern, d=$d n=$n m=$m dv=$dv" for kern in (FlashAttn(), SDPAKA()), (d, n, m, dv) in SDPA_SIZES
            q, k, v = inputs(rng, d, n, m, dv)
            o = KernelOps.forward(SDPA(), kern, MtlArray(q), MtlArray(k), MtlArray(v))
            @test o isa MtlArray && size(o) == (dv, n)
            @test Array(o) ≈ sdpa_reference(q, k, v) rtol = 1e-3
        end

        @testset "variants are built on first use and persisted" begin
            q, k, v = inputs(rng, 16, 10, 7, 8)
            qd, kd, vd = MtlArray(q), MtlArray(k), MtlArray(v)
            for kern in (FlashAttn(), SDPAKA())
                KernelOps.forward(SDPA(), kern, qd, kd, vd)
                vs = KernelOps.variants(SDPA(), kern)
                @test collect(keys(vs)) == [(:Float32,)]
                bin = vs[(:Float32,)][:fwd]
                @test isfile(bin.file)
                @test isfile(joinpath(KernelOps.kernel_dir(SDPA(), kern), "manifest.toml"))
                t = mtime(bin.file)
                KernelOps.forward(SDPA(), kern, qd, kd, vd)             # found, not rebuilt
                @test KernelOps.variants(SDPA(), kern) === vs && mtime(bin.file) == t
            end
        end

        @testset "no variant for an element type the kernel lacks" begin
            # `FlashAttn` is Float32 only: its `build!` declines any other key.
            @test_throws ArgumentError KernelOps.variant(SDPA(), FlashAttn(), (:Float16,))
        end
    end
end

# The example's own test file, in a module of its own: its top-level helpers stay out of this one,
# and the file stays the single source of truth for what the tutorial claims.
if !Sys.isapple() || !Metal.functional()
    @warn "Metal not functional; skipping the SDPAOps example's own tests" maxlog = 1
else
    @testset "SDPAOps example tests" begin
        Base.include(Module(:SDPAExampleTests), joinpath(pkgdir(SDPAOps), "test", "runtests.jl"))
    end
end

sdpa_mps = Sys.isapple() ? mps_client() : nothing
if sdpa_mps === nothing || !Metal.functional()
    @warn "no jax-mps Metal client (or Metal not functional); skipping traced SDPAOps tests" maxlog = 1
else
    prev = Reactant.XLA.default_backend()
    Reactant.set_default_backend(sdpa_mps)
    try
        @testset "Reactant" begin
            rng = Xoshiro(4)
            q, k, v = inputs(rng, 16, 100, 70, 8)
            ref = sdpa_reference(q, k, v)
            args = map(Reactant.to_rarray, (q, k, v))
            # The kernel is named in the call, not switched: nothing global changes under the trace.
            for kern in (FlashAttn(), SDPAKA())
                @testset "$kern" begin
                    f(q, k, v) = KernelOps.forward(SDPA(), kern, q, k, v)
                    @test Array(@jit f(args...)) ≈ ref rtol = 1e-3
                    @test occursin("mps.metal_kernel_lib", repr(Reactant.@code_hlo optimize = false f(args...)))
                end
            end
            @test current_kernel(SDPA()) === :flash
        end
    finally
        Reactant.set_default_backend(prev)
    end
end
