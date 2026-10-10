# Type stability: the small routing/constructor functions on the call path must infer concretely.
# Pure ones need no device; the Metal ones run only where Metal is functional. `@inferred` alone for
# launches (a second run would dispatch a second kernel); `@test_nowarn` where a warning is the failure.
# The op is SDPAOps: `FlashAttn` is a plain binary, `SDPAKA` a KA binary with `extras`.

using Test, KernelDispatch, SDPAOps
# Metal (and the jax-mps client) only exist on a Mac: elsewhere these sections skip.
@static if Sys.isapple()
    using Metal
end
isdefined(@__MODULE__, :Residual) || include("fixtures.jl")

const KOI = KernelDispatch

@testset "OutArray" begin
    @test (@inferred OutArray(Float32, 8)) isa OutArray{Float32,1}
    @test (@inferred OutArray(Float32, 2, 3)) isa OutArray{Float32,2}
    @test (@inferred OutArray(Float32, (2, 3, 4))) isa OutArray{Float32,3}
    @test (@inferred OutArray(Float32, 8; zero=true)).zero
    @test @inferred(eltype(OutArray(Float16, 4))) === Float16
end

@testset "routing on host arrays" begin
    x = rand(Float32, 4, 6)
    @test @inferred(KOI.backend_of(x)) isa KOI.CPUBackend
    @test @inferred(KOI.backend_of(x, x)) isa KOI.CPUBackend
    @test @inferred(KOI.backend_of(view(x, :, 1:2))) isa KOI.CPUBackend
    be = KOI.backend_of(x)
    @test @inferred(KOI.device_dense(be, x)) === x
    @test @inferred(KOI.device_dense(be, 2.0f0)) === 2.0f0
    @test @inferred(KOI.device_dense(be, Val(64))) === Val(64)
end

# `extras` of a length known only at run time: merged into the arguments, they would hide which
# argument is an output. Bound separately, they cannot reach `call_binary`'s result.
struct SDPADyn <: AbstractKernel end
KOI.kernelname(::SDPADyn) = :dyn
KOI.variant_key(op::SDPA, ::SDPADyn, args...) = variant_key(op, SDPAKA(), args...)
KOI.extras(::SDPA, ::SDPADyn, b::KernelBinary) = Tuple(Val(b.params.tg) for _ in 1:1)


# The binaries' argument lists, each in its own order (see SDPAOps.jl).
sdpa_scale(q) = inv(sqrt(Float32(size(q, 1))))
sdpa_binary_args(::FlashAttn, o, q, k, v) = (q, k, v, o, Int32(size(q, 2)), Int32(size(k, 2)),
    Int32(size(q, 1)), Int32(size(v, 1)), sdpa_scale(q))
sdpa_binary_args(::Union{SDPAKA,SDPADyn}, o, q, k, v) = (o, q, k, v, Int32(size(q, 1)), Int32(size(q, 2)),
    Int32(size(k, 2)), Int32(size(v, 1)), sdpa_scale(q))

@testset "op hooks" begin
    q, k, v = rand(Float32, 16, 8), rand(Float32, 16, 6), rand(Float32, 4, 6)
    op, kern = SDPA(), SDPAKA()
    args = sdpa_binary_args(kern, OutArray(Float32, 4, 8), q, k, v)
    @test @inferred(variant_key(op, kern, args...)) === (:Float32,)
    b = KernelBinary("unused.metallib"; threadgroup=64, params=(; tg=64))
    # `KernelBinary.params` is an abstract `NamedTuple` (binaries live in a `Dict` and reload from a manifest),
    # so a hook that reads a tuning number from it cannot infer: the documented one dynamic dispatch
    # per launch. `call_binary` binds `extras` apart from `args`. The launch size reads only `tile`.
    @test extras(op, kern, b) === (Val(64),)
    @test_broken (@inferred extras(op, kern, b); true)
    x, y = rand(Float32, 8), rand(Float32, 8)
    l = @inferred KOI.prepare_launch(b, OutArray(Float32, 8), x, y, 2.0f0; extent=8)
    @test l.groups === l.grid === (1, 1, 1)
    @test isconcretetype(typeof(l))
end

@testset "host path of an op" begin
    q, k, v = rand(Float32, 16, 8), rand(Float32, 16, 6), rand(Float32, 4, 6)
    @test_nowarn @inferred SDPA()(q, k, v)
end

if !Sys.isapple() || !Metal.functional()
    @warn "Metal not functional; skipping Metal inference tests" maxlog = 1
else
    @testset "routing on Metal arrays" begin
        W = Base.get_extension(KernelDispatch, :KernelDispatchMetalExt).WrappedMtlArray
        x = MtlArray(rand(Float32, 4, 6))
        tag = KOI.MetalBackendTag()
        plain = (x, reshape(x, 6, 4), view(x, :, 2:3))        # Metal.jl hands back an `MtlArray`
        wrapped = (view(x, 2:3, :), x', transpose(x))         # these stay wrappers
        for a in plain
            @test @inferred(KOI.backend_of(a)) === tag
            @test @inferred(KOI.device_dense(tag, a)) === a
        end
        for a in wrapped
            @test a isa W
            @test @inferred(KOI.backend_of(a)) === tag
            @test @inferred(KOI.device_dense(tag, a)) isa MtlArray
        end
        @test @inferred(KOI.backend_of(x, rand(Float32, 3))) === tag     # the first array decides
        @test @inferred(KOI.device_dense(tag, 2.0f0)) === 2.0f0
    end

    @testset "KA backend hooks" begin
        tag, cuda = KOI.MetalBackendTag(), KOI.CUDABackendTag()
        @test @inferred(KOI.ka_state_words(tag)) === (UInt32(0),)
        @test @inferred(KOI.ka_backend_object(tag)) isa Metal.MetalBackend
        @test @inferred(KOI.ka_grid(tag, (2, 3, 4))) === (24, 1, 1)             # flat
        ctx, ghost = KOI.ka_context(tag, (2, 1, 1), 64)
        @test ghost isa Bool
        # No extension, no default: a backend that forgets a hook fails loudly.
        for f in (KOI.ka_state_words, KOI.ka_backend_object)
            @test_throws ArgumentError f(cuda)
        end
        @test_throws ArgumentError KOI.ka_grid(cuda, (1, 1, 1))
        @test_throws ArgumentError KOI.ka_context(cuda, (1, 1, 1), 64)
    end

    @testset "launch path" begin
        d, n, m, dv = 16, 8, 6, 4
        q, k, v = (MtlArray(rand(Float32, r, c)) for (r, c) in ((d, n), (d, m), (dv, m)))
        ref = KernelDispatch.host(SDPA(), Array(q), Array(k), Array(v))
        register_kernel!(SDPA(), SDPADyn())
        add_variant!(SDPA(), SDPADyn(), (:Float32,); KOI.variant(SDPA(), SDPAKA(), (:Float32,))...)
        @test !isconcretetype(Base.promote_op(extras, SDPA, SDPADyn, KernelBinary))   # length unknown
        # `extras` read `KernelBinary.params` (typed at run time): the result still infers.
        for kern in (FlashAttn(), SDPAKA(), SDPADyn()), zero in (false, true)
            args = sdpa_binary_args(kern, OutArray(Float32, dv, n; zero), q, k, v)
            z = @inferred KOI.call_binary(SDPA(), kern, :fwd, args...; extent=n)
            @test Array(only(z)) ≈ ref rtol = 1e-3
        end
        @test @inferred(forward(SDPA(), FlashAttn(), q, k, v)) isa MtlArray
        # The op call infers too: its kernel is a method (`@default_kernel`), not a table entry.
        @test_nowarn @inferred SDPA()(q, k, v)
        user(q, k, v) = sum(SDPA()(q, k, v)) + 1.0f0
        @test @inferred(user(q, k, v)) isa Float32
    end

    # The binary-level layers, for a KA-compiled binary (Vector{Any} prelude, flat grid) and a plain
    # one: the `Launch` keeps the argument types, so running and timing it infer.
    @testset "Launch path" begin
        d, n, m, dv = 16, 300, 6, 4
        q, k, v = (MtlArray(rand(Float32, r, c)) for (r, c) in ((d, n), (d, m), (dv, m)))
        ka = KOI.variant(SDPA(), SDPAKA(), (:Float32,))[:fwd]
        plain = KOI.variant(SDPA(), FlashAttn(), (:Float32,))[:fwd]
        out() = OutArray(Float32, dv, n)
        cases = (ka => (sdpa_binary_args(SDPAKA(), out(), q, k, v)..., Val(64)),
                 plain => sdpa_binary_args(FlashAttn(), out(), q, k, v))
        for (b, args) in cases
            l = @inferred prepare_launch(b, args...; extent=n)
            @test isconcretetype(typeof(l))
            @test only(@inferred KOI.execute(l)) isa MtlMatrix{Float32}
            @test only(@inferred run_binary(b, args...; extent=n)) isa MtlMatrix{Float32}
            @test @inferred(time_binary(l; reps=1, warmup=0)) isa Float64
            @test @inferred(time_binary(b, args...; extent=n, reps=1, warmup=0)) isa Float64
        end
        # By op and kernel: the registered variant, with `extras` bound for it. (`FlashAttn` has none.)
        @test @inferred(time_binary(SDPA(), FlashAttn(), :fwd, last(last(cases))...; extent=n, reps=1,
            warmup=0)) isa Float64
    end
end
