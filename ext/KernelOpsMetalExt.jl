module KernelOpsMetalExt

# Metal's half of KernelOps: binding arguments to a compiled `.metallib` and dispatching it (the eager
# `bind_launch` every `call_binary` ends in), and ahead-of-time compilation of
# KernelAbstractions kernels.

import Metal: AS, GPUCompiler, LinearAlgebra

using Metal
using KernelOps

const KO = KernelOps

dptr(::Type{T}) where {T} = Core.LLVMPtr{T,AS.Device}

# --- binding and dispatch -------------------------------------------------------------------------

# Pipeline creation (metallib load -> MTLFunction -> pipeline state) is expensive; on the launch path
# it put a large constant on every measurement and dominated small problems.
const _PIPES = Dict{Tuple{String,String},Any}()
function pipeline(path, entry)
    get!(_PIPES, (path, entry)) do
        dev = Metal.device()
        Metal.MTLComputePipelineState(dev,
            Metal.MTLFunction(Metal.MTLLibraryFromFile(dev, path), entry))
    end
end

# Metal caps live command queues at ~64; one per launch survived a few reps and then threw
# `UndefRefError`, after first producing impossibly fast timings. One queue per process.
const _QUEUE = Ref{Any}(nothing)
queue() = (isnothing(_QUEUE[]) && (_QUEUE[] = Metal.MTLCommandQueue(Metal.device())); _QUEUE[])

# A bits value bound by value lives in a tiny device buffer of its own. Cached per value: the same
# scalars (and KA contexts) recur on every call.
const _ARG_BUFFERS = Dict{Any,MtlArray}()
arg_buffer(a) = get!(_ARG_BUFFERS, (typeof(a), a)) do
    MtlArray([a])
end

"""
    max_threads(path, entry) -> Int

The compiled pipeline's own threadgroup ceiling — what a kernel's register use leaves of the
device's. Dispatching over it returns garbage AND a fast time, so `launch_binary` refuses it.
"""
max_threads(path, entry) = Int(pipeline(path, entry).maxTotalThreadsPerThreadgroup)

function KO.launch_binary(::KO.MetalBackendTag, path, entry, args, grid::NTuple{3,Int}, tg::Integer)
    pipe = pipeline(path, entry)
    tg <= Int(pipe.maxTotalThreadsPerThreadgroup) || throw(ArgumentError(
        "threadgroup $tg exceeds this pipeline's limit of $(pipe.maxTotalThreadsPerThreadgroup); " *
        "dispatching over it returns garbage AND a fast time"))
    # Metal.jl batches its own commands (fills, copies, argument-buffer uploads) on ITS queue; this
    # launch commits to another. Without draining Metal.jl first the kernel can run before they land.
    Metal.synchronize()
    cb = Metal.MTLCommandBuffer(queue())
    enc = Metal.MTLComputeCommandEncoder(cb)
    Metal.set_function!(enc, pipe)
    for (i, a) in enumerate(args)
        if a isa MtlArray
            Metal.set_buffer!(enc, a.data[], a.offset, i)   # 1-based, despite Metal's 0-based convention
        else
            Metal.set_buffer!(enc, arg_buffer(a).data[], 0, i)
        end
    end
    Metal.dispatchThreadgroups!(enc, Metal.MTLSize(grid...), Metal.MTLSize(Int(tg), 1, 1))
    Metal.endEncoding!(enc)
    Metal.commit!(cb)
    Metal.wait_completed(cb)
    return nothing
end

KO.launch_binary(be::KO.MetalBackendTag, path, entry, args, ngroups::Integer, tg::Integer) =
    KO.launch_binary(be, path, entry, args, (Int(ngroups), 1, 1), tg)

# --- the eager binder -----------------------------------------------------------------------------

_alloc(o::KO.OutArray{T}) where {T} = o.zero ? Metal.zeros(T, o.dims...) : MtlArray{T}(undef, o.dims...)

"""
    bind_launch(::AbstractArray, path, entry, tg, prelude, args, extras, grid, label) -> Tuple

Allocate every `OutArray` of `args`, bind `(prelude..., args..., extras...)` in order (a `Val` takes no
slot), dispatch, and return the allocated outputs. Dispatches on `AbstractArray`: the Reactant extension's method, on
traced arrays, is more specific.
"""
function KO.bind_launch(::AbstractArray, path, entry, tg, prelude, args, extras, grid, label)
    bound = map(a -> a isa KO.OutArray ? _alloc(a) : a, args)
    slots = Any[prelude...]
    for a in (bound..., extras...)
        a isa Val || push!(slots, a)
    end
    if KO.recording()
        KO.record!(label, Any[(:bytes => p for p in prelude)...,
            (a isa KO.OutArray ? (:out => b) : b isa AbstractArray ? (:in => b) : (:scalar => b)
             for (a, b) in zip((args..., extras...), (bound..., extras...)) if !(b isa Val))...])
    else
        KO.launch_binary(KO.MetalBackendTag(), path, entry, slots, grid, tg)
    end
    return KO._pick_outs(args, bound)
end

# --- compiling -------------------------------------------------------------------------------------

"""
    compile_metallib(kernel, signature, tg, entry) -> Vector{UInt8}

Ahead-of-time compile the KernelAbstractions `kernel` (instantiated at workgroup size `tg`) for the
argument tuple type `signature`, as a `.metallib` whose entry point is `entry`.
"""
function compile_metallib(kernel, sig::Type, tg::Integer, entry::AbstractString)
    obj = kernel(MetalBackend(), Int(tg))
    job = GPUCompiler.CompilerJob(GPUCompiler.methodinstance(typeof(obj.f), sig),
        Metal.compiler_config(Metal.device(); name=String(entry), kernel=true))
    return Metal.compile(job).metallib
end

_ka_argtype(::Type{A}) where {A<:AbstractArray} = dptr(eltype(A))
_ka_argtype(::Type{T}) where {T} = T

function KO._ka_compile(::KO.MetalBackendTag, kernel, argtypes::Tuple, tg::Int, name::String,
        dir::String)
    mkpath(dir)
    path = joinpath(dir, name * ".metallib")
    isfile(path) && return path
    ctx, _ = KO.ka_context(KO.MetalBackendTag(), (1, 1, 1), tg)
    sig = Tuple{typeof(ctx),map(_ka_argtype, argtypes)...}
    write(path, compile_metallib(kernel, sig, tg, name))
    return path
end

# --- routing --------------------------------------------------------------------------------------

# Wrappers over device memory that Metal.jl leaves as a `SubArray` / `ReshapedArray` / `Adjoint` /
# `Transpose` rather than an `MtlArray` (it returns an `MtlArray` with an offset for a contiguous
# `view` or a `reshape`).
const WrappedMtlArray = Union{
    SubArray{<:Any,<:Any,<:MtlArray},
    Base.ReshapedArray{<:Any,<:Any,<:MtlArray},
    LinearAlgebra.Adjoint{<:Any,<:MtlArray},
    LinearAlgebra.Transpose{<:Any,<:MtlArray},
}

KO.backend_of(::MtlArray) = KO.MetalBackendTag()
KO.backend_of(::WrappedMtlArray) = KO.MetalBackendTag()

# The launcher binds a raw buffer (plus offset) and no strides, so a wrapper that is not an `MtlArray`
# (a strided `view`, an adjoint) is copied once, on the device. Anything else is left alone: a host array among
# the arguments should fail at binding, not be copied into another host array.
KO.device_dense(::KO.MetalBackendTag, x::MtlArray) = x
KO.device_dense(::KO.MetalBackendTag, x::WrappedMtlArray) = copy(x)

KO.ka_backend_object(::KO.MetalBackendTag) = MetalBackend()

# Metal.jl's KA kernels take a kernel-state word (a `UInt32`) ahead of KA's context.
KO.ka_state_words(::KO.MetalBackendTag) = (UInt32(0),)

# KA's Metal backend reads only the x component of the threadgroup position and rebuilds the N-D
# indices from the context, so the binary is dispatched as a flat 1-D grid.
KO.ka_grid(::KO.MetalBackendTag, groups::NTuple{3,Int}) = (prod(groups), 1, 1)

const _AVAILABLE = Ref{Union{Nothing,Bool}}(nothing)
function available()
    v = _AVAILABLE[]
    isnothing(v) || return v
    ok = try
        Metal.functional() && !isnothing(Metal.device())
    catch
        false
    end
    _AVAILABLE[] = ok
    return ok
end

function __init__()
    available() && push!(KO.LOADED_BACKENDS, :metal)
    return nothing
end

end # module
