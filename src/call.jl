"""
    OutArray(T, dims...; zero=false)

An output argument: a fresh `T` array of size `dims`, allocated eagerly or declared as a result in a
trace, in the argument's place. `zero = true` zero-fills it first, for a kernel that accumulates.
"""
struct OutArray{T,N}
    dims::NTuple{N,Int}
    zero::Bool
end
# `realtype`: a traced element type (`TracedRNumber{Float32}`) names its storage type.
OutArray(::Type{T}, dims::Integer...; zero::Bool=false) where {T} =
    OutArray{realtype(T),length(dims)}(Int.(dims), zero)
OutArray(::Type{T}, dims::Tuple; zero::Bool=false) where {T} = OutArray(T, dims...; zero)
Base.eltype(::OutArray{T}) where {T} = T

function _first_array(xs)   
    i = findfirst(Base.Fix2(isa, AbstractArray), xs);
    return isnothing(i) ? nothing : xs[i]
end

"""
    (op::AbstractKernelOp)(args...)

Run `op`: [`host`](@ref) on host arrays (and, with a warning, on a device no kernel exists for);
[`forward`](@ref) with the selected kernel on device arrays. On a device there is no silent fallback:
a missing kernel is an error, not a 10–100× slowdown behind a correct answer.
"""
function (op::AbstractKernelOp)(args...)
    a = _first_array(args)
    isnothing(a) && throw(ArgumentError("op `:$(opname(op))` was called without any array argument"))
    be = backend_of(a)
    be isa CPUBackend && return host(op, args...)
    if be isa UnknownBackend
        @warn "Unrecognised device backend for op `:$(opname(op))`; falling back to its host path." maxlog = 1
        return host(op, args...)
    end

    check_traced_autodiff(a, "op `:$(opname(op))`")

    return forward(op, args...)
end

"""
    call_binary(op, kernel, name, args...; key=variant_key(op, kernel, args...)) -> Tuple

Launch binary `name` of the variant for `key`, binding `(args..., extras(op, kernel, binary)...)`
IN THE ORDER GIVEN: an [`OutArray`](@ref) is allocated (or a traced result), an array is bound in place,
a number by value, a `Val` not at all. The grid is [`grid`](@ref)`(op, kernel, binary, args...)`.
Returns the outputs, in `OutArray` order. Getting the order right is the caller's job.

The outputs are picked from `args` alone: `extras` (which may read `KernelBinary.params`, typed only
at run time) are only bound, so whatever they return never reaches the result's type. They must
therefore not contain `OutArray`s.
"""
function call_binary(op::AbstractKernelOp, k::AbstractKernel, name::Symbol, args...; key::Tuple=variant_key(op, k, args...))
    v = variant(op, k, key)
    b = get(v, name) do
        throw(
            ArgumentError(
                "variant $key of kernel `:$(kernelname(k))` has no binary `:$name`; " *
                "it has $(sort(collect(keys(v))))"
            )
        )
    end
    g = NTuple{3,Int}(grid(op, k, b, args...))
    proto = _first_array(args)
    be = backend_of(proto)
    prelude = Any[]
    if b.ka
        # A KA-compiled binary: KA's context (built by KA from this grid) ahead of the arguments, and the
        # backend's own dispatch grid.
        ctx, ghost = ka_context(be, g, b.threadgroup)
        prelude = ka_prelude(be, ctx, ghost)
        g = ka_grid(be, g)
    end
    _make_dense = Base.Fix1(device_dense, be)

    return bind_launch(proto, b.file, b.entry, b.threadgroup, prelude, map(_make_dense, args),
        map(_make_dense, extras(op, k, b)), g, name)
end

"""
    bind_launch(proto, path, entry, threadgroup, prelude, args, extras, grid, label) -> Tuple

Bind `(prelude..., args..., extras...)` to `entry` in the binary at `path` and run it on a 3-D `grid`
of threadgroups, returning the outputs in `OutArray` order. The outputs are picked from `args` only:
`extras` are bound and nothing else, so their types cannot reach the result. They are the kernel's
[`extras`](@ref) from [`call_binary`](@ref), and `()` for a kernel whose [`extras`](@ref) is empty. 
Implemented by backend extensions, on the array kind of `proto`: eager device arrays (launched) or 
traced arrays (a custom call).
"""
function bind_launch end

# --- inspecting what would be bound -----------------------------------------------------------------

const _RECORD = Ref{Union{Nothing,Vector{Any}}}(nothing)

"""
    record_bindings(f) -> Vector{(label, args)}

Run `f()` with every eager launch replaced by a record of what it would bind: the binary's name and
its slots in order, each `kind => value` with `kind` one of `:bytes` (the KA prelude), `:out` (an
allocated output), `:in` (an array), `:scalar`. Nothing is dispatched.
"""
function record_bindings(f)
    prev = _RECORD[]
    _RECORD[] = Any[]
    try
        f()
        return _RECORD[]
    finally
        _RECORD[] = prev
    end
end

# Called by an eager binder in place of launching while recording.
recording() = _RECORD[] !== nothing
record!(label, args) = push!(_RECORD[], (label, Tuple(args)))

# The allocated outputs, picked out of the bound arguments in `OutArray` order. Recursive over the tuple,
# so the result type is inferred.
_pick_outs(::Tuple{}, ::Tuple{}) = ()
_pick_outs(a::Tuple, b::Tuple) = first(a) isa OutArray ?
    (first(b), _pick_outs(Base.tail(a), Base.tail(b))...) : _pick_outs(Base.tail(a), Base.tail(b))
