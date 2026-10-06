"""
    AbstractKernelOp

An operation that runs as compiled kernels. Subtype it with a singleton and give it a name:

    struct Axpy <: AbstractKernelOp end
    KernelOps.opname(::Axpy) = :axpy

Calling it (`Axpy()(args...)`) runs [`host`](@ref) on host arrays and [`forward`](@ref) on device
arrays, with its [`default_kernel`](@ref) or the one selected by [`use_kernel!`](@ref).
"""
abstract type AbstractKernelOp end

"""
    AbstractKernel

One implementation of an op, switchable by name with [`use_kernel!`](@ref):

    struct AxpyKA <: AbstractKernel end
    KernelOps.kernelname(::AxpyKA) = :ka

Its methods ([`variant_key`](@ref), [`grid`](@ref), [`extras`](@ref),
[`nearest`](@ref), [`build!`](@ref), [`source_tag`](@ref)) say how its binaries are launched.
"""
abstract type AbstractKernel end

"""
    KernelBinary(file; entry=basename without extension, threadgroup, params=(;), ka=false)

One compiled launch. `params` are the tuning numbers it was built with — what [`grid`](@ref) and
[`extras`](@ref) read — and are persisted with it.

`ka` says the binary was compiled from a KernelAbstractions kernel ([`ka_compile`](@ref) sets it), so
it takes the backend's [`ka_state_words`](@ref) and KA's launch context ahead of its arguments, and is
dispatched flat (design §5). It is a property of the binary, not of the kernel type that holds it: one
kernel can register a KA-compiled binary for one launch and a Triton one for another. Persisted too.
"""
struct KernelBinary
    file::String
    entry::String
    threadgroup::Int
    params::NamedTuple
    ka::Bool
    # Fields of `params` kept sorted, so a binary compares equal to itself after a manifest round-trip.
    KernelBinary(file, entry, threadgroup, params::NamedTuple, ka::Bool) =
        new(String(file), String(entry), Int(threadgroup), _sorted(params), ka)
end

KernelBinary(file::AbstractString;
    entry::AbstractString=splitext(basename(file))[1],
    threadgroup::Integer,
    params::NamedTuple=NamedTuple(),
    ka::Bool=false) = KernelBinary(file, entry, threadgroup, params, ka)

_sorted(nt::NamedTuple) = NamedTuple{Tuple(sort(collect(keys(nt))))}(nt)

# --- the interface ---------------------------------------------------------------------------------

"""`opname(op) -> Symbol`: the op's name, what [`use_kernel!`](@ref) takes."""
function opname end

"""`kernelname(kernel) -> Symbol`: the kernel's name within its op."""
function kernelname end

"""
    host(op, args...)

The op on host arrays. Runs whatever kernel is selected: the selection applies on a device.
"""
host(op::AbstractKernelOp, args...) =
    throw(ArgumentError("op `:$(opname(op))` has no host implementation; move the arguments to a device"))

"""
    forward(op, args...)
    forward(op, kernel, args...)

The op on device arrays. The first form runs the selected kernel. The second is what to overload
per op and kernel: its default passes `args` straight to the kernel's `:fwd` binary
([`call_binary`](@ref)); an overload can reshape and reorder in plain Julia first.
"""
forward(op::AbstractKernelOp, args...) = forward(op, selected_kernel(op), args...)
forward(op::AbstractKernelOp, k::AbstractKernel, args...) = call_binary(op, k, :fwd, args...)

"""
    backward(op, kernel, args, outs, cotangents) -> Tuple

Gradients of `forward(op, kernel, args...)`, one per argument (`nothing` for one with none), given
its outputs `outs` and their `cotangents`. No default: overload it to make an op differentiable.
"""
backward(op::AbstractKernelOp, k::AbstractKernel, args, outs, cots) = throw(ArgumentError(
    "op `:$(opname(op))` kernel `:$(kernelname(k))` has no `backward`; overload " *
    "`KernelOps.backward` to differentiate it"))

"""`variant_key(op, kernel, args...) -> Tuple`: which tuned variant serves these arguments. Default `()`."""
variant_key(::AbstractKernelOp, ::AbstractKernel, args...) = ()

"""`grid(op, kernel, binary, args...) -> NTuple{3,Int}`: threadgroups to dispatch for these arguments."""
grid(op::AbstractKernelOp, k::AbstractKernel, b::KernelBinary, args...) = throw(ArgumentError(
    "kernel `:$(kernelname(k))` of op `:$(opname(op))` defines no `KernelOps.grid`"))

"""`extras(op, kernel, binary) -> Tuple`: arguments appended to the caller's (tuning values). Default `()`."""
extras(::AbstractKernelOp, ::AbstractKernel, ::KernelBinary) = ()

"""`nearest(op, kernel, keys, key)`: a registered key to use when `key` has no exact variant, or `nothing`."""
nearest(::AbstractKernelOp, ::AbstractKernel, keys, key) = nothing

"""`build!(op, kernel, key)`: create the variant for `key` (a lazy tune) via [`add_variant!`](@ref). Default: nothing."""
build!(::AbstractKernelOp, ::AbstractKernel, key) = nothing

"""`source_tag(kernel) -> String`: a manifest recorded with another tag is ignored. Default `""`."""
source_tag(::AbstractKernel) = ""

"""`cache_dir(op) -> String`: where the op's kernels keep their manifests and binaries."""
cache_dir(op::AbstractKernelOp) = joinpath(cache_root(), "ops", String(opname(op)))

# --- tables ----------------------------------------------------------------------------------------

const Variant = Dict{Symbol,KernelBinary}
const OPS = Dict{Symbol,AbstractKernelOp}()
const KERNELS = Dict{Symbol,Dict{Symbol,AbstractKernel}}()
const VARIANTS = Dict{Tuple{Symbol,Symbol},Dict{Tuple,Variant}}()

"""`variants(op, kernel) -> Dict{Tuple,Variant}`: the kernel's registered variants, by key."""
variants(op::AbstractKernelOp, k::AbstractKernel) =
    get!(() -> Dict{Tuple,Variant}(), VARIANTS, (opname(op), kernelname(k)))

"""
    register_kernel!(op, kernel; select=false) -> kernel

Make `kernel` available to `op` (and `op` known by its name), loading the kernel's persisted
variants. `select = true` also selects it ([`use_kernel!`](@ref)).

These tables are filled at run time: a package declaring ops registers its kernels in `__init__`
(registrations made while it precompiles are not kept).
"""
function register_kernel!(op::AbstractKernelOp, k::AbstractKernel; select::Bool=false)
    on, kn = opname(op), kernelname(k)
    OPS[on] = op
    get!(() -> Dict{Symbol,AbstractKernel}(), KERNELS, on)[kn] = k
    merge!(variants(op, k), load_manifest(op, k))
    select && use_kernel!(op, kn)
    return k
end

_op(op::AbstractKernelOp) = op
_op(::Type{O}) where {O<:AbstractKernelOp} = get(OPS, opname(O()), O())
_op(name::Symbol) = get(OPS, name) do
    throw(ArgumentError("no op named `:$name`. Ops: " * _names(keys(OPS))))
end
_names(xs) = isempty(xs) ? "(none)" : join(sort([":$x" for x in xs]), ", ")

# --- selection -------------------------------------------------------------------------------------
#
# Which kernel an op runs is a METHOD, not a table entry, so it is a compile-time constant and an op
# call infers. Two layers: the op's default (`default_kernel`, defined by `@default_kernel` in the op's
# own package from a compile-time preference, so it is baked into precompile caches) and a session
# override (`_override`, redefined by `use_kernel!`, which invalidates and recompiles the callers).
# A redefinition is visible from the next top-level expression on: see `throw_stale_switch`.

"""
    default_kernel(op) -> kernel or nothing

The kernel `op` runs unless [`use_kernel!`](@ref) overrides it. Defined by [`@default_kernel`](@ref)
in the op's package; `nothing` (no default) otherwise.
"""
default_kernel(::AbstractKernelOp) = nothing

# The session override per op type, redefined by `use_kernel!`. Owned by KernelOps, so a switch
# never overwrites a method of the op's package.
_override(::AbstractKernelOp) = nothing

# Bumped on every switch; `_switch_epoch()` is redefined to match. They disagree exactly when the code
# running was compiled in a world from before the latest switch.
const SWITCH_EPOCH = Ref(0)
_switch_epoch() = 0

_generating_output() = ccall(:jl_generating_output, Cint, ()) == 1

function _set_override!(o::AbstractKernelOp, k::Union{Nothing,AbstractKernel})
    Base.invokelatest(_override, o) === k && return nothing      # unchanged: invalidate nothing
    _generating_output() && throw(ArgumentError(
        "kernels cannot be switched while precompiling; set the default of op `:$(opname(o))` with " *
        "`KernelOps.@default_kernel` (and its preference with `set_default_kernel!`) instead"))
    n = SWITCH_EPOCH[] + 1
    @eval begin
        _override(::$(typeof(o))) = $k
        _switch_epoch() = $n
    end
    SWITCH_EPOCH[] = n
    return nothing
end

"""
    use_kernel!(op, kernel_name) -> Symbol

Select which kernel `op` runs for the rest of the session, overriding its [`default_kernel`](@ref).
`op` is the op (`Axpy()`), its type (`Axpy`) or its name (`:axpy`). Not persisted: see
[`set_default_kernel!`](@ref).

The selection is a method, so op calls stay inferred; code calling the op recompiles on its next call.
It takes effect from the NEXT top-level expression (REPL input, script statement): later in the same
function or `begin`/`let` block the op call throws, rather than run the previous kernel. Use
[`with_kernel`](@ref) to switch and run in one go. (A `@testset` body sees it at once.)
"""
function use_kernel!(op, kname::Symbol)
    o = _op(op)
    _set_override!(o, _kernel(o, kname))
    return kname
end

function _kernel(o::AbstractKernelOp, kname::Symbol)
    ks = get(KERNELS, opname(o), Dict{Symbol,AbstractKernel}())
    haskey(ks, kname) || throw(ArgumentError(
        "op `:$(opname(o))` has no kernel `:$kname`. Kernels: " * _names(keys(ks))))
    return ks[kname]
end

"""`reset_kernel!(op)`: drop the [`use_kernel!`](@ref) override; `op` runs its default again."""
reset_kernel!(op) = (_set_override!(_op(op), nothing); nothing)

"""
    with_kernel(f, op, kernel_name)

Run `f()` with `op` switched to `kernel_name` (in the newest world, so the switch is visible), then
restore the previous selection. For comparing kernels from inside one function.
"""
function with_kernel(f, op, kname::Symbol)
    o = _op(op)
    prev = Base.invokelatest(_override, o)
    _set_override!(o, _kernel(o, kname))
    try
        return Base.invokelatest(f)
    finally
        _set_override!(o, prev)
    end
end

"""`current_kernel(op) -> Symbol or nothing`: the kernel `op` runs (as of the newest world)."""
function current_kernel(op)
    k = Base.invokelatest(_selected, _op(op))
    return k === nothing ? nothing : kernelname(k)
end

"""`list_kernels(op)`: every kernel of `op`, its variant count, and which is selected."""
function list_kernels(op)
    o = _op(op)
    sel = current_kernel(o)
    return [(; name=kn, nvariants=length(variants(o, k)), selected=(kn === sel))
            for (kn, k) in sort(collect(get(KERNELS, opname(o), Dict{Symbol,AbstractKernel}())); by=first)]
end

_selected(op::AbstractKernelOp) = (k = _override(op); k === nothing ? default_kernel(op) : k)

function selected_kernel(op::AbstractKernelOp)
    _switch_epoch() == SWITCH_EPOCH[] || throw_stale_switch(op)
    k = _selected(op)
    k === nothing && throw_no_kernel(op)
    return k
end

@noinline throw_no_kernel(op) = throw(ArgumentError(
    "no kernel selected for op `:$(opname(op))`; call `use_kernel!(:$(opname(op)), name)`, or give " *
    "the op a default with `KernelOps.@default_kernel`. Kernels: " *
    _names(keys(get(KERNELS, opname(op), Dict())))))

@noinline throw_stale_switch(op) = throw(ArgumentError(
    "op `:$(opname(op))` was called from code compiled before the latest kernel switch: `use_kernel!` " *
    "(or `reset_kernel!`) takes effect from the next top-level expression, not later in the same " *
    "function or block. Switch at top level, or use `with_kernel(f, op, name)`."))

"""
    variant(op, kernel, key) -> Variant

The variant serving `key`: an exact match, else the kernel's [`nearest`](@ref) choice, else what
its [`build!`](@ref) creates. Throws when none exists.
"""
function variant(op::AbstractKernelOp, k::AbstractKernel, key::Tuple)
    vs = variants(op, k)
    v = _lookup(op, k, vs, key)
    v === nothing || return v
    build!(op, k, key)
    v = _lookup(op, k, vs, key)
    v === nothing || return v
    throw(ArgumentError("kernel `:$(kernelname(k))` of op `:$(opname(op))` has no variant for key " *
        "$key. Registered: $(sort(collect(keys(vs)); by=string))"))
end

function _lookup(op, k, vs, key)
    haskey(vs, key) && return vs[key]
    isempty(vs) && return nothing
    nk = nearest(op, k, keys(vs), key)
    return nk === nothing ? nothing : vs[nk]
end

# --- manifest: one TOML per (op, kernel) -------------------------------------------------------------

kernel_dir(op::AbstractKernelOp, k::AbstractKernel) = joinpath(cache_dir(op), String(kernelname(k)))
manifest_path(op, k) = joinpath(kernel_dir(op, k), "manifest.toml")

# Key elements are Symbols, integers or Bools; `repr` round-trips them through `Meta.parse`.
_key_str(key::Tuple) = [repr(x) for x in key]
_key_parse(strs) = Tuple(begin
    x = Meta.parse(s)
    x isa QuoteNode ? x.value : x
end for s in strs)

"""
    add_variant!(op, kernel, key; binaries...) -> Variant

Register the variant for `key`: one [`KernelBinary`](@ref) per launch name (`fwd=…, dq=…`). Each binary is
COPIED into the kernel's cache directory (so it survives its original moving) and the manifest is
rewritten.
"""
function add_variant!(op::AbstractKernelOp, k::AbstractKernel, key::Tuple; binaries...)
    isempty(binaries) && throw(ArgumentError("a variant needs at least one binary"))
    dir = kernel_dir(op, k)
    mkpath(dir)
    v = Variant()
    for (name, b) in binaries
        isfile(b.file) || throw(ArgumentError("binary `$(b.file)` does not exist"))
        id = first(string(hash(read(b.file), hash((key, name))); base=16, pad=16), 8)
        dst = joinpath(dir, "$(name)_$(id)$(lowercase(splitext(b.file)[2]))")
        abspath(b.file) == abspath(dst) || cp(b.file, dst; force=true)
        v[name] = KernelBinary(dst, b.entry, b.threadgroup, b.params, b.ka)
    end
    variants(op, k)[key] = v
    save_manifest(op, k)
    return v
end

function save_manifest(op, k)
    vs = variants(op, k)
    tbl = Dict{String,Any}("source_tag" => source_tag(k), "variant" => [
        Dict{String,Any}("key" => _key_str(key), "binary" => [
            Dict{String,Any}("name" => String(n), "file" => basename(b.file), "entry" => b.entry,
                "threadgroup" => b.threadgroup, "ka" => b.ka, "params" => Dict(String(p) => x for (p, x) in pairs(b.params)))
            for (n, b) in sort(collect(v); by=first)])
        for (key, v) in sort(collect(vs); by=string ∘ first)])
    open(io -> TOML.print(io, tbl), manifest_path(op, k), "w")
    # Binaries no variant names any more (a re-registered variant's predecessor) are removed.
    keep = Set(basename(b.file) for v in values(vs) for b in values(v))
    for f in readdir(kernel_dir(op, k))
        (f == "manifest.toml" || f in keep) || rm(joinpath(kernel_dir(op, k), f); force=true)
    end
end

# Never throws: a missing, corrupt or stale manifest is simply no variants — a kernel that can build
# its own (`build!`) rebuilds, any other is registered again.
function load_manifest(op, k)
    out = Dict{Tuple,Variant}()
    path = manifest_path(op, k)
    isfile(path) || return out
    try
        raw = TOML.parsefile(path)
        get(raw, "source_tag", "") == source_tag(k) || return out
        dir = dirname(path)
        for vd in get(raw, "variant", Any[])
            v = Variant()
            for bd in vd["binary"]
                file = joinpath(dir, bd["file"])
                isfile(file) || (v = nothing; break)
                v[Symbol(bd["name"])] = KernelBinary(file, bd["entry"], bd["threadgroup"],
                    NamedTuple(Symbol(p) => x for (p, x) in bd["params"]), get(bd, "ka", false))
            end
            v === nothing || (out[_key_parse(vd["key"])] = v)
        end
    catch
    end
    return out
end
