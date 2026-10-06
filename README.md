# KernelOps.jl

Run an operation as compiled kernels:
- eagerly on device arrays (`MtlArray`);
- **embedded in a Reactant-compiled program** as a `stablehlo.custom_call`;
- or as plain Julia on the host.

Each operation can have several **kernels** (implementations) that you switch between at runtime,
and each kernel has **tuned variants** picked from the inputs.

KernelOps does not interpret your arguments. You hand a binary its arguments in the order it takes
them, with outputs marked `OutArray(T, dims...)`, and get the outputs back. Everything else (reshapes,
transposes, scalars) is ordinary Julia in your `forward` method.

Backends today: Metal, eager, and traced through the jax-mps PJRT plugin.
[LuxTriangleAttention.jl](../LuxTriangleAttention.jl) runs all its attention kernels this way: a
KernelAbstractions flash kernel, a matrix-unit variant, and trifast's Triton metallibs.

## Concepts

| | what | you write |
|:--|:--|:--|
| **op** | an operation: `struct Axpy <: AbstractKernelOp end` | `opname`, `host`, and usually `forward`/`backward` methods |
| **kernel** | one implementation of it, switchable by name: `struct AxpyKA <: AbstractKernel end` | `kernelname`, `grid`, and optionally `variant_key`, `extras`, `nearest`, `build!`, `source_tag` |
| **variant** | one tuned build of a kernel for a class of inputs: launch name → `KernelBinary` | `add_variant!` (persisted to a short TOML manifest) |
| **`KernelBinary`** | one compiled launch: file, entry point, threadgroup, tuning `params` | — |

Ops and kernels are fieldless singletons, so their behaviour comes from dispatch. Which kernels an
op has and which variants exist are kept in KernelOps' tables, by name. Which kernel is *selected* is
a method, so an op call is type-stable (see [Selecting kernels](#selecting-kernels)).

## Example

```julia
using KernelOps, KernelAbstractions, Metal
import KernelOps: opname, kernelname, host, forward, backward, grid, variant_key, OutArray

@kernel unsafe_indices = true function axpy_fwd!(z, x, y, a::Float32, n::Int32, ::Val{TG}) where {TG}
    i = (@index(Group, Linear) - 1) * TG + @index(Local, Linear)
    i <= n && unsafe_store!(z, a * unsafe_load(x, i) + unsafe_load(y, i), i)
end

struct Axpy <: AbstractKernelOp end
struct AxpyKA <: AbstractKernel end
opname(::Axpy) = :axpy
kernelname(::AxpyKA) = :ka
host(::Axpy, x, y, a) = a .* x .+ y                    # host arrays
variant_key(::Axpy, ::AxpyKA, args...) = (nameof(eltype(KernelOps._first_array(args))),)
grid(::Axpy, ::AxpyKA, b::KernelBinary, args...) =           # threadgroups; `params` hold tuning numbers
    (cld(length(KernelOps._first_array(args)), b.params.tg), 1, 1)

# The binary's arguments, in its order; outputs come back in `OutArray` order.
forward(op::Axpy, k::AxpyKA, x, y, a) =
    only(call_binary(op, k, :fwd, OutArray(eltype(x), length(x)), x, y, Float32(a), Int32(length(x))))

fwd = ka_compile(KernelOps.MetalBackendTag(), axpy_fwd!,       # a KernelBinary with `ka=true`
    (Vector{Float32}, Vector{Float32}, Vector{Float32}, Float32, Int32, Val{64});
    tg=64, name="axpy_fwd", params=(; tg=64))
KernelOps.@default_kernel Axpy AxpyKA()               # the default (more candidates may follow)
register_kernel!(Axpy(), AxpyKA())                     # in a package: in `__init__`
add_variant!(Axpy(), AxpyKA(), (:Float32,); fwd)

Axpy()(x, y, 2f0)       # Array: host · MtlArray: the kernel · traced (jax-mps): a custom call
```

Variants are persisted (the binaries are copied into the cache), so they survive a restart.

## Selecting kernels

The kernel an op runs is a method, not a table entry: a constant to the compiler, so
`Axpy()(x, y, a)` and every function calling it infer. It comes in two layers:

| | set with | takes effect | cost |
|:--|:--|:--|:--|
| **default** | `@default_kernel Op k₁ k₂ …` in the op's package, `set_default_kernel!(op, name)` | next session | the package and its dependents re-precompile once |
| **session override** | `use_kernel!(op, name)`, undone by `reset_kernel!(op)` | next top-level statement | code calling the op recompiles on its next call |

Tuned variants are plain values: tuning never recompiles anything. A typical session:

```julia
using BioFold                            # precompiled with the default kernel
use_kernel!(:attention, :flash)          # try another kernel, this session only
tune!(...)                               # variants go to the manifest
model(x)                                 # runs :flash, inferred
set_default_kernel!(:attention, :flash)  # persist: next session precompiles with :flash
```

- **The default** is `k₁`, unless the preference `"kernel.<opname>"` of the package declaring the op
  names another candidate. `set_default_kernel!` writes that preference to the active project's
  `LocalPreferences.toml`; `clear_default_kernel!` removes it. Outside a package (a script) there
  are no preferences and the default is `k₁`.
- **World age.** `use_kernel!` redefines a method, which Julia makes visible from the next top-level
  expression (REPL input, script statement). Switching and calling the op later in the *same*
  function or `begin`/`let` block throws rather than silently run the previous kernel. To switch
  inside a function (a benchmark over kernels), use `with_kernel(() -> model(x), :attention, :flash)`.
- **Precompilation.** Kernel tables are filled at run time: register kernels in your package's
  `__init__`. `use_kernel!` refuses to run while precompiling; the default is what precompiles.
- **Reactant.** A compiled thunk (`@compile`) holds the kernel selected when it was traced;
  re-`@compile` after switching.

## Calling and overloading

```julia
(op::AbstractKernelOp)(args...)        # host arrays → host(op, args...); device → forward(op, args...)
forward(op, args...)                   # = forward(op, selected kernel, args...)
forward(op, kernel, args...)           # default: call_binary(op, kernel, :fwd, args...); overload it
backward(op, kernel, args, outs, cots) # gradients aligned with args; overload to differentiate
call_binary(op, kernel, name, args...; key=variant_key(op, kernel, args...))
```

`call_binary` finds the variant for `key`: an exact match, else your `nearest`, else your `build!`
(a lazy tune). It then binds `(args..., extras(op, kernel, binary)...)` in the order given:
- an `OutArray` is allocated eagerly, or becomes a result in a trace;
- an array is bound in place;
- a number is bound by value;
- a `Val` takes no slot.

It launches on `grid(op, kernel, binary, args...)` threadgroups and returns the outputs. For an
binary compiled with `ka_compile` (`KernelBinary.ka`), KA's launch context is built for you and the launch is dispatched flat. The result
type is inferred from the `OutArray`s alone, whatever your `grid` and `extras` read from `params`.

## AD

- **Eager Enzyme:** one rule covers every op. It fires on the device call and runs your `backward`,
  so Enzyme never enters your `forward`'s plain Julia or the binary.
- **Under Reactant:** call `backward` directly. A gradient traced *through* a kernel throws, because
  Enzyme-JAX cannot differentiate a custom call yet (EnzymeAD/Enzyme#2516). Set
  `KernelOps.KERNEL_IN_TRACED_AUTODIFF[] = true` once it can.

## Also

- `ka_compile(backend, kernel, argtypes; tg, name)`: compile a KernelAbstractions kernel to a binary
  (built in `ka_binary_cache_dir()`, `<cache_root>/ka`, and found there next time), then register it with `add_variant!` like any other binary (see the
  example). It is the KA producer of binaries, the counterpart of a Triton compile script: it builds
  against the launch context that `ka` binaries are bound with, so compile KA kernels with it. It returns a `KernelBinary` with `ka=true`, which tells `call_binary` to bind that context; a Triton binary leaves it `false`. A KA
  kernel is then just another source of binaries: an op keeps one kernel type per implementation
  (`AxpyKA`, `AxpyTriton`, …), each registering its own binaries, and switching between them never
  changes the op. Registered binaries live in the registry (`cache_dir(op)/<kernelname>/`), copied from
  wherever they were built.
- **Not for tracing KA kernels.** The binary is compiled ahead of time, and under Reactant the program
  holds a `stablehlo.custom_call` that only *names* the file, so XLA never sees the kernel's code. To
  have Reactant trace a KA kernel itself (fusable, differentiable), run Reactant over the `@kernel`
  function directly and leave this package out of it.
- `KernelOps.record_bindings(f)`: runs `f` with every launch replaced by a record of exactly what
  would be bound. Useful for pinning a binary's argument list in a test, as LuxTriangleAttention's
  golden-ABI test does.
