# KernelDispatch.jl

Simplify working with compiled kernel binaries built anywhere, directly in Julia!
- Define an op
- Register kernels for it
- Select default kernel and run the op on CPU and device arrays simply through multiple dispatch!
- Kernels can even run embedded in a Reactant-compiled program as a stablehlo.custom_call!

An operation can have several **kernels** (implementations) you switch between at runtime, each with
**tuned variants** picked from the inputs. Pure-Julia [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl) 
kernels are also supported as one more source of binaries: `ka_compile` turns one into a binary, and 
from there it is launched like any other. 

**Tutorials** in [`docs/`](docs/README.md) walk through a complete example: switching kernels
and persisting defaults, registering a compiled binary or a KA kernel, custom `forward`/`backward` with
Enzyme gradients, and compiling a Triton kernel to a `.metallib`.

## Concepts

| | what | you write |
|:--|:--|:--|
| **op** | an operation: `SDPA <: AbstractKernelOp` | `host`, `@default_kernel` |
| **kernel** | one implementation: `Trifast <: AbstractKernel` | `forward` (builds the binary's arguments), optionally `backward`, `variant_key`, `extras`, `nearest`, `build!`, `source_tag` |
| **variant** | one (tuned) build of a kernel for a class of inputs: launch name → `KernelBinary` | `add_variant!` |
| **`KernelBinary`** | one compiled launch: file, entry point, `threadgroup`, `tile`, tuning `params`, `is_ka` | — |

Ops and kernels are fieldless singletons; behaviour comes from dispatch. Which kernel is *selected* is
a method, so an op call is type-stable.

## Example

Registering a single kernel for a scaled dot-product attention operation (`q: d×n`, `k: d×m`, `v: dv×m` → `o: dv×n`):

```julia
import KernelDispatch: host, forward

using KernelDispatch

# Define your operation
struct SDPA <: AbstractKernelOp end

# CPU default:
softmax(s) = (p = exp.(s .- maximum(s; dims=1)); p ./ sum(p; dims=1))
host(::SDPA, q, k, v) = v * softmax((k' * q) ./ sqrt(Float32(size(q, 1))))

# Define our kernel:
struct FlashAttn <: AbstractKernel end

# For package developers
KernelDispatch.@default_kernel SDPA FlashAttn()

add_variant!(
  SDPA(), FlashAttn(); fwd = KernelBinary("flash.metallib"; threadgroup=64)
)

# Every kernel needs a `forward`: it builds the binary's arguments from the op's and launches it.
function forward(op::SDPA, kernel::FlashAttn, q, k, v)
    (d, n), m, dv = Int32.(size(q)), Int32(size(k, 2)), Int32(size(v, 1))
    o = OutArray(Float32, dv, n) # Provide the shapes of the output
    scale = inv(sqrt(Float32(d)))
    # `extent`: the problem size (one query each); KernelDispatch divides it by the binary's `tile`.
    # :fwd should match the kwarg that is used in add_variant!
    result = call_binary(op, kernel, :fwd, q, k, v, o, n, m, d, dv, scale; extent=n) # Returns a Tuple
    return only(result)                                       # of the OutArrays: here, just `o`
end
```

Calling the op picks the path from its arguments, and switching kernels is one line:

```julia
q, k, v = (randn(Float32, 16, 100), randn(Float32, 16, 70), randn(Float32, 8, 70)) .|> MtlArray

o = SDPA()(q, k, v)                                       # MtlArray: the default kernel, FlashAttn
SDPA()(Array(q), Array(k), Array(v))                      # Array: `host` · traced (jax-mps): a custom call
use_kernel!(SDPA(), SDPAKA())                             # this session, from the next top-level statement
SDPA()(q, k, v)                                           # runs the KA kernel (defined below)
forward(SDPA(), FlashAttn(), q, k, v)                     # one call with a given kernel: no global switch
```

## KernelAbstractions kernels

A KernelAbstractions kernel is just one more source of binaries. Written against the same argument list,
it joins the op as another kernel: its own `forward`, and a binary from `ka_compile` instead of a file:

```julia
using KernelAbstractions
import KernelDispatch: extras

@kernel unsafe_indices = true function sdpa_ka!(q, k, v, o, n::Int32, m::Int32, d::Int32, dv::Int32,
        scale::Float32, ::Val{TG}) where {TG}
    i = (@index(Group, Linear) - 1) * TG + @index(Local, Linear)   # this thread's query
    if i <= n
        # … softmax(k[:, j]' q[:, i] * scale) over j, weighting v into o[:, i], through
        #   unsafe_load / unsafe_store! (see examples/SDPAOps for a full kernel)
    end
end

struct SDPAKA <: AbstractKernel end
extras(::SDPA, ::SDPAKA, b::KernelBinary) = (Val(b.threadgroup),)  # the compiled-in tile size: no slot

function forward(op::SDPA, kernel::SDPAKA, q, k, v)               # same argument order as FlashAttn's
    (d, n), m, dv = Int32.(size(q)), Int32(size(k, 2)), Int32(size(v, 1))
    o = OutArray(Float32, dv, n)
    result = call_binary(op, kernel, :fwd, q, k, v, o, n, m, d, dv, inv(sqrt(Float32(d))); extent=n)
    return only(result)
end

argtypes = (
  Vector{Float32}, Vector{Float32}, Vector{Float32}, Vector{Float32}, 
  Int32, Int32, Int32, Int32, Float32, Val{64}
)

fwd = ka_compile(KernelDispatch.MetalBackendTag(), sdpa_ka!, argtypes; tg=64, name="sdpa_ka_v1")
add_variant!(SDPA(), SDPAKA(); fwd)
use_kernel!(SDPA(), SDPAKA())
```

- `ka_compile(backend, kernel, argtypes; tg, name, params, tile)` returns a `KernelBinary` with
  `is_ka=true` and, unless given, `tile=(tg, 1, 1)`: one item per thread, as `sdpa_ka!` indexes. It
  builds into `ka_binary_cache_dir()` (`<cache_root>/ka`) and finds it there next time, so put a
  source hash in `name`. Its `argtypes` are the arguments after KA's context: `Vector{T}` for a
  buffer (a raw device pointer, indexed with `unsafe_load`/`unsafe_store!`), a bits type, `Val{x}`.
- `is_ka=true` tells `call_binary` to bind KA's launch context and the backend's state word ahead of the
  arguments, and to dispatch flat. Compile KA kernels with `ka_compile` so the binary matches. It is a
  property of the binary, so one kernel type can hold a KA `fwd` and a Triton `bwd`.
- A KA kernel is just another source of binaries: switching between a KA, a Metal and a Triton kernel
  never changes the op. To make it a persistable default, list it in `@default_kernel` (after its type
  is defined).
- **Not for tracing KA kernels.** Under Reactant the program only *names* the compiled file, so XLA
  never sees the kernel's code. To have Reactant trace a KA kernel itself, run Reactant over the
  `@kernel` function instead of running through KernelDispatch. The ability to optionally trace kernels is 
  planned functionality (but is currently only supported through CUDA in Reactant).

## AD

- **Eager Enzyme:** one rule on `forward(op, kernel, args...)` covers every op and runs your
  `backward`, so Enzyme never enters your `forward` or the binary.
- **Under Reactant:** call `backward` directly. A gradient traced *through* a kernel throws, because
  Enzyme-JAX cannot differentiate a custom call yet (EnzymeAD/Enzyme#2516); set
  `KernelDispatch.KERNEL_IN_TRACED_AUTODIFF[] = true` once it can.
