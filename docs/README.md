# KernelOps.jl documentation

The tutorials build on one example package, [`examples/SDPAOps`](examples/SDPAOps): scaled
dot-product attention as an op with two kernels, a hand-written Metal binary (standing in for one from
Triton) and a KernelAbstractions kernel. Every snippet is taken from it, and its test suite
(`examples/SDPAOps/test/runtests.jl`) checks them on Metal.

1. [Switching kernels and persisting defaults](tutorial/1-switching-kernels.md): select a kernel for a
   session, pick a default with `@default_kernel`, and persist it as a preference across sessions.
2. [Registering kernels](tutorial/2-registering-kernels.md): the whole process for a prebuilt
   `.metallib` and for a KernelAbstractions kernel, from the binary to the registry.
3. [Custom forward and backward, and gradients with Enzyme](tutorial/3-forward-backward-enzyme.md).
4. [Compiling a Triton kernel to a `.metallib` with triton-msl](tutorial/4-triton-metallib.md): the
   Python side of a Triton binary, as trifast does it for LuxTriangleAttention. Not covered by the
   example's test suite (it needs a Python toolchain).

[`design.md`](design.md) is the reference: why the package is shaped the way it is.

## Running the example

From a Julia environment of your own:

```julia
using Pkg
Pkg.develop([PackageSpec(path="path/to/KernelOps.jl"),
             PackageSpec(path="path/to/KernelOps.jl/docs/examples/SDPAOps")])
Pkg.add(["Metal", "Enzyme"])
```

```julia
using SDPAOps, KernelOps, Metal
q, k, v = MtlArray(randn(Float32, 16, 100)), MtlArray(randn(Float32, 16, 70)), MtlArray(randn(Float32, 8, 70))
o = SDPA()(q, k, v)        # 8×100: one column per query
```

The first call on a device compiles or copies the kernel's binary into the registry
(`~/.cache/KernelOps`, or `$KERNELOPS_CACHE`); later calls and later sessions find it there.
