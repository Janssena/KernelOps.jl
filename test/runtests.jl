# Run everything:        julia --project=test test/runtests.jl
# Run some sections:     julia --project=test test/runtests.jl kernels sdpa

using Test

const SECTIONS = ["kernels", "inference", "preferences", "metal", "sdpa", "reactant"]
unknown = setdiff(ARGS, SECTIONS)
isempty(unknown) || error("unknown test section(s) $unknown; known: $SECTIONS")
selected(name) = isempty(ARGS) || name in ARGS

@testset "KernelOps" begin
    include("fixtures.jl")
    
    selected("kernels") && @testset "kernels and registry" begin
        include("kernel.test.jl")
    end
    
    selected("inference") && @testset "type stability" begin
        include("inference.test.jl")
    end
    
    selected("preferences") && @testset "preferences" begin
        include("preferences.test.jl")
    end
    
    selected("sdpa") && @testset "SDPAOps" begin
        include("sdpa.test.jl")
    end

    selected("metal") && Sys.isapple() && @testset "Metal" begin
        include("metal.test.jl")
        
        selected("reactant") && @testset "Reactant" begin
            include("reactant.test.jl")
            include("custom_call.test.jl") # Technically Reactant based, but uses jax-mps
        end
    end
end
