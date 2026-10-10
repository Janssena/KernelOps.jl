# Reading facts back out of an emitted MLIR module, for the tests that assert on the SHAPE of what
# gets emitted rather than on numbers.
#
# Used by `custom_call.test.jl` (mock kernels, any backend), so the reverse-rule convention is
# described in one place.

"The symbol `enzyme.reverse` points at, or `nothing`."
function reverse_symbol(ir::AbstractString)
    m = match(r"enzyme\.reverse = @\"?([^\"\s,}]+)\"?", ir)
    return m === nothing ? nothing : m.captures[1]
end

"`(arg_types, result_types)` of the reverse function, read back out of the emitted module."
function reverse_signature(ir::AbstractString)
    sym = reverse_symbol(ir)
    sym === nothing && return nothing
    for line in split(ir, '\n')
        (occursin("func.func", line) && occursin(sym, line)) || continue
        args = line[findfirst('(', line):findfirst(')', line)]
        rets = line[last(findfirst("->", line)):end]
        count_tensors(s) = length(collect(eachmatch(r"tensor<[^>]*>", s)))
        return (count_tensors(args), count_tensors(rets))
    end
    return nothing
end
