include("utils.jl")

function secondOrderHelmholtz(x::Array{ComplexF64}, m::Array{ComplexF64}, h::Vector{Float64})
    Δ = getLapacianConv(h)
    return Δ(x) - x.*m
end

function fourthOrderHelmholtz(x::Array{ComplexF64}, m::Array{ComplexF64}, h::Vector{Float64})
    Δ = getLapacianConv(h; center_coeff=(-2/3), edge_coeff=(-1/6))
    stencil = reshape([0 1/12 0; 1/12 2/3 1/12; 0 1/12 0],3,3,1,1)
    return Δ(x) - m.*Conv(stencil, zeros(1); pad=1)(x) # veify secoond term
end
