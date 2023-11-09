using Flux


function getLapacianConv(h::Vector{Float64}; center_coeff=-1, edge_coeff=0, pad=1)
    h1 = center_coeff / (h[1]^2)
    h2 = center_coeff / (h[2]^2)    
    stencil = Float64.(reshape([edge_coeff h1 edge_coeff;h2 -2*(h1+h2)+4*edge_coeff h2;edge_coeff h1 edge_coeff],3,3,1,1))
    return Conv(stencil, zeros(Float64, 1); pad=pad)
end

function secondOrderHelmholtz(x::Array{ComplexF64}, matrix, h::Vector{Float64})
    Δ = getLapacianConv(h; pad=0)
    
    x_repeated = pad_repeat(x, (1,1,1,1))
    return Δ(x_repeated)- x.*matrix
end

function fourthOrderHelmholtz(x::Array{ComplexF64}, matrix::Array{ComplexF64}, h::Vector{Float64})
    Δ = getLapacianConv(h; center_coeff=(-2/3), edge_coeff=(-1/6))
    stencil = reshape([0 1/12 0; 1/12 2/3 1/12; 0 1/12 0],3,3,1,1)

    return Δ(x) - matrix.*Conv(stencil, zeros(1); pad=1)(x) # veify secoond term
end
