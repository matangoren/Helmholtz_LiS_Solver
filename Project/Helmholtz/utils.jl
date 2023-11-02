using Flux

function getHelmholtzMatrices(kappa, omega, gamma; alpha=0.5)
    shifted_laplacian_matrix = (kappa.^2) .* omega .* (omega .- (im .* gamma) .- (im .* omega .* alpha))
    helmholtz_matrix = (kappa.^2) .* omega .* (omega .- (im .* gamma))
    return shifted_laplacian_matrix, helmholtz_matrix
end

function getLapacianConv(h::Vector{Float64}; center_coeff=-1, edge_coeff=0)
    h1 = center_coeff / (h[1]^2)
    h2 = center_coeff / (h[2]^2)    
    stencil = reshape([edge_coeff h1 edge_coeff;h2 -2*(h1+h2)+4*edge_coeff h2;edge_coeff h1 edge_coeff],3,3,1,1)
    return Conv(stencil, zeros(1); pad=1)
end
