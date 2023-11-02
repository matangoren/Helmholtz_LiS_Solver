using FFTW

include("utils.jl")

function getGreensFunction(n::Vector{Int}, h::Vector{Float64}, kappa_0::ComplexF64, q::Array{ComplexF64}, pad::Int)
    """
    n - size of the grid [n1, n2]
    kernel - the differential operator kernel
    kappa_0 - const value of the incident wavefield slowness sqaured medium
    q - delta function source term
    """
    h1 = -1 / (h[1]^2)
    h2 = -1 / (h[2]^2)    
    kernel = ComplexF64([[0 h1 0];[h2 -2*(h1+h2)-kappa_0 h2];[0 h1 0]])

    n_padded = n .+ 2*pad
    kernal_op = getKernelOperator(kernel, n_padded)
    q_padded = zeros(ComplexF64,n_padded...)
    q_padded[pad+1:pad+n[1],pad+1:pad+n[2]] .= q

    G = ifft(fft(kernal_op) ./ fft(q_padded))
    return fftshift(G[Int(n[1]/2):Int(5n[1]/2)-1,Int(n[2]/2):Int(5n[2]/2)-1])
end


function solveHelmholtz(n::Vector{Int}, h::Vector{Float64}, q::Array{CopmlexF64}, kappa_0::ComplexF64, pad::Int)
    δ = zeros(ComplexF64, n...);
    δ[div(n[1],2), div(n[2],2)] = 1.0;
    G = getGreensFunction(n, h, kappa_0, δ, pad)
    q_padded = zeros(ComplexF64,(2 .* n)...)
    q_padded[Int(n[1]/2)+1:Int(3n[1]/2),Int(n[2]/2)+1:Int(3n[2]/2)] .= q

    sol = ifft(fft(G) .* fft(q_padded))
    return sol[Int(n[1]/2)+1:Int(3n[1]/2),Int(n[2]/2)+1:Int(3n[2]/2)]
end

struct LiS_solver
    A_0_inv::
    fft_g::Array{ComplexF64}        # F(g) where g is a Green's function
    pad:: Int                       # Green's padding
    n::Vector{Int}                  # grid size
    δ::Array{ComplexF64}            # delta function source term
    h::Vector{Float64}              # discretization step size
end

function getLiSSolver(n::Vector{Int}, h::Vector{Float64}, q::Array{CopmlexF64}, kappa_0::ComplexF64, pad::Int)

end
