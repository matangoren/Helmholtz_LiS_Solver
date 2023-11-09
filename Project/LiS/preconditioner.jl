using FFTW

include("utils.jl")

function getFFTGreensFunction(n::Vector{Int64}, h::Vector{Float64}, m_0::ComplexF64, q::Array{ComplexF64}, pad::Vector{Int64})
    """
    n - size of the grid [n1, n2]
    kernel - the differential operator kernel
    kappa_0 - const value of the incident wavefield slowness sqaured medium
    q - delta function source term
    """
    h1 = -1 / (h[1]^2)
    h2 = -1 / (h[2]^2)
    kernel = ComplexF64.([[0 h1 0];[h2 -2*(h1+h2)-m_0 h2];[0 h1 0]])

    n_padded = n + 2 .* pad
    kernal_op = getKernelOperator(kernel, n_padded)
    q_padded = zeros(ComplexF64,n_padded...)
    q_padded[pad[1]+1:pad[1]+n[1],pad[2]+1:pad[2]+n[2]] .= q

    g = ifft(fft(q_padded) ./ fft(kernal_op))
    g = g[div(n[1],2):div(n[1],2)+2*n[1]-1,div(n[2],2):div(n[2],2)+2*n[2]-1] # crop
    # g = g[Int(n[1]/2):Int(5n[1]/2)-1,Int(n[2]/2):Int(5n[2]/2)-1] # crop
    

    # figure()
    # imshow(real(g)); colorbar();
    # savefig("my_green.png")
    # close()
    g = fftshift(g)
    # figure()
    # imshow(real(fft(g))); colorbar();
    # savefig("my_green_after_shift_and_fft_real.png")
    # close()
    # figure()
    # imshow(imag(fft(g))); colorbar();
    # savefig("my_green_after_shift_and_fft_imag.png")
    # close()

    return fft(g)
end


# function solveHelmholtz(n::Vector{Int64}, h::Vector{Float64}, q::Array{ComplexF64}, kappa_0::ComplexF64, pad::Vector{Int64})
#     δ = zeros(ComplexF64, n...);
#     δ[div(n[1],2), div(n[2],2)] = 1.0;
#     Fg = getGreensFunction(n, h, kappa_0, δ, pad)

#     q_padded = zeros(ComplexF64,(2 .* n)...)
#     q_padded[Int(n[1]/2)+1:Int(3n[1]/2),Int(n[2]/2)+1:Int(3n[2]/2)] .= q

#     sol = ifft(Fg .* fft(q_padded))
#     return sol[Int(n[1]/2)+1:Int(3n[1]/2),Int(n[2]/2)+1:Int(3n[2]/2)]
# end

struct LiS_solver
    Fg_inv::Array{ComplexF64}       # F(g) where g is the Green's function of A(κ_0)^-1
    pad:: Vector{Int64}             # Green's padding
    n::Vector{Int64}                # grid size
    δ::Array{ComplexF64}            # delta function source term
    h::Vector{Float64}              # discretization step size
end

function getLiS_solver(n::Vector{Int}, h::Vector{Float64}, δ::Array{ComplexF64}, m_0::ComplexF64, pad::Vector{Int64})
    g = getFFTGreensFunction(n, h, m_0, δ, pad)
    return LiS_solver(g, pad, n, δ, h)
end

function LiS_solve(solver::LiS_solver, r)
    n = solver.n
    r_padded = zeros(ComplexF64,(2 .* n)...)
    r_padded[div(n[1],2)+1:div(n[1],2)+n[1],div(n[2],2)+1:div(n[2],2)+n[2]] .= r

    e = ifft(solver.Fg_inv .* fft(r_padded))
    return e[div(n[1],2)+1:div(n[1],2)+n[1],div(n[2],2)+1:div(n[2],2)+n[2]]
end
