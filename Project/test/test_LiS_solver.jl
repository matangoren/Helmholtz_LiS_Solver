using Helmholtz
using PyPlot
using Printf

include("../solver.jl")
include("../auxiliary.jl")
include("../LiS/utils.jl")


initial_n = [64, 64]
kappa_coeff = [[0.5,0.7],[0.2,0.4], [0.9], [0.25,1.0], [0.5, 1.0], [0.75,1.0]]


# FGMRES iterations
max_iter = 40
restart = 20

# global v = []

for i=3:3
    n = 2^(i-1) .* initial_n
    h = 1 ./ n
    println("===== testing for $(n[1])x$(n[2]) grid =====")
    for c in kappa_coeff
        t = c[1]
        lower = upper = 1.0
        if length(c) == 2
            t = 1.0
            lower = c[1]
            upper = c[2]
        end
        println("$(length(c)==1 ? "const $(t)*ones" : "linear($(lower),$(upper))") test")        
        m = t.*linear_grid_ratio(lower, upper, n[1])
        omega = (0.2*pi) / (maximum(h)*maximum(sqrt.(m))) # wkh = 0.2pi
        gamma = getABL(n,false,ones(Int64,2)*16,omega) .+ 0.1*4*pi # 0.05*omega
        helmholtz_matrix = m.* omega .* (omega .- (im .* gamma))

        δ = zeros(ComplexF64, n...)
        δ[div(n[1],2),div(n[2],2)] = 1.0

        # point-sorce rhs
        b = zeros(ComplexF64, n...)
        b[div(n[1],2),div(n[2],2)] = 1.0

        # m_0 = average model
        gamma_0 = (maximum(m) - minimum(m)) < 0.1 ? (mean(m) < 0.5 ? mean(m) : 1-mean(m)) : (maximum(m) - minimum(m))./maximum(m)
        println("\t gamma_0 = $(gamma_0)")
        m_0 = (omega^2)*mean(m)*(1-im*gamma_0)
        
        solver = getLiS_solver(n, h, δ, m_0, n)
        x, iterations, error = solve(solver, n, b, h, m, gamma, omega, restart, max_iter);
        figure()
        imshow(reshape(real(x), n...)); colorbar();
        # append!(v, [x])

        println("\t iterations = $(iterations) with error=$(error)\n")
        
    end
    
end