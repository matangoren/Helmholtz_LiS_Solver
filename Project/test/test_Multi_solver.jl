using Helmholtz
using PyPlot
using Printf

include("../solver.jl")
include("../auxiliary.jl")
include("../LiS/utils.jl")


initial_n = [64, 64]
m_coeff = [[0.5,0.7],[0.2,0.4], [0.1], [0.25,1.0], [0.5, 1.0], [0.75,1.0]]


# FGMRES iterations
max_iter = 40
restart = 20

# global v = []

for i=4:4
    n = 2^(i-1) .* initial_n .+ 1
    h = 1 ./ n
    println("===== testing for $(n[1])x$(n[2]) grid =====")
    for c in m_coeff[4:4]
        t = c[1]
        lower = upper = 1.0
        if length(c) == 2
            t = 1.0
            lower = c[1]
            upper = c[2]
        end
        println("$(length(c)==1 ? "const $(t)*ones" : "linear($(lower),$(upper))") test")        
        m = t.*linear_grid_ratio(lower, upper, n[1])
        figure()
        imshow(m); colorbar();
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
        
        solver_MG = getMG_solver(secondOrderHelmholtz, helmholtzJacobi, h, m, gamma, omega, 3, 1)
        solver_LiS = getLiS_solver(n, h, δ, m_0, n)


        x, iterations, error = solve(solver_LiS, n, b, h, m, gamma, omega, restart, max_iter);
        println("LiS \t iterations = $(iterations) with error=$(error)\n")
        
        x, iterations, error = solve(solver_MG, n, b, h, m, gamma, omega, restart, max_iter);
        println("Multidrid \t iterations = $(iterations) with error=$(error)\n")
        
        solver_Multi = getMulti_solver([solver_MG, solver_LiS]; alternate=false)
        x, iterations, error = solve(solver_Multi, n, b, h, m, gamma, omega, restart, max_iter);
        println("Multi combined \t iterations = $(iterations) with error=$(error)\n")

        solver_Multi = getMulti_solver([solver_MG, solver_LiS]; alternate=true)
        x, iterations, error = solve(solver_Multi, n, b, h, m, gamma, omega, restart, max_iter);
        println("Multi alternate \t iterations = $(iterations) with error=$(error)\n")

        figure()
        imshow(reshape(real(x), n...)); colorbar();
        
    end
    
end