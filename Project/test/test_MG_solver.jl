using Helmholtz
using PyPlot
using Printf

include("../solver.jl")
include("../auxiliary.jl")


initial_n = [64, 64]
m_coeff = [[0.1], [0.9], [0.25,1.0], [0.5, 1.0], [0.75,1.0]]


# FGMRES iterations
max_iter = 40
restart = 20

global v_MG = []

for i=3:3
    n = 2^(i-1) .* initial_n .+ 1
    h = 1 ./ n
    println("===== testing for $(n[1])x$(n[2]) grid =====")
    for c in m_coeff[1:1]
        t = c[1]
        if length(c) == 2
            t = 1.0
        end
        lower = c[1]
        upper = c[end]
        println("$(length(c)==1 ? "const $(t)*ones" : "linear($(lower),$(upper))") test")        
        m = t.*linear_grid_ratio(lower, upper, n[1])

        omega = (0.2*pi) / (maximum(h)*maximum(sqrt.(m))) # wkh = 0.2pi
        # omega = (0.2*pi) / (maximum(h)*maximum(kappa)) # wkh = 0.2pi
        gamma = getABL(n,false,ones(Int64,2)*16,omega) .+ 0.1*4*pi # 0.05*(omega/maximum(sqrt.(m)))#.+ 0.01*4*pi

        solver = getMG_solver(secondOrderHelmholtz, helmholtzJacobi, h, m, gamma, omega, 3, 1)
        
        # point-sorce rhs
        b = zeros(ComplexF64, n...)
        b[div(n[1],2),div(n[2],2)] = 1.0

        x, iterations, error = solve(solver, n, b, h, m, gamma, omega, restart, max_iter);
        figure()
        imshow(reshape(real(x), n...)); colorbar();
        # append!(v_MG, [x])

        println("\t iterations = $(iterations) with error=$(error)\n")
        
    end
    println()
end