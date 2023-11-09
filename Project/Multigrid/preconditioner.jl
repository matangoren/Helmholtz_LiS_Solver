using LinearAlgebra
using KrylovMethods

function getHelmholtzMatrices(m, omega, gamma; alpha=0.5)
    shifted_laplacian_matrix = m .* omega .* (omega .- (im .* gamma) .- (im .* omega .* alpha))
    helmholtz_matrix = m .* omega .* (omega .- (im .* gamma))
    return shifted_laplacian_matrix, helmholtz_matrix
end

# x is assumed to be of size (H,W,C,N) where N = batch size
# kappa and gamma are two-dimensional (H,W)


# function jacobi(A, x, b; iterations=1, w=0.8)
#     w_D_inv = w ./ diag(A)
#     for i=1:iterations
#         r = b - A*x
#         x .+= w_D_inv.*r
#     end
    
#     return x
# end


function helmholtzJacobi(x, b, h, matrix; A=secondOrderHelmholtz, w=0.8, max_iter=1)
    h1 = 1.0 / (h[1]^2)
    h2 = 1.0 / (h[2]^2)
    D = 2.0*(h1 + h2) .- matrix 
    w_D_inv = w ./ D
    for _ in 1:max_iter
        residual = b - A(x, matrix, h)  
        x += (w_D_inv.*residual)
    end
    return x
end

function restriction(x) 
    return Conv(Float64.(reshape((1/16)*[1 2 1;2 4 2;1 2 1],3,3,1,1)), (zeros(Float64,1)), stride=2,pad=1)(x)
end

function prolongation(x) 
    return ConvTranspose(Float64.(reshape((1/4)*[1 2 1;2 4.0 2;1 2 1],3,3,1,1)), zeros(Float64, 1), stride=2,pad=1)(x)
end

function helmholtzVCycle(n, x, b, h, m, gamma, omega; A=secondOrderHelmholtz, smoother=helmholtzJacobi, level=3, relax_iter=1)
    sl_m, helmholtz_m = getHelmholtzMatrices(m, omega, gamma)
    x = smoother(x, b, h, sl_m; A=A, max_iter=relax_iter)

    if level > 1
        r = b - A(x, helmholtz_m, h)

        m_coarse = restriction(reshape(m,size(m)...,1,1))[:,:,1,1]
        gamma_coarse = restriction(reshape(gamma,size(gamma)...,1,1))[:,:,1,1] 
        r_coarse = restriction(real(r)) + im*restriction(imag(r))

        n_coarse = div.(n,2)
        e_coarse = zeros(ComplexF64, (n_coarse.+1)...,1,1)

        # call v_cycle for Ae=r on the coarse grid
        e_coarse = helmholtzVCycle(n_coarse, e_coarse, r_coarse, h.*2, m_coarse, gamma_coarse, omega;
                            A=A, smoother=smoother, level=level-1, relax_iter=relax_iter)
        
        fine_error = (prolongation(real(e_coarse)) + im * prolongation(imag(e_coarse)))
        x .+= fine_error # correct x

    else
        # coarsest grid
        x_size = size(x)
        A_Coarsest(v) =  vec(A(reshape(v,x_size), sl_m, h))
        M_Coarsest(v) = vec(smoother(x, reshape(v,x_size), h, sl_m; A=A, max_iter=1))
        x, flag, err, iter, resvec = fgmres(A_Coarsest, vec(b), 10, tol=1e-15, maxIter=1,
                                            M=M_Coarsest, x=vec(x), out=-1, flexible=true)
        x = reshape(x, x_size)
    end

    x = smoother(x, b, h, sl_m; A=A, max_iter=relax_iter)
    return x
end


struct MG_solver
    A
    smoother
    h::Vector{Float64}
    m::Array{Float64}
    gamma::Array{Float64}
    omega::Float64
    level::Int64
    relax_iter::Int64
end

function getMG_solver(A, smoother, h, m, gamma, omega, level, relax_iter)
    return MG_solver(A, smoother, h, m, gamma, omega, level, relax_iter)
end

function MG_solve(solver::MG_solver, x, b, n)
    return helmholtzVCycle(n, x, b, solver.h, solver.m, solver.gamma, solver.omega;
                            A=solver.A, smoother=solver.smoother, level=solver.level, relax_iter=solver.relax_iter)
end