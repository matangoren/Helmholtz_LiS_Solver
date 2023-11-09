using KrylovMethods

include("LiS/preconditioner.jl")
include("Multigrid/preconditioner.jl")
include("operators.jl")

# solving inside FGMRES - Ae=r
function solveLinearSystem(solver::LiS_solver, x, b, n)
    return LiS_solve(solver, b)
end

function solveLinearSystem(solver::MG_solver, x, b, n)
    return MG_solve(solver, x, b, n)
end

function fgmresWrapper(A, M, b, restart, max_iter)
    x_init = zeros(ComplexF64, size(b))   
    x,flag,err,iter,resvec = fgmres(A, vec(b), restart, tol=1e-6, maxIter=max_iter,
                                                    M=M, x=vec(x_init), out=-1, flexible=true)

    return x, length(resvec), err    
end


# function solve(solver::LiS_solver, b, restart, max_iter, A_func)

#     A(v) = A_func*v
#     M(v) = vec(solveLinearSystem(solver, zeros(ComplexF64, size(b)), reshape(v, size(b))))

#     return fgmresWrapper(A, M, b, restart, max_iter)
                                                
# end

function solve(solver::Union{MG_solver,LiS_solver}, n, b, h, m, gamma, omega, restart, max_iter; helmholtzOp=secondOrderHelmholtz)
    _, helmholtz_matrix = getHelmholtzMatrices(m, omega, gamma; alpha=0.5)

    b = reshape(b, size(b)..., 1, 1)

    A(v) = vec(helmholtzOp(reshape(v, size(b)), helmholtz_matrix, h))
    M(v) = vec(solveLinearSystem(solver, zeros(ComplexF64, size(b)), reshape(v, size(b)), n))

    return fgmresWrapper(A, M, b, restart, max_iter)
                                            
end

