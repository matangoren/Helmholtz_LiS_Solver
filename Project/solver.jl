using KrylovMethods

include("LiS/preconditioner.jl")
include("Multigrid/preconditioner.jl")
include("operators.jl")


mutable struct Multi_solver
    solvers::Vector          # vector containing our preconditioners
    alternate::Bool         # true=alternate between preconditioners, false=one after the other
    index::Int64            # when alternate=true the index indicates the current preconditioner
end

function getMulti_solver(solvers; alternate=false)
    return Multi_solver(solvers, alternate, 1)
end

# solving inside FGMRES - Ae=r
function solveLinearSystem(solver::LiS_solver, x, b, n)
    return LiS_solve(solver, b)
end

function solveLinearSystem(solver::MG_solver, x, b, n)
    return MG_solve(solver, x, b, n)
end

function solveLinearSystem(solver::Multi_solver, x, b, n)
    if solver.alternate
        res = solveLinearSystem(solver.solvers[solver.index], x, b, n)
        solver.index = mod((solver.index),length(solver.solvers))+1
        return res
    end
    # for current_solver in solver.solvers
    #     # println(typeof(current_solver))
    #     x = solveLinearSystem(current_solver, x, b, n)
    # end
    e = solveLinearSystem(solver.solvers[1], x, b, n)
    A = solver.solvers[1].A
    _, helmholtz_matrix = getHelmholtzMatrices(solver.solvers[1].m, solver.solvers[1].omega, solver.solvers[1].gamma; alpha=0.5)

    return e+solveLinearSystem(solver.solvers[2], x, b-A(e, helmholtz_matrix, solver.solvers[1].h), n)
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

function solve(solver, n, b, h, m, gamma, omega, restart, max_iter; helmholtzOp=secondOrderHelmholtz)
    _, helmholtz_matrix = getHelmholtzMatrices(m, omega, gamma; alpha=0.5)

    b = reshape(b, size(b)..., 1, 1)

    A(v) = vec(helmholtzOp(reshape(v, size(b)), helmholtz_matrix, h))
    M(v) = vec(solveLinearSystem(solver, zeros(ComplexF64, size(b)), reshape(v, size(b)), n))

    return fgmresWrapper(A, M, b, restart, max_iter)
                                            
end

