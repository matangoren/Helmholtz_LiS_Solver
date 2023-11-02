using LinearAlgebra
using KrylovMethods

include("utils.jl")
include("operators.jl")


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

function helmholtzJacobi(x, b, h, m; A=secondOrderHelmholtz, w=0.8, max_iter=1)
    h1 = 1.0 / (h[1]^2)
    h2 = 1.0 / (h[2]^2)
    D = 2.0*(h1 + h2) .- m 
    w_D_inv = w ./ D
    for _ in 1:max_iter
        residual = b - A(x, m, h)  
        x += w_D_inv.*residual
    end
    return x
end

restriction(x) = Conv(reshape((1/16)*[1 2 1;2 4 2;1 2 1],3,3,1,1), (zeros(1)), stride=2,pad=1)(x)

prolonation(x) = ConvTranspose(reshape((1/4)*[1 2 1;2 4.0 2;1 2 1],3,3,1,1), zeros(1), stride=2,pad=1)(x)

function helmholtzVCycle(x, b, h, kappa, gamma, omega; A=secondOrderHelmholtz, smoother=helmholtzJacobi, level=2, relax_iter=1)
    sl_m, helmholtz_m = getHelmholtzMatrices(kappa, omega, gamma)
    x = smoother(x, b, h, sl_m; A=A, iterations=relax_iter)

    if level > 0
        r = b - A(x, helmholtz_m, h)

        kappa_coarse = restriction(reshape(kappa,size(kappa)...,1,1))[:,:,1,1]
        gamma_coarse = restriction(reshape(gamma,size(gamma)...,1,1))[:,:,1,1]
        r_coarse = restriction(r)
        e_coarse = zeros(size(r_coarse))

        # call v_cycle for Ae=r on the coarse grid
        e_coarse = v_cycle(e_coarse, r_coarse, kappa_coarse, gamma_coarse, omega;
                            A=A, smoother=smoother, level=level-1, relax_iter=relax_iter)

        x .+= prolonation(e_coarse) # correct x

    else
        # coarsest grid
        x_size = size(x)
        A_Coarsest(v) =  vec(A(reshape(v,x_size), sl_m, h))
        M_Coarsest(v) = vec(smoother(x, reshape(v,x_size), sl_m; A=A, max_iter=1))
        x, flag, err, iter, resvec = fgmres(A_Coarsest, vec(b), 10, tol=1e-15, maxIter=1,
                                            M=M_Coarsest, x=vec(x), out=-1, flexible=true)
        x = reshape(x, x_size)
    end

    x = smoother(x, b, h, sl_m; A=A, iterations=relax_iter)
    return x
end