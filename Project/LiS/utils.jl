using FFTW
using SparseArrays
using LinearAlgebra

function getKernelOperator(kernel::Array{Complex64}, n::Vector{Int64})
    k_n = size(kernel, 1) # the kernel is of size (kernel_n,kernel_n)
    kernel_op = zeros(ComplexF64,n...)
    kernel_op[1:k_n-1,1:k_n-1] = kernel[2:end,2:end]
    kernel_op[end,1:k_n-1] = kernel[1,2:end]
    kernel_op[1:k_n-1,end] = kernel[2:end,1]
    kernel_op[end,end] = kernel[1,1]

    return kernel_op
end

function getLaplace1D(n::Int64, h::Float64, kappa_0)
    Lap1D = spdiagm(0=>(2/h^2)*ones(ComplexF64, n),1=>(-1/h^2)*ones(ComplexF64, n-1),-1=>(-1/h^2)*ones(ComplexF64, n-1))
    Lap1D[1,1] = Lap1D[end,end] =1/h^2 - im * sqrt(real(kappa_0)) * (1.0/h)
    
    return Lap1D
end

function getLaplace2DMatrix(n::Vector{Int64}, kappa_0::ComplexF64, h::Vector{Float64})
    In(n::Int) = spdiagm(0=>ones(ComplexF64, n));

    n_y, n_x = n
    h_y, h_x = h
    Lap1D_x = getLaplace1D(n_x, h_x, kappa_0)
    Lap1D_y = getLaplace1D(n_y, h_y, kappa_0)

    Lap2D = kron(Lap1D_x, In(n_y)) + kron(In(n_x),Lap1D_y) - kappa_0 .* spdiagm(0=>ones(prod(n)))  # D_yy ⊗ I + I ⊗ D_xx
    return Lap2D
end 