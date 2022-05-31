using Plots
using PyPlot
using FFTW
using SparseArrays
using LinearAlgebra
using Images, FileIO



# hop = Lap2D(hx, nx) - m * In(3);
# m = (0.1/(hx^2))*(1.0 + 1im*0.01)

function matrix_conv()
    nx = 100
    ny = 100
    hx = 1/nx         # Why in LiS.jl is it 2.0/n?
    hy = 1/ny
    b = zeros(nx*ny);
    b[div(end, 2)] = 1;
    # b[Int(nx*ny/2), Int(ny*nx/2)] = 1;

    In = (n::Int64)->(return spdiagm(0=>ones(n)));

    Lap1D = (h::Float64,n::Int64) -> 
        (return spdiagm(0=>(-2/h^2)*ones(n),1=>(1/h^2)*ones(n-1),-1=>(1/h^2)*ones(n-1)));

    Lap2D = kron(In(ny), Lap1D(hx,nx)) + kron(Lap1D(hy,ny), In(nx));

    return Lap2D\b;
end
mat = matrix_conv()
resh_mat = reshape(mat, (100, 100))
heatmap(abs.(resh_mat))


# Using fft:
function fft_conv(kernel,n,b,m::ComplexF64)
    hop = zeros(ComplexF64,size(b)[1],size(b)[2]);
    hop[1:2,1:2] = kernel[2:3,2:3]
    hop[end,1:2] = kernel[1,2:3]
    hop[1:2,end] = kernel[2:3,1]
    hop[end,end] = kernel[1,1]
    hath = fft(hop);
    hatb = fft(b);
    hatu = hatb ./ hath;
    u = ifft(hatu);
    return u;
    end
n = 100;
# pad = 20;
# n = n+2*pad
h = 1.0/n;
# m = (0.1/(h^2))*(1.0 + 1im*0.00)          # m = k^2. In this case it is constant through space (x).

kernel = zeros(ComplexF64, 3, 3)
kernel += [[0 -1 0];[-1 2 -1];[0 -1 0]] / h^2 # - [[0 0 0];[0 m 0];[0 0 0]]
# m is more or less k^2
b = zeros(ComplexF64, n, n);
b[div(n,2), div(n,2)] = 1.0;

temp = fft_conv(kernel,n,b,m::ComplexF64)
heatmap(abs.(temp))



# Lap2D_kernel_x = ((1/hy)^2) * [[0, 0, 0], [1, -2, 1], [0, 0, 0]]
# Lap2D_kernel_y = ((1/hx)^2) * [[0, 1, 0], [0, -2, 0], [0, 1, 0]]
# Lap2D_kernel = Lap2D_kernel_x + Lap2D_kernel_y



