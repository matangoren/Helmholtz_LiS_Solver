using Plots
using PyPlot
using FFTW
using SparseArrays
using LinearAlgebra
using Images, FileIO

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

function matrix_conv(n, h, b)
    In = (n::Int64)->(return spdiagm(0=>ones(n)));

    Lap1D = (h::Float64,n::Int64) -> 
        (return spdiagm(0=>(-2/h^2)*ones(n),1=>(1/h^2)*ones(n-1),-1=>(1/h^2)*ones(n-1)));

    Lap2D = kron(In(n), Lap1D(h,n)) + kron(Lap1D(h,n), In(n));
    b = reshape(b, (n*n, 1))
    return reshape((Lap2D\b),(n,n))
end 



n = 200;
# pad = 20;
# n = n+2*pad
h = 2.0/n;
m = (0.1/(h^2))*(1.0 + 1im*0.00)          # m = k^2. In this case it is constant through space (x).

kernel = zeros(ComplexF64, 3, 3);
kernel += [[0 -1 0];[-1 2 -1];[0 -1 0]] / h^2 - [[0 0 0];[0 m 0];[0 0 0]];
# m is more or less k^2
b = zeros(ComplexF64,n, n);
b[div(n,2), div(n,2)] = 1.0;

temp = fft_conv(kernel,n,b,m::ComplexF64);
heatmap(abs.(temp))

mat = matrix_conv(n,h,b);
heatmap(abs.(mat))



mat - temp


# img_path = "Helmholtz_Solver\\Convolution_2D\\logo_bgu_png.png"
# img = load(img_path)
# gray_image = Gray.(img)
# 
# g_x_r = padding_conv(gray_image, [[1,0,-1] [2,0,-2] [1,0,-1]], "same")
# heatmap(g_x_r, color = :greys)