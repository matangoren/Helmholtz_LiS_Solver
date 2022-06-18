using PyPlot
using Plots
using FFTW
using SparseArrays
using LinearAlgebra
using Images, FileIO

function fft_conv(kernel, n, b, m::ComplexF64)
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

In = (n::Int64)->(return spdiagm(0=>ones(n)));

function matrix_conv(n, h, b, m)
    Lap1D = (h::Float64,n::Int64) -> 
        (A = spdiagm(0=>(2/h^2)*ones(n),1=>(-1/h^2)*ones(n-1),-1=>(-1/h^2)*ones(n-1));
        # A[1,end] = -1/h^2;            # Periodic BC.
        # A[end,1] = -1/h^2;
        A[1,1]=1/h^2;                   # Neuman BC. See NumericalPDEs to understand why.
        A[n,n]=1/h^2;
        return A;
        );
    
    fact = 10 * sqrt(real(m)) * (1.0/h);
    Sommerfeld = spdiagm(0=>zeros(n*n))
    Sommerfeld[1, :] .= fact
    Sommerfeld[:, 1] .= fact
    Sommerfeld[end, :] .= fact
    Sommerfeld[:, end] .= fact
    Sommerfeld = 1im .* Sommerfeld

    Lap2D = kron(In(n), Lap1D(h,n)) + kron(Lap1D(h,n), In(n)) - m .* spdiagm(0=>ones(n*n)) - Sommerfeld;
    print(Lap2D[1, 1])
    b = reshape(b, (n*n, 1))
    return reshape((Lap2D\b),(n,n))
end 


n = 200;
# pad = 20;
# n = n+2*pad
h = 2.0/n;
m = (0.1/(h^2))*(1.0 + 1im*0.00)          # m = k^2. In this case it is constant through space (x).

kernel = zeros(ComplexF64, 3, 3);
kernel += [[0 -1 0];[-1 4 -1];[0 -1 0]] / h^2 - m .* [[0 0 0];[0 1 0];[0 0 0]];
# m is more or less k^2
b = zeros(ComplexF64, n, n);
b[div(n,2), div(n,2)] = 1.0;

# temp = fft_conv(kernel, n, b, m);
# heatmap(real.(temp))

mat = matrix_conv(n, h, b, m);
heatmap(real.(mat))
heatmap(imag.(mat))
heatmap(abs.(mat))
