using Plots
using PyPlot
using FFTW
using SparseArrays
using LinearAlgebra
using Images, FileIO

nx = 200
ny = 200
hx = 1/nx         # Why in LiS.jl is it 2.0/n?
hy = 1/ny
m = (0.1/(h^2))*(1.0 + 1im*0.01)

In = (n::Int64)->(return spdiagm(0=>ones(n)));

Lap1D = (h::Float64,n::Int64) -> 
    (return spdiagm(0=>(-2/h^2)*ones(n),1=>(1/h^2)*ones(n-1),-1=>(1/h^2)*ones(n-1)));

Lap2D = kron(In(ny), Lap1D(hx,nx)) + kron(Lap1D(hy,ny), In(nx));

hop = Lap1D(h, n) - m * In(3);







Lap2D_kernel_x = ((1/hy)^2) * [[0, 0, 0], [1, -2, 1], [0, 0, 0]]
Lap2D_kernel_y = ((1/hx)^2) * [[0, 1, 0], [0, -2, 0], [0, 1, 0]]
Lap2D_kernel = Lap2D_kernel_x + Lap2D_kernel_y















img_path = "Lap2D/bgu_logo.jpg"
img = load(img_path)
channels = channelview(img)
r = channels[1,:,:]
grad_x = [[1,0,-1] [2,0,-2] [1,0,-1]]
grad_y = [[1,2,1] [0,0,0] [-1,-2,-1]]

grad_fft = fft_conv(r, grad_x)
image = chan(Gray, abs.(grad_fft))




function fft_conv(img, kernel)
    hop = zeros(ComplexF64,size(img)[1],size(img)[2]);
    hop[1:3,1:3] = kernel
    hath = fft(hop);
    hatimg = fft(img);
    hatu = hath .* hatimg;
    u = ifft(hatu);
    return u;
end

# Stupid convolution 2D
function padding_conv(input, filter, padding="valid")
    input_r, input_c = size(input)
    filter_r, filter_c = size(filter)

    if padding == "same"
        pad_r = (filter_r - 1) ÷ 2 # Integer division.
        pad_c = (filter_c - 1) ÷ 2 # Needed because of Type-stability feature of Julia

        input_padded = zeros(input_r+(2*pad_r), input_c+(2*pad_c))
        for i in 1:input_r, j in 1:input_c
            input_padded[i+pad_r, j+pad_c] = input[i, j]
        end
        input = input_padded
        input_r, input_c = size(input)
    elseif padding == "valid"
        # We don't need to do anything here
    else 
        throw(DomainError(padding, "Invalid padding value"))
    end

    result = zeros(input_r-filter_r+1, input_c-filter_c+1)
    result_r, result_c = size(result)

    for i in 1:result_r
        for j in 1:result_c
            for k in 1:filter_r 
                for l in 1:filter_c 
                    result[i,j] += input[i+k-1,j+l-1]*filter[k,l]
                end
            end
        end
    end

    return result
end




























