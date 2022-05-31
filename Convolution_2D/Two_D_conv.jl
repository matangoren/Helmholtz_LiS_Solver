using Plots
using PyPlot
using FFTW
using SparseArrays
using LinearAlgebra
using Images, FileIO


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



img_path = "C:/Users/razmo/Pictures/2018-07/DSC_0027.JPG"

# img_path = "D:\\University\\CS Project\\Helmholtz_Solver\\Convolution_2D\\IMG_1452.JPG"

img = load(img_path)
channels = channelview(img)

r = channels[1,:,:]
g = channels[2,:,:]
b = channels[3,:,:]

grad_x = [[1,0,-1] [2,0,-2] [1,0,-1]]
grad_y = [[1,2,1] [0,0,0] [-1,-2,-1]]

g_x_r = padding_conv(r, grad_x, "same")
g_x_g = padding_conv(g, grad_x, "same")
g_x_b = padding_conv(b, grad_x, "same")
g_y_r = padding_conv(r, grad_y, "same")
g_y_g = padding_conv(g, grad_y, "same")
g_y_b = padding_conv(b, grad_y, "same")

recombine_x = colorview(RGB, g_x_r, g_x_g,  g_x_b)
recombine_y = colorview(RGB, g_y_r, g_y_g,  g_y_b)
squar_r = sqrt.(g_x_r .* g_x_r + g_y_r .* g_y_r)
squar_g = sqrt.(g_x_g .* g_x_g + g_y_g .* g_y_g)
squar_b = sqrt.(g_x_b .* g_x_b + g_y_b .* g_y_b)
recombine_tot = colorview(RGB, squar_r, squar_g, squar_b)


function fft_conv(img, kernel)
    hop = zeros(ComplexF64,size(img)[1],size(img)[2]);
    hop[1:2,1:2] = kernel[2:3,2:3]
    hop[end,1:2] = kernel[1,2:3]
    hop[1:2,end] = kernel[2:3,1]
    hop[end,end] = kernel[1,1]
    hath = fft(hop);
    hatimg = fft(img);
    hatu = hath .* hatimg;
    u = ifft(hatu);
    return u;
end

# Old implementation of the above function.
# function fft_conv(img, kernel)
#     hop = zeros(ComplexF64,size(img)[1],size(img)[2]);
#     hop[1:3,1:3] = kernel
#     hath = fft(hop);
#     hatimg = fft(img);
#     hatu = hath .* hatimg;
#     u = ifft(hatu);
#     return u;
# end

g_x_r_fft = fft_conv(r, grad_x)
g_x_g_fft = fft_conv(g, grad_x)
g_x_b_fft = fft_conv(b, grad_x)
g_y_r_fft = fft_conv(r, grad_y)
g_y_g_fft = fft_conv(g, grad_y)
g_y_b_fft = fft_conv(b, grad_y)

recombine_x = colorview(RGB, abs.(g_x_r_fft), abs.(g_x_g_fft),  abs.(g_x_b_fft))
recombine_y = colorview(RGB, abs.(g_y_r_fft), abs.(g_y_g_fft),  abs.(g_y_b_fft))
squar_r_fft = sqrt.(g_x_r_fft .* conj.(g_x_r_fft) + g_y_r_fft .* conj.(g_y_r_fft))
squar_g_fft = sqrt.(g_x_g_fft .* conj.(g_x_g_fft) + g_y_g_fft .* conj.(g_y_g_fft))
squar_b_fft = sqrt.(g_x_b_fft .* conj.(g_x_b_fft) + g_y_b_fft .* conj.(g_y_b_fft))
recombine_tot = colorview(RGB, abs.(squar_r_fft), abs.(squar_g_fft), abs.(squar_b_fft)) # No need for abs.



