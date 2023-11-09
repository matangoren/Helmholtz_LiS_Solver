using PyPlot
# using Plots
using FFTW
using SparseArrays
using LinearAlgebra
# using Images, FileIO
using KrylovMethods
using Printf
include("auxiliary.jl");
include("subdomains.jl");

# Random.seed!(1234);


In = (n::Int64)->(return spdiagm(0=>ones(ComplexF64, n)));

function init_params(n)
    n = n;
    h = 2.0/n;
    m_base = (0.1/(h^2))*(1.0 + 1im*0.05)         # m = k^2. In this case it is constant through space (x).

    # Define a point-source in the middle of the grid.
    b = zeros(ComplexF64, n, n);
    b[div(n,2), div(n,2)] = 1.0;
    pad_green = n

    return n, h, m_base, b, pad_green
end

function fft_conv(kernel, n, pad, b)
    # Pad with pad at each side of the grid -> overall (n+2pad)*(n+2pad) grid.
    hop = zeros(ComplexF64,n+2pad,n+2pad);
    hop[1:2,1:2] = kernel[2:3,2:3]
    hop[end,1:2] = kernel[1,2:3]
    hop[1:2,end] = kernel[2:3,1]
    hop[end,end] = kernel[1,1]
    hath = fft(hop);
    b_new = zeros(ComplexF64,n+2pad,n+2pad)
    b_new[pad+1:pad+n,pad+1:pad+n] .= b
    hatb = fft(b_new);
    hatu = hatb ./ hath;
    u = ifft(hatu);
    return u;
end

function matrix_conv(n, h, b, m_base, ratios)
    Lap1D = spdiagm(0=>(2/h^2)*ones(ComplexF64, n),1=>(-1/h^2)*ones(ComplexF64, n-1),-1=>(-1/h^2)*ones(ComplexF64, n-1)); #- Sommerfeld;
    # Lap1D[1,end] = -1/h^2;                                # Periodic BC.
    # Lap1D[end,1] = -1/h^2;
    Lap1D[1,1]=1/h^2;                                       # Neuman BC. See NumericalPDEs to understand why.
    Lap1D[1,1] -= 1im * sqrt(real(m_base)) * (1.0/h);            # Sommerfeld
    Lap1D[n,n]=1/h^2;
    Lap1D[n,n] -= 1im * sqrt(real(m_base)) * (1.0/h);            # Sommerfeld

    # This is another way to add the Sommerfeld BC. When using this, also uncomment the comment at the end of line 75.
    # fact = 1 * sqrt(real(m_base)) * (1.0/h);
    # Sommerfeld = zeros(n, n)
    # Sommerfeld[1, :] .= fact
    # Sommerfeld[:, 1] .= fact
    # Sommerfeld[end, :] .= fact
    # Sommerfeld[:, end] .= fact
    # Lap2D = kron(In(n), Lap1D) + kron(Lap1D, In(n)) - m_base .* spdiagm(0=>ones(ComplexF64, n*n)); #- Sommerfeld;
    Lap2D = kron(In(n), Lap1D) + kron(Lap1D, In(n)) - m_base .* spdiagm(0=>ratios[:]); #- Sommerfeld;
    # b = reshape(b, (n*n, 1))
    # return reshape((Lap2D\b),(n,n)), Lap2D
    return Lap2D
end 

function generate_green(n, kernel, b, pad_green)
    # Generate G (Green's function - solution for a single source in the middle of the grid).
    green = fft_conv(kernel, n, pad_green, b);
    # heatmap(real.(green))
    green = green[Int(n/2):Int(5n/2)-1,Int(n/2):Int(5n/2)-1]
    # heatmap(real.(green))
    green = fftshift(green)
    # heatmap(real.(green))
    return green
end

function solve_helm(n, q:: Matrix{ComplexF64}, green)
    q_pad = zeros(ComplexF64,2n,2n)
    q_pad[Int(n/2)+1:Int(3n/2),Int(n/2)+1:Int(3n/2)] .= q
    
    # Perform the convolution of the Green's function with the source.
    sol = ifft(fft(green) .* fft(q_pad))
    sol = sol[Int(n/2)+1:Int(3n/2),Int(n/2)+1:Int(3n/2)]
    # heatmap(real.(sol))
    return sol
end

function sanity_check()
    q = q = zeros(ComplexF64, n, n);                                  # Point source at [n/4, n/4].
    q[div(n,4), div(n,4)] = 1.0;
    # Sanity check: L*u needs to return q approximately
    m_base = (0.1/(h^2))*(1.0 + 1im*0.05)
    ratios = zeros(ComplexF64, n, n) .+ 0.85             # Make sure this is broadcasted.
    ratios[Int(n/4)+1: Int(3n/4), Int(n/4)+1:Int(3n/4)] = ones(Int(n/2), Int(n/2))
    sol, hop = matrix_conv(n, h, q, m_base, ratios)             # hop is Lap2D, calculated in matrix_conv.

    m_0 = m_base
    kernel = zeros(ComplexF64, 3, 3);
    kernel += [[0 -1 0];[-1 4 -1];[0 -1 0]] / h^2 - m_0 .* [[0 0 0];[0 1 0];[0 0 0]];
    f = () -> fft_conv(kernel, n, 0, q)
    fft_conv_sol = f()
    f2 = () -> norm(hop * vec(fft_conv_sol) .- vec(q)) / norm(vec(q))
    f3 = () -> norm(hop * vec(sol) .- vec(q)) / norm(vec(q))
    display(f2())
    display(f3())
    heatmap(real.(vec(fft_conv_sol) - vec(sol)))

    return norm(hop\vec(q) - vec(sol)) / norm(hop\vec(q))
end

function whole_process()
    # Need to update to run properly.

    q = zeros(ComplexF64, n, n);                                  # Point source at [n/4, n/4].
    q[div(n,4), div(n,4)] = 1.0;
    init_params()
    g_temp = generate_green(n, kernel, b, pad_green)
    sol = solve_helm(n, q, g_temp)
end

function get_M(n, q, g_temp, kernel)
    # Generate the Greens function, if didn't get it as param.
    if isempty(g_temp)
        g_temp = generate_green(n, kernel, b, pad_green)
    end
    # Solve the system
    q = reshape(q, (n, n))
    sol = solve_helm(n, q:: Matrix{ComplexF64}, g_temp)
    return sol[:]
end

function M_gen(q, n, h, m_g, b, pad_green)
    try
        m = take!(m_g)
        kernel = zeros(ComplexF64, 3, 3);
        kernel += [[0 -1 0];[-1 4 -1];[0 -1 0]] / h^2 - m .* [[0 0 0];[0 1 0];[0 0 0]];
        g_temp = generate_green(n, kernel, b, pad_green)
        return get_M(n, q, g_temp, kernel)
    catch e
        println("Some problem occured in M_gen!")
    end
end

function create_gen_m(m_0s)
    Channel() do ch2
        for j in 1:size(m_0s)[1]
            put!(ch2, m_0s[j])
        end
    end
end

"""
# The output values of fgmres are: 
#   1. strage long matrix (40K x num of iter).
#   2. flag (-1 for maxIter reached without converging and -9 for right hand side was zero).
#   3. Min value.
#   4. Number of iterations.
#   5. The history of the gmres sequensce. 
"""
function fgmres_sequence(q, ratios, m_0s, n, h, m_base, b, pad_green, max_iter=10, restrt=10)
    A = matrix_conv(n, h, q, m_base, ratios)     # A is hop (Helmholtz Operator).
    A_func = x -> A * x
    tol = 1e-6;
    m_g = create_gen_m(m_0s)
    M = q -> M_gen(q, n, h, m_g, b, pad_green)
    # test printing and behaviour for early stopping
    try
        xtt = fgmres(A_func, q[:], restrt, tol=tol, maxIter=max_iter, M=M, out=2, storeInterm=false, flexible=true)
        figure()
        imshow(reshape(real(xtt[1]),n,n))

        return xtt
    catch e
        println("Probably reached the maximal number of iterations without converging!")
        return [-1 -1 -1 -1 -1]
    end
end

"""
# execute fgmres sequence for 'number_of_repetitions' times and print the result (with standard deviation for each method).
#   number_of_repetitions - number of times each m_0s method is executed.
#   m_0s_names - array with names of methods (Practicaly used for more informative dispaly).
#   m_0s_methods - an array with the methods used for choosing m_0.
"""
function test_fgmres_avg(m_base, ratio, grid_name, max_iter, restrt, n, h, b, pad_green, number_of_repetitions, m_0s_names, m_0s_methods)
    res = []
    make_m_0s = (m_0_method, ratio) -> m_0_method(m_base, ratio, max_iter, restrt)
    for (i,method) in enumerate(m_0s_methods)
        num_of_iter, val, iter_arr = 0, 0, []
        for j in 1:number_of_repetitions
            println("Method: ", m_0s_names[i], ",   Iteration Number: ", j)
            # Each time we generate a different source.
            q = rand(ComplexF64, n, n)
            q = zeros(ComplexF64, n, n)
            q[div(n,2),div(n,2)]=1.0
            # Creating an array of m_0s for a specific method. 
            m_0s = make_m_0s(method, ratio)
            # Excute fgmres sequence.
            r = fgmres_sequence(q, ratio, m_0s, n, h, m_base, b, pad_green, max_iter, restrt)
            # updating several parameters for statistics.
            num_of_iter += size(r[5]) != () ? length(r[5]) : max_iter * restrt
            val += size(r[5]) != () ? r[3] : 1e-6
            append!(iter_arr, size(r[5]) != () ? length(r[5]) : max_iter * restrt)
        end
        # updating more parameters for statistics.
        std_err = compute_stderr(iter_arr, number_of_repetitions)
        num_of_iter = num_of_iter / number_of_repetitions
        push!(res, (i,num_of_iter, (val / number_of_repetitions), std_err))
    end
    sort!(res, by=(x) -> (x[2],x[3]))
    print_result_with_err(res, m_0s_names, grid_name)
end


# m_0s_names = Dict(1 => "Average m", 2 => "Linear m",  3 => "Gaussian Range m", 4 => "Gaussian Deprecated",
# 5 => "Monte Carlo m", 6 => "Monte Carlo + Avarage m", 7 => "Min Max m")
# m_0s_methods = [avg_m, linear_m, gaussian_range_m, gaussian_depricated_m, monte_carlo_m, combined_monte_carlo_avg, min_max_m]
# m_0s_names = Dict(1 => "Average m", 2 => "Min Max m", 3 => "Monte Carlo m");
# m_0s_methods = [avg_m, min_max_m, monte_carlo_m];
m_0s_names = Dict(1 => "Average m");
m_0s_methods = [avg_m];

max_iter, restrt = 40, 20;
# q = rand(ComplexF64, n, n) # + 1im * rand(ComplexF64, n, n)      # Random initializaton.

n_0 = 256;
# rat = 0.1n_0
n, h, m_base, b, pad_green = init_params(n_0);
linear_ratio = 0.1*linear_grid_ratio(1.0, 1.0, n)
test_fgmres_avg(m_base, linear_ratio, "Linear grid", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)

# 9 subdomains wedge experiment
# wedge11, wedge12, wedge13, wedge21, wedge22, wedge23, wedge31, wedge32, wedge33 = wedge_9_subdomains(0.25, 1, n);
# test_fgmres_avg(m_base, wedge11, "Wedge grid - patch [1,1]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge12, "Wedge grid - patch [1,2]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge13, "Wedge grid - patch [1,3]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge21, "Wedge grid - patch [2,1]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge22, "Wedge grid - patch [2,2]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge23, "Wedge grid - patch [2,3]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge31, "Wedge grid - patch [3,1]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge32, "Wedge grid - patch [3,2]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge33, "Wedge grid - patch [3,3]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)

# 16 subdomains wedge experiment
# wedge11, wedge12, wedge13, wedge14, wedge21, wedge22, wedge23, wedge24, wedge31, wedge32, wedge33, wedge34, wedge41, wedge42, wedge43, wedge44 = wedge_16_subdomains(0.25, 1, n);
# test_fgmres_avg(m_base, wedge11, "Wedge grid - patch [1,1]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge12, "Wedge grid - patch [1,2]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge13, "Wedge grid - patch [1,3]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge14, "Wedge grid - patch [1,4]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge21, "Wedge grid - patch [2,1]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge22, "Wedge grid - patch [2,2]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge23, "Wedge grid - patch [2,3]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge24, "Wedge grid - patch [2,4]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge31, "Wedge grid - patch [3,1]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge32, "Wedge grid - patch [3,2]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge33, "Wedge grid - patch [3,3]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge34, "Wedge grid - patch [3,4]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge41, "Wedge grid - patch [4,1]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge42, "Wedge grid - patch [4,2]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge43, "Wedge grid - patch [4,3]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)
# test_fgmres_avg(m_base, wedge44, "Wedge grid - patch [4,4]", max_iter, restrt, n, h, b, pad_green, 1, m_0s_names, m_0s_methods)

