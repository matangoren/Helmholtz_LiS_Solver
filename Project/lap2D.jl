# using PyPlot
using Plots
using FFTW
using SparseArrays
using LinearAlgebra
using Images, FileIO
using KrylovMethods
using Printf
include("auxiliary.jl");

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
    Lap1D = (h::Float64,n::Int64) -> 
        (A = spdiagm(0=>(2/h^2)*ones(ComplexF64, n),1=>(-1/h^2)*ones(ComplexF64, n-1),-1=>(-1/h^2)*ones(ComplexF64, n-1)); #- Sommerfeld;
        # A[1,end] = -1/h^2;                                # Periodic BC.
        # A[end,1] = -1/h^2;
        A[1,1]=1/h^2;                                       # Neuman BC. See NumericalPDEs to understand why.
        A[1,1] -= 1im * sqrt(real(m_base)) * (1.0/h);            # Sommerfeld
        A[n,n]=1/h^2;
        A[n,n] -= 1im * sqrt(real(m_base)) * (1.0/h);            # Sommerfeld
        return A;
        );

    # This is another way to add the Sommerfeld BC. When using this, also uncomment the comment at the end of line 75.
    # fact = 1 * sqrt(real(m_base)) * (1.0/h);
    # Sommerfeld = zeros(n, n)
    # Sommerfeld[1, :] .= fact
    # Sommerfeld[:, 1] .= fact
    # Sommerfeld[end, :] .= fact
    # Sommerfeld[:, end] .= fact
    # Sommerfeld = 1im .* Sommerfeld
    # Sommerfeld = spdiagm(0=>Sommerfeld[:])

    # Lap2D = kron(In(n), Lap1D(h,n)) + kron(Lap1D(h,n), In(n)) - m_base .* spdiagm(0=>ones(ComplexF64, n*n)); #- Sommerfeld;
    Lap2D = kron(In(n), Lap1D(h,n)) + kron(Lap1D(h,n), In(n)) - m_base .* spdiagm(0=>ratios[:]); #- Sommerfeld;
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
        xtt = fgmres(A_func, q[:], restrt, tol=tol, maxIter=max_iter, M=M, out=2, storeInterm=true, flexible=true)
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


m_0s_names = Dict(1 => "Average m", 2 => "Linear m",  3 => "Gaussian Range m", 4 => "Gaussian Deprecated",
5 => "Monte Carlo m", 6 => "Monte Carlo + Avarage m", 7 => "Min Max m")
m_0s_methods = [avg_m, linear_m, gaussian_range_m, gaussian_depricated_m, monte_carlo_m, combined_monte_carlo_avg, min_max_m]
# m_0s_names = Dict(1 => "Average m", 2 => "Linear m",  3 => "Gaussian Range m", 4 => "Gaussian Deprecated")
# m_0s_methods = [avg_m, linear_m, gaussian_range_m, gaussian_depricated_m]

max_iter, restrt = 15, 20
# q = rand(ComplexF64, n, n) # + 1im * rand(ComplexF64, n, n)      # Random initializaton.

n_0 = 256
# rat = 0.1n_0
n, h, m_base, b, pad_green = init_params(n_0)
interpolated_split_ratio = interpolated_split_grid_ratio(0.5, 1, n)
test_fgmres_avg(m_base, interpolated_split_ratio, "Int Binary grid", max_iter, restrt, n, h, b, pad_green, 3, m_0s_names, m_0s_methods)

triple_ratio = triple_grid_ratio(0.5, 0.75, 1,n)
split_ratio = split_grid_ratio(0.5, 1, n)
gaussian_ratio = gaussian_grid_ratio(1,0.2,n)
octa_ratio = octagon_grid_ratio(0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,n)
dual_ratio = dual_grid_ratio(0.5, 1, n)


# m_0s_linear = linear_m(m_base, deltas_ratio, max_iter, restrt)
# m_0s_avg = avg_m(m_base, dual_ratio, max_iter, restrt)
# m_0s_avg = avg_m(m_base, dual_ratio, max_iter, restrt)
# m_0s_rand = random_min_max_m(m_base, deltas_ratio, max_iter, restrt)
# m_0s_gaussian = gaussian_m(m_base, dual_ratio, max_iter, restrt)
# m_0s_monte_carlo = monte_carlo_m(m_base, deltas_ratio, max_iter, restrt)
# m_0s_minmax = min_max_m(m_base, dual_ratio, max_iter, restrt)
# m_0s_rand_no_rep = random_rep_m(m_base, dual_ratio, max_iter, restrt)
# m_0s_rand_no_rep = random_no_rep_m(m_base, deltas_ratio, max_iter, restrt)