using PyPlot
using Plots
using FFTW
using SparseArrays
using LinearAlgebra
using Images, FileIO
using KrylovMethods
using Printf
using Random
using Distributions

Random.seed!(1234);

In = (n::Int64)->(return spdiagm(0=>ones(ComplexF64, n)));

function init_params()
    n = 200;
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
    b = reshape(b, (n*n, 1))
    return reshape((Lap2D\b),(n,n)), Lap2D
end 

function generate_green(n, kernel, b, pad_green)
    # Generate G (Green's function - solution for a single source in the middle of the grid).
    temp = fft_conv(kernel, n, pad_green, b);
    # heatmap(real.(temp))
    g_temp = temp[Int(n/2):Int(5n/2)-1,Int(n/2):Int(5n/2)-1]
    # heatmap(real.(g_temp))
    g_temp = fftshift(g_temp)
    # heatmap(real.(g_temp))
    return g_temp
end

function solve_helm(n, q:: Matrix{ComplexF64}, g_temp)
    q_pad = zeros(ComplexF64,2n,2n)
    q_pad[Int(n/2)+1:Int(3n/2),Int(n/2)+1:Int(3n/2)] .= q
    
    # Perform the convolution of the Green's function with the source.
    sol = ifft(fft(g_temp) .* fft(q_pad))
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
# sanity_check()

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
        println("Some problem occured in M_temp_gen!")
    end
end

function create_gen_m(m_0s)
    Channel() do ch2
        for j in 1:size(m_0s)[1]
            put!(ch2, m_0s[j])
        end
    end
end

function fgmres_sequence(q, ratios, m_0s, n, h, m_base, b, pad_green, max_iter=10, restrt=10)
    _ , A = matrix_conv(n, h, q, m_base, ratios)     # A is hop (Helmholtz Operator).
    A_func = x -> A * x
    tol = 1e-6;
    m_g = create_gen_m(m_0s)
    M = q -> M_gen(q, n, h, m_g, b, pad_green)
    # test printing and behaviour for early stopping
    xtt = fgmres(A_func, q[:], restrt, tol=tol, maxIter=max_iter, M=M, out=2, storeInterm=true)
    return xtt
end

function get_value(A, operator)
    _, indices = operator(norm.(A))
    i = indices[1]
    j = indices[2]
    return A[i,j];
end

function dual_grid_ratio(p, n)
    ratios = zeros(ComplexF64, n, n) .+ p
    ratios[Int(n/4)+1: Int(3n/4), Int(n/4)+1:Int(3n/4)] = ones(Int(n/2), Int(n/2))
    return ratios;
end

function const_grid_ratio(n) 
    return  ones(ComplexF64, n, n);
end

function linear_m(m_base, ratio, max_iter, restrt)
    m_grid = m_base * ratio
    min_m, max_m = get_value(m_grid, findmin), get_value(m_grid, findmax);
    delta = (real(max_m) - real(min_m)) / (max_iter * restrt) + abs(imag(max_m) - imag(min_m))im / (max_iter * restrt);
    m_0_reals = collect((i for i in real(min_m):real(delta):real(max_m)))
    m_0_ims = collect((i for i in imag(min_m):imag(delta):imag(max_m)))
    m_0s = zeros(ComplexF64, size(m_0_reals)[1])
    for i in 1:size(m_0s)[1]
        m_0s[i] = m_0_reals[i] + m_0_ims[i]im
    end
    return m_0s;
end

function random_m(m_base, ratio, max_iter, restrt)
    m_grid = m_base * ratio
    min_m, max_m = get_value(m_grid, findmin), get_value(m_grid, findmax)
    return rand(Uniform(real(min_m), real(max_m)), max_iter * restrt) .+ 
    1im * rand(Uniform(imag(min_m), imag(max_m)), max_iter * restrt);
end

function avg_m(m_base, ratio, max_iter, restrt)
    m_grid = m_base * ratio
    avg_m = sum(m_grid) / (size(m_grid)[1] * size(m_grid)[2])
    return avg_m * ones(ComplexF64, max_iter * restrt);
end

n, h, m_base, b, pad_green = init_params()
max_iter, restrt = 15, 15
q = rand(ComplexF64, n, n) # + 1im * rand(ComplexF64, n, n)      # Random initializaton.

dual_ratio = dual_grid_ratio(0.85, n)

m_0s_linear = linear_m(m_base, dual_ratio, max_iter, restrt)
m_0s_avg = avg_m(m_base, dual_ratio, max_iter, restrt)
m_0s_rand = random_m(m_base, dual_ratio, max_iter, restrt)
# The output values of fgmres are: 
#   1. strage long matrix (40K x num of iter).
#   2. flag (-1 for maxIter reached without converging and -9 for right hand side was zero).
#   3. Min value.
#   4. Number of iterations.
#   5. The history of the gmres sequensce. 
x = fgmres_sequence(q, dual_ratio, m_0s_linear, n, h, m_base, b, pad_green, max_iter, restrt)
size(x[5])
t1 = x[3]

y = fgmres_sequence(q, dual_ratio, m_0s_avg, n, h, m_base, b, pad_green, max_iter, restrt)
size(y[5])
t2 = y[3]

z = fgmres_sequence(q, dual_ratio, m_0s_rand, n, h, m_base, b, pad_green, max_iter, restrt)
size(z[5])
t3 = z[3]




