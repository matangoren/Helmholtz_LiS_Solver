using PyPlot
using Plots
using FFTW
using SparseArrays
using LinearAlgebra
using Images, FileIO
using KrylovMethods

function fft_conv(kernel, n, pad, b, m::ComplexF64)
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

function matrix_conv(n, h, b, m)
    Lap1D = (h::Float64,n::Int64) -> 
        (A = spdiagm(0=>(2/h^2)*ones(ComplexF64, n),1=>(-1/h^2)*ones(ComplexF64, n-1),-1=>(-1/h^2)*ones(ComplexF64, n-1)); #- Sommerfeld;
        # A[1,end] = -1/h^2;                                # Periodic BC.
        # A[end,1] = -1/h^2;
        A[1,1]=1/h^2;                                       # Neuman BC. See NumericalPDEs to understand why.
        A[1,1] -= 1im * sqrt(real(m)) * (1.0/h);            # Sommerfeld
        A[n,n]=1/h^2;
        A[n,n] -= 1im * sqrt(real(m)) * (1.0/h);            # Sommerfeld
        return A;
        );

    # This is another way to add the Sommerfeld BC. When using this, also uncomment the comment at the end of line 75.
    # fact = 1 * sqrt(real(m)) * (1.0/h);
    # Sommerfeld = zeros(n, n)
    # Sommerfeld[1, :] .= fact
    # Sommerfeld[:, 1] .= fact
    # Sommerfeld[end, :] .= fact
    # Sommerfeld[:, end] .= fact
    # Sommerfeld = 1im .* Sommerfeld
    # Sommerfeld = spdiagm(0=>Sommerfeld[:])

    # Add a space-dependency to m (which is approximately k^2).
    m_x = zeros(ComplexF64, n, n) .+ 0.85             # Make sure this is broadcasted.
    m_x[Int(n/4)+1: Int(3n/4), Int(n/4)+1:Int(3n/4)] = ones(Int(n/2), Int(n/2))

    # Lap2D = kron(In(n), Lap1D(h,n)) + kron(Lap1D(h,n), In(n)) - m .* spdiagm(0=>ones(ComplexF64, n*n)); #- Sommerfeld;
    Lap2D = kron(In(n), Lap1D(h,n)) + kron(Lap1D(h,n), In(n)) - m .* spdiagm(0=>m_x[:]); #- Sommerfeld;
    b = reshape(b, (n*n, 1))
    return reshape((Lap2D\b),(n,n)), Lap2D
end 

function init_params()
    n = 200;
    h = 2.0/n;
    m = (0.1/(h^2))*(1.0 + 1im*0.05)         # m = k^2. In this case it is constant through space (x).
                                            # m is more or less k^2
    kernel = zeros(ComplexF64, 3, 3);
    kernel += [[0 -1 0];[-1 4 -1];[0 -1 0]] / h^2 - m .* [[0 0 0];[0 1 0];[0 0 0]];

    In = (n::Int64)->(return spdiagm(0=>ones(ComplexF64, n)));
end

function generate_Green(n, kernel, pad, m)
    # Define a point-source in the middle of the grid.
    b = zeros(ComplexF64, n, n);
    b[div(n,2), div(n,2)] = 1.0;

    # Generate G (Green's function - solution for a single source in the middle of the grid).
    temp = fft_conv(kernel, n, pad, b, m);
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
    # Sanity check: L*u needs to return q approximately
    sol_temp, hop = matrix_conv(n, h, b, m)             # hop is Lap2D, calculated in matrix_conv.
    f = () -> hop * vec(sol)
    f2 = () -> norm(hop * vec(sol) .- vec(q)) / norm(vec(q))
    display(f2())
    t = reshape(f(), (n, n))
    # heatmap(real.(t))
    # heatmap(reshape(real.(hop\vec(q) - vec(sol)), (n, n)))
    norm(vec(sol))
    return norm(hop\vec(q) - vec(sol)) / norm(hop\vec(q))
end

function whole_process()
    q = zeros(ComplexF64, n, n);                                  # Point source at [n/4, n/4].
    q[div(n,4), div(n,4)] = 1.0;
    init_params()
    g_temp = generate_Green(n, kernel, pad, m)
    sol = solve_helm(n, q, g_temp)
end

function M(n, m, q, h, g_temp)
    # Return solution given the parameters.

    # Initialize kernel to be the five-point Laplacian operator (NumericalPDEs, pg. 56).
    kernel = zeros(ComplexF64, 3, 3);
    kernel += [[0 -1 0];[-1 4 -1];[0 -1 0]] / h^2 - m .* [[0 0 0];[0 1 0];[0 0 0]];

    # Generate the Greens function, if didn't get it as param.
    if isempty(g_temp)
        g_temp = generate_Green(n, kernel, pad, m)
    end

    # Solve the system
    q = reshape(q, (n, n))
    sol = solve_helm(n, q:: Matrix{ComplexF64}, g_temp)
    return sol[:]
end

function M_temp(q)
    n = 200;
    h = 2.0/n;
    m = (0.1/(h^2))*(1.0 + 1im*0.05)         # m = k^2. In this case it is constant through space (x).
    g_temp = generate_Green(n, kernel, pad, m)
    return M(n, m, q, h, g_temp)
end

function gmres_sequence()
    # init_params
    n = 200;
    h = 2.0/n;
    m = (0.1/(h^2))*(1.0 + 1im*0.05)         # m = k^2. In this case it is constant through space (x).
                                            # m is more or less k^2
    kernel = zeros(ComplexF64, 3, 3);
    kernel += [[0 -1 0];[-1 4 -1];[0 -1 0]] / h^2 - m .* [[0 0 0];[0 1 0];[0 0 0]];

    q = rand(ComplexF64, n * n) # + 1im * rand(ComplexF64, n, n)      # Random initializaton.
    # q = q = zeros(ComplexF64, n, n);                                  # Point source at [n/4, n/4].
    # q[div(n,4), div(n,4)] = 1.0;
    # q = q[:]
    _ , A = matrix_conv(n, h, q, m)
    tol = 1e-6;
    A_func = x -> A * x
    # test printing and behaviour for early stopping
    xtt = fgmres(A_func,q ,10,tol=tol,maxIter=5,M=M_temp, out=2,storeInterm=true)
    return xtt
end

gmres_sequence()


