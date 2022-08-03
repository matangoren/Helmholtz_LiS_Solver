using PyPlot
using Plots
using FFTW
using SparseArrays
using LinearAlgebra
using Images, FileIO
using KrylovMethods
using Printf

In = (n::Int64)->(return spdiagm(0=>ones(ComplexF64, n)));

function init_params()
    n = 200;
    pad = n;
    h = 2.0/n;
    m = (0.1/(h^2))*(1.0 + 1im*0.05)         # m = k^2. In this case it is constant through space (x).
                                            # m is more or less k^2
    kernel = zeros(ComplexF64, 3, 3);
    kernel += [[0 -1 0];[-1 4 -1];[0 -1 0]] / h^2 - m .* [[0 0 0];[0 1 0];[0 0 0]];
    b = zeros(ComplexF64, n, n);
    b[div(n,2), div(n,2)] = 1.0;
    return kernel, n, pad, h, m, b;
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
    m_x = zeros(ComplexF64, n, n) .+ 0.8             # Make sure this is broadcasted.
    m_x[Int(n/4)+1: Int(3n/4), Int(n/4)+1:Int(3n/4)] = ones(Int(n/2), Int(n/2))

    # Lap2D = kron(In(n), Lap1D(h,n)) + kron(Lap1D(h,n), In(n)) - m .* spdiagm(0=>ones(ComplexF64, n*n)); #- Sommerfeld;
    Lap2D = kron(In(n), Lap1D(h,n)) + kron(Lap1D(h,n), In(n)) - m .* spdiagm(0=>m_x[:]); #- Sommerfeld;
    b1 = reshape(b, (n*n, 1))
    return reshape((Lap2D\b1),(n,n)), Lap2D
end 


function generate_green(kernel, n, pad, b, m)
    # Generate G (Green's function for a single source in the middle of the grid). Call it 'g_temp'.
    temp = fft_conv(kernel, n, pad, b);
    # heatmap(real.(temp))
    g_temp = temp[Int(n/2):Int(5n/2)-1,Int(n/2):Int(5n/2)-1]
    # heatmap(real.(g_temp))
    return fftshift(g_temp)
    # heatmap(real.(g_temp))
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

function M(n, m, q, kernel, pad, g_temp, b)
    # Return solution given the parameters.
    # Generate the Greens function, if didn't get it as param.
    if isempty(g_temp)
        g_temp = generate_green(kernel, n, pad, b, m)
    end

    # Solve the system
    q = reshape(q, (n, n))
    sol = solve_helm(n, q:: Matrix{ComplexF64}, g_temp)
    return sol[:]
end

function Mtemp(q, n, h, pad, curr_m, b, delta, condition)
    m = curr_m[1];
    kernel = zeros(ComplexF64, 3, 3);
    kernel += [[0 -1 0];[-1 4 -1];[0 -1 0]] / h^2 - m .* [[0 0 0];[0 1 0];[0 0 0]];
    println(m)
    if condition
        curr_m .-= delta;
    end
    g_temp = generate_green(kernel, n, pad, b, m)
    return M(n, m, q, kernel, pad, g_temp, b)
end

function sanity_check()
    # Sanity check: L*u needs to return q approximately
    kernel, n, pad, h, m, b = init_params()
    sol_temp, hop = matrix_conv(n, h, b, m)             # hop is Lap2D, calculated in matrix_conv.
    f = () -> hop * vec(sol)
    f2 = () -> norm(hop * vec(sol) .- vec(q)) / norm(vec(q))
    display(f2())
    t = reshape(f(), (n, n))
    # heatmap(real.(t))
    # heatmap(reshape(real.(hop\vec(q) - vec(sol)), (n, n)))
    norm(vec(sol))
    display(norm(hop\vec(q) - vec(sol)) / norm(hop\vec(q)))
end

function whole_process()
    q = zeros(ComplexF64, n, n);                                  # Point source at [n/4, n/4].
    q[div(n,4), div(n,4)] = 1.0;
    init_params()
    g_temp = generate_Green(n, kernel, pad, m)
    sol = solve_helm(n, q, g_temp)
end

function gmres_sequence(diff, q, m_x)
    # init_params
    kernel, n, pad, h, m, b = init_params()
    m_x = m .* m_x
    # q = rand(ComplexF64, n , n) # + 1im * rand(ComplexF64, n, n)      # Random initializaton.
    # q = q = zeros(ComplexF64, n, n);                                  # Point source at [n/4, n/4].
    # q[div(n,4), div(n,4)] = 1.0;
    # q = q[:]
    _ , A = matrix_conv(n, h, q, m)
    min_m = get_value(m_x, findmin);
    max_m = get_value(m_x, findmax);
    @printf("m max value: %f + %fi, m min value: %f + %fi\n", real(max_m), imag(max_m), real(min_m), imag(min_m));    maxiter = 7
    restrt = 10
    maxiter = 7
    delta = (real(max_m) - real(min_m)) / (maxiter * restrt) + abs(imag(max_m) - imag(min_m))im / (maxiter * restrt);
    println(delta)
    curr_m = [max_m];
    M_temp = q1 -> Mtemp(q1, n, h, pad, curr_m, b, delta, diff);
    tol = 1e-6;
    A_func = x -> A * x
    # test printing and behaviour for early stopping
    return fgmres(A_func,q[:] ,restrt,tol=tol,maxIter=maxiter,M=M_temp, out=2,storeInterm=true)
end

function get_value(A, operator)
    _, indices = operator(real.(A))
    i = indices[1]
    j = indices[2]
    return A[i,j];
end


n = 200
m_x = zeros(ComplexF64, n, n) .+ 0.8             # Make sure this is broadcasted.
m_x[Int(n/4)+1: Int(3n/4), Int(n/4)+1:Int(3n/4)] = ones(Int(n/2), Int(n/2))
q = rand(ComplexF64, n , n) # + 1im * rand(ComplexF64, n, n)      # Random initializaton.
# The output values of fgmres are: 
#   1. strage long matrix (40K x num of iter).
#   2. flag (-1 for maxIter reached without converging and -9 for right hand side was zero).
#   3. Min value.
#   4. Number of iterations.
#   5. The history of the gmres sequensce. 
x = gmres_sequence(false, q, m_x)
size(x[5])
t1 = x[3]
# Use 'clone' in Julia when using krylov methods of Eran's code (GMRES).

x = gmres_sequence(true, q, m_x)
size(x[5])
t2 = x[3]

t1 - t2
