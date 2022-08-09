using Random
using Distributions
using StatsBase

# Grid ratio initialization
function dual_grid_ratio(p, n)
    ratios = zeros(ComplexF64, n, n) .+ p
    ratios[Int(n/4)+1: Int(3n/4), Int(n/4)+1:Int(3n/4)] = ones(Int(n/2), Int(n/2))
    return ratios;
end

function tripel_grid_ratio(p1, p2, n)
    ratios = zeros(ComplexF64, n, n) .+ p1
    ratios[Int(n/6)+1: Int(n/4), Int(n/6)+1:Int(n/4)] .= p2
    ratios[Int(n/4)+1: Int(3n/4), Int(n/4)+1:Int(3n/4)] = ones(Int(n/2), Int(n/2))
    return ratios;
end

function const_grid_ratio(n) 
    return ones(ComplexF64, n, n);
end

function random_grid_ratio(n)
    return rand(n, n)
end

function delta_grid_ratio(p, n)
    ratios = ones(ComplexF64, n, n)
    ratios[Int(n/2), Int(n/2)] *= p
    return ratios;
end

function mc_grid_ratio(mu, sigma, n)
    d = Normal(mu, sigma)
    ratios = rand(d, n, n)
    return ratios
end

# m(x,y) Initialization
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

function random_min_max_m(m_base, ratio, max_iter, restrt)
    m_grid = m_base * ratio
    min_m, max_m = get_value(m_grid, findmin), get_value(m_grid, findmax)
    return rand(Uniform(real(min_m), real(max_m)), max_iter * restrt) .+ 
    1im * rand(Uniform(imag(min_m), imag(max_m)), max_iter * restrt);
end

function random_rep_m(m_base, ratio, max_iter, restrt)
    m_grid = m_base * ratio
    return sample(m_grid[:], max_iter * restrt, replace = false)
end

function monte_carlo_m(m_base, ratio, max_iter, restrt)
    m_grid = m_base * ratio
    return sample(m_grid[:], max_iter * restrt)
end

function avg_m(m_base, ratio, max_iter, restrt)
    m_grid = m_base * ratio
    avg_m = sum(m_grid) / (size(m_grid)[1] * size(m_grid)[2])
    return avg_m * ones(ComplexF64, max_iter * restrt);
end

function gaussian_m(m_base, ratio, max_iter, restrt)
    d = fit(Normal, real.(ratio[:]))
    samples = rand(d, max_iter * restrt)
    return m_base * samples
end

function min_max_m(m_base, ratio, max_iter, restrt)
    m_grid = m_base * ratio
    min_m, max_m = get_value(m_grid, findmin), get_value(m_grid, findmax);
    m_0s = zeros(ComplexF64, max_iter * restrt)
    for i in 1:size(m_0s)[1]
        m_0s[i] = i % 2 == 0 ? min_m : max_m
    end
    return m_0s;
end

# other functions
# get element from matrix, given an operator (findmin/findmax)
function get_value(A, operator)
    _, indices = operator(norm.(A))
    i = indices[1]
    j = indices[2]
    return A[i,j];
end