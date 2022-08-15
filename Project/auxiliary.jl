using Random
using Distributions
using StatsBase

# Grid ratio initialization
"""
return grid with two different intencities.
"""
function dual_grid_ratio(p, n)
    ratios = zeros(ComplexF64, n, n) .+ p
    ratios[Int(n/4)+1: Int(3n/4), Int(n/4)+1:Int(3n/4)] = ones(Int(n/2), Int(n/2))
    return ratios;
end

function triple_grid_ratio(p1, p2, n)
    ratios = zeros(ComplexF64, n, n) .+ p1
    ratios[div(2n,12)+1: div(10n,12), div(2n,12)+1:div(10n,12)] .= p2 
    ratios[div(5n,12)+1: div(7n,12), div(5n,12)+1:div(7n,12)] .= 1
    return ratios;
end

function octagon_grid_ratio(p1, p2, p3, p4, p5, p6, p7, p8, n)
    ratios = zeros(ComplexF64, n, n) .+ p1
    ratios[div(2n,16): div(15n,16), div(2n,16): div(15n,16)] .= p2
    ratios[div(3n,16): div(14n,16), div(3n,16):div(14n,16)] .= p3 
    ratios[div(4n,16): div(13n,16), div(4n,16):div(13n,16)] .= p4 
    ratios[div(5n,16): div(12n,16), div(5n,16):div(12n,16)] .= p5 
    ratios[div(6n,16): div(11n,16), div(6n,16):div(11n,16)] .= p6 
    ratios[div(7n,16): div(10n,16), div(7n,16):div(10n,16)] .= p7
    ratios[div(8n,16): div(9n,16), div(8n,16):div(9n,16)] .= p8 
    return ratios;
end

function const_grid_ratio(n) 
    return ones(ComplexF64, n, n);
end

function random_grid_ratio(n)
    return rand(n, n)
end

"""
n - grid size (n X n matrix)
p - intencity of delta
return a grid with a delta in the center of the grids.
"""
function delta_grid_ratio(p, n)
    ratios = ones(ComplexF64, n, n)
    ratios[Int(n/2), Int(n/2)] *= p
    return ratios;
end

"""
num - number of deltas
p1 - intencity of delta
p2 - intencitiy of the background
n - grid size (n X n matrix)
returns a grid with num deltas.
"""
function deltas_grid_ratio(num, p1, p2, n)
    if num > n * n
        num = n * n
    end
    indices = sample(1:n * n, num, replace = false)
    ratios = p2 .* ones(ComplexF64, n * n)
    for i in 1:num
        ratios[indices[i]] = p1
    end
    return reshape(ratios, n, n);
end

"""
return a grid with gaussian distribution.
"""
function gaussian_grid_ratio(mu, sigma, n)
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

"""
return an array with size of max_iter * restrt,
containing random valuse from ratio without repetitions.
"""
function random_no_rep_m(m_base, ratio, max_iter, restrt)
    m_grid = m_base * ratio
    return sample(m_grid[:], max_iter * restrt, replace = false)
end

"""
return an array with size of max_iter * restrt,
elements are taken from ratio with repetitions.
"""
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
    lo, hi = quantile.(d, [0.48, 0.52])
    x = range(lo, hi; length = max_iter * restrt)
    # samples = pdf.(d, x)
    return m_base * x
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

function print_result(res, names_dict, grid_name)
    s = @sprintf "Sequence Results for %s:\n" grid_name 
    for (i, (j, num_of_iteration, val)) in enumerate(res)
        s = @sprintf "%s%d. %-30s ---> Number of iteration: %d | Value: %0.3e\n" s i names_dict[j] num_of_iteration val
    end
    @printf "%s" s
end
