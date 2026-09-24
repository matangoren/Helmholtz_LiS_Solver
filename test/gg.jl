include("./test_import.jl")

include("../src/solvers/LiS_solver.jl")
include("../src/solvers/multigrid_solver.jl")



# CUDA.allowscalar(true)


laplacian_stencil = ones(Float64, (7,7,7)) |> device
mass_stencil      = ones(Float64, (7,7,7)) |> device

laplacian_stencil_forH = ones(Float64, (7,7,7,1,1)) |> device
mass_stencil_forH      = ones(Float64, (7,7,7,1,1)) |> device

dim = 3


# ---------------------------------------------------------
# Timing
# ---------------------------------------------------------

for n1 in [64 128]#, 256]#, 512

    println("LiS timing for n1=$(n1)")

    n = [n1, n1, n1] .+ 1
    h = 1 ./ n


    # -----------------------------------------------------
    # Delta
    # -----------------------------------------------------

    delta = zeros(ComplexF64, n...)
    delta[div(n[1],2)+2,div(n[2],2)+2,div(n[3],2)+2] = 1.0

    delta = ComplexF64.(delta |> device)


    # -----------------------------------------------------
    # LiS solver
    # -----------------------------------------------------

    omega = 1.0
    vals = [0.5, 0.5, 0.5]

    m0_s = ComplexF64.((omega)^2 .* vals .* (1 - im*(0.01)))

    lis_solver = getLiS_solver(laplacian_stencil,mass_stencil,n,h,delta,m0_s,4 * ones(Int64, dim),4 * ones(Int64, dim))


    lis_solver.T_LiS = [ones(ComplexF64, n...) |> device for i = 1:length(vals)]

    b = ones(ComplexF64, n...) |> device


    
    # -----------------------------------------------------
    # LiS timing
    # -----------------------------------------------------
 total_time = 0.0

    for i = 1:10

        if CUDA.functional()
            CUDA.synchronize()
        end

        t0 = time_ns()

        x = weighted_LiS_solve(lis_solver, b)

        if CUDA.functional()
            CUDA.synchronize()
        end

        t = (time_ns() - t0) / 1e9
        total_time += t
    end

    println("\t -> Warmup LiS Time: $(total_time / 10)")

    total_time = 0.0

    for i = 1:10

        if CUDA.functional()
            CUDA.synchronize()
        end

        t0 = time_ns()

        x = weighted_LiS_solve(lis_solver, b)

        if CUDA.functional()
            CUDA.synchronize()
        end

        t = (time_ns() - t0) / 1e9
        total_time += t
    end

    println("\t -> LiS Time: $(total_time / 10)")


    # -----------------------------------------------------
    # Jacobi timing
    # -----------------------------------------------------

    total_time = 0.0

    x = reshape(b, size(b)..., 1, 1)

    for i = 1:10

        if CUDA.functional()
            CUDA.synchronize()
        end

        t0 = time_ns()

        temp = helmholtzJacobi(x,x,h,x,x,x,laplacian_stencil_forH,mass_stencil_forH;max_iter = 1)

        if CUDA.functional()
            CUDA.synchronize()
        end

        t = (time_ns() - t0) / 1e9
        total_time += t
    end

    println("\t -> Jacoby Time: $(total_time / 10)")
end