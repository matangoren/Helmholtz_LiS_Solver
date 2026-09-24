using MAT
using PyPlot
using Distributed
using ArgParse
using JLD2

include("./test_import.jl")
include("./test_setup.jl")

interType = T
laplace_type=T
mass_type=T

mat_filename = "$(@__DIR__)/dump/Marmousi/512_2048_DD_4_16.mat"

function showDist(A)

    values = vec(A)

    figure()
    hist(values, bins=100)

    xlabel("Pixel intensity")
    ylabel("Frequency")
    title("Pixel intensity distribution")
    grid(true)

    savefig("$(@__DIR__)/dump/Dist2.png", dpi=300, bbox_inches="tight")
    close()
end

function plotAll(mat_filename)
    data = matread(mat_filename)
    delta = 0.1

    figure()

    avg = 0;

    for (name, value) in data

        global avg += (maximum(value)-minimum(value)) / (delta);

        println("$(name) - (max-min)/(max*delta) = $((maximum(value)-minimum(value)) / (delta))")
        # Skip non-array entries
        if !(value isa AbstractArray)
            continue
        end

        values = vec(value) ./ maximum(value)

        # Compute histogram
        counts, edges = hist(values, bins=100)

        # Bin centers
        centers = (edges[1:end-1] + edges[2:end]) ./ 2

        # Plot distribution
        plot(centers, counts)

    end

    avg /= length(data)

    println("AVG = $(avg)")

    xlabel("Pixel intensity")
    ylabel("Frequency")
    title("Pixel intensity distributions")
    grid(true)

    savefig("$(@__DIR__)/dump/Dist1.png", dpi=300, bbox_inches="tight")
    close()
end

function getTest(n;model="linear")

	println(model)
    patch_size = n[2]
    println("patch_size = $(patch_size)")
	if model == "marmousi"
		A = readdlm("$(@__DIR__)/GeoModels/MarmousiVp_small.dat", data_type);
		m = expandModelNearest(A*1e-3, size(A),n);
		m = 1 ./ (m.^2)
		m = Matrix(transpose(m))
        n2 = div(size(m,2),2)

        m = m[:,n2-div(patch_size,2):n2+div(patch_size,2)] # take center column crop (all rows)

		n = collect(size(m))
		domain_m = [0.0,1,0.0,div(size(m,2)-1,size(m,1)-1)].*4
	elseif model == "tunnel"
		n = [96,96,48] .+ 1;
		A = read(joinpath(@__DIR__, "TunnelModels", "tunnel_network.bin"))
		A = data_type.(reshape(A,n...))
		vals = reshape(range(2000, 2200; length=n[3]), 1, 1, :);
		m = convert.(data_type, vals .+ zeros(data_type, n[1], n[2], n[3]));
		m[Bool.(A)] .= 340; # to verify.
		m .*= 1e-3;
		m = 1 ./ (m.^2);
		domain_m = [0.0,1,0.0,1,0.0,0.2] # just for now
		
	else # "linear"
		lb = 0.25
		if model == "const"
			lb = 1.0
		end
		m_coeff = [lb, 1.0]
		lower = m_coeff[1]
		upper = m_coeff[end]
		if length(n) == 2
			m = linear_grid_ratio(lower, upper, n; data_type=data_type)
			domain_m = [0.0,1,0.0,1]
		else
			# 3D linear model
			vals = range(lower, upper; length=n[2])
			m = convert.(data_type, reshape(vals, 1, n[2], 1) .* ones(data_type, n[1], 1, n[3]))
			domain_m = [0.0,1,0.0,1,0.0,1]
		end
	end

	Minv = getRegularMesh(domain_m,collect(size(m)) .- 1);

	h = Minv.h

	println("########## h = $(h) ##########")
	omega = real(getMaximalFrequency(m,Minv));
	# omega = real(getMaximalFrequency(m,Minv)*0.9);
	pad = 20*ones(Int64,length(n));
	ABLamp = omega;
	println("omega is ",omega/pi," times pi")
	println("grid size $(size(m))")

	gamma = getABL(Minv.n.+1,false,pad,Float64(omega)) .+ gamma_0*omega

	b = zeros(ComplexF64, n...)
	if length(n) == 2
		b[div(n[1],2)+1,div(n[2],2)+1] = 1.0
	else
		b[div(n[1],2)+1,div(n[2],2)+1, div(n[3],2)+1] = 1.0
	end

	return m,Minv,h,omega,gamma,b
end

function SPAI_optimization(n, X, AX, V_jacoby, lis_solver)
    if length(n) == 2
        V_LiS = single_LiS_solve(lis_solver, AX[:,:,:,1,1]) # a vector of all lis solutions
    else
        V_LiS = single_LiS_solve(lis_solver, AX[:,:,:,:,1,1]) # a vector of all lis solutions
    end

    omega_jacobi = zeros(ComplexF64,n...)
    omega_lis = [zeros(ComplexF64,n...) for i=1:length(V_LiS)]
    


    @views for I in CartesianIndices(size(AX)[1:length(n)])
        Ti = hcat(vec(V_jacoby[I,:]), hcat([vec(V_LiS[i][I,:]) for i=1:length(V_LiS)]...))
        
        vi = vec(X[I,:]);
        omega = Ti\vi;
	    omega_jacobi[I] = omega[1];
        for i=2:length(omega)
            omega_lis[i-1][I] = omega[i]
        end
    end

    return omega_jacobi, omega_lis, V_LiS
end

function SPAI_optimization_forPLOT(
    output_file,
    solver_matrices,
    laplacian_stencil,
    mass_stencil,
    n,
    h,
    lis_solvers,
    names
)

    @assert length(lis_solvers) == length(names)

    # --------------------------------------------------
    # Build X
    # --------------------------------------------------
    X = Matrix{Float64}(I, prod(n), prod(n))
    X = reshape(X, (n..., prod(n), 1, 1))

    sl_m, helmholtz_m, correction = solver_matrices

    # --------------------------------------------------
    # Compute AX
    # --------------------------------------------------
    AX = zeros(ComplexF64, size(X))

    Threads.@threads for i in axes(X, 3)
        @views AX[:, :, i] .= HelmholtzOperator(
            Array(X[:, :, i, :, :]),
            sl_m,
            correction,
            h,
            laplacian_stencil,
            mass_stencil
        )
    end

    # --------------------------------------------------
    # Jacobi
    # --------------------------------------------------
    D = getJacobiDiagonal(
        h,
        sl_m,
        correction,
        laplacian_stencil,
        mass_stencil
    )

    V_jacoby = AX ./ D

    # --------------------------------------------------
    # Compute SPAI and eigenvalues for every LiS solver
    # --------------------------------------------------
    eigenvalues = Vector{Any}(undef, length(lis_solvers))
    omega_lis_lengths = zeros(Int, length(lis_solvers))
    for k in eachindex(lis_solvers)

        println("Processing LiS solver: ", names[k])

        # ----------------------------------------------
        # SPAI optimization
        # ----------------------------------------------
        omega_jacobi,
        omega_lis,
        V_LiS = SPAI_optimization(
            n,
            X,
            AX,
            V_jacoby,
            lis_solvers[k]
        )

        # ----------------------------------------------
        # Construct M
        # ----------------------------------------------
        M = X - omega_jacobi .* V_jacoby

        for i in eachindex(omega_lis)
            M .-= omega_lis[i] .* V_LiS[i]
        end

        M = reshape(M, (prod(n), prod(n)))

        # ----------------------------------------------
        # Eigenvalues
        # ----------------------------------------------
        eigenvalues[k] = eigvals(M)
        omega_lis_lengths[k] = length(omega_lis)
        println(
            "  max |eigenvalue| = ",
            maximum(abs.(eigenvalues[k]))
        )
    end

    # --------------------------------------------------
    # Plot all cases as subplots
    # --------------------------------------------------
    num_plots = length(lis_solvers)

    ncols = ceil(Int, sqrt(num_plots))
    nrows = ceil(Int, num_plots / ncols)

    figure(figsize=(7 * ncols, 7 * nrows))

    for k in 1:num_plots

        subplot(nrows, ncols, k)

        λ = eigenvalues[k]

        # Eigenvalues
        scatter(
            real.(λ),
            imag.(λ),
            s=30,
            label="Eigenvalues"
        )

        # Unit circle
        θ = range(0, 2π, length=500)

        plot(
            cos.(θ),
            sin.(θ),
            linewidth=2,
            label="Unit circle"
        )

        axhline(0, linewidth=0.5)
        axvline(0, linewidth=0.5)

        axis("equal")

        xlabel("Real")
        ylabel("Imaginary")

        title("$(names[k])\nLiS basis: $(omega_lis_lengths[k]), max |λ|: $(maximum(abs.(λ)))")

        grid(true)
        legend()
    end

    tight_layout()

    # --------------------------------------------------
    # Save figure
    # --------------------------------------------------


    savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    close()
end

function kmeans_pixel_centers(A, K)
    # All pixel intensities as a vector
    x = Float64.(vec(A))

    # K-means expects observations in columns
    data = reshape(x, 1, :)

    result = kmeans(data, K)

    # Cluster centers
    centers = vec(result.centers)


    return centers
end

shift = 0.1
gamma_0 = 0.01

n = [512,128] .+ 1;
# n = [128,128] .+ 1;

# model = "linear" 
model = "marmousi"
m,Minv,h,omega,gamma,b = getTest(n;model=model)

Rs_Grid, Rs, Ps, laplacian_stencils, mass_stencils = buildCycle(level, repeat([interType],level-1), laplace_type, mass_type; dim=dim)

h_coarse = h .* (2^(level-1))

solver_matrices = getMatrices(Rs_Grid, m, omega, gamma, h, laplacian_stencils, mass_stencils; level=level, alphas=alphas, add_sommerfeld=add_sommerfeld, BC=orderNeumannBC)[1]


m_coarse = get_coarse_m(m, level, Rs_Grid)
h_coarse = h .* (2^(level-1))
n_coarse = collect(size(m_coarse))

delta_coarse = zeros(ComplexF64, n_coarse...)
delta_coarse[div(n_coarse[1],2)+2,div(n_coarse[2],2)+2] = 1.0


folder = joinpath(@__DIR__, "dump/MiddleCut_experiment")
mkpath(folder)

exp_folder = joinpath(folder, "$(model)/$(model)_$(n[2])_$(n[1])")
mkpath(exp_folder)


println("size of m = $(size(m))")
println("size of m_coarse = $(size(m_coarse))")
figure()
imshow(real(m_coarse))
colorbar()
tight_layout()

savefig(joinpath(exp_folder,"m.png"), dpi=300, bbox_inches="tight")
close()

lis_solvers = []
names = []

# uniform
# for delta in [0.05, 0.075,0.1,0.125, 0.15, 0.175, 0.2]
#     push!(names, "pUniform_delta=$(delta)")
#     values = collect((minimum(m)/maximum(m)):delta:1) .* maximum(m)
#     println(length(values))
#     m0_s = (omega)^2 .*  values .* (1 - im*(gamma_0+shift))
#     lis_solver_uniform = getLiS_solver(squeezeConvResult(laplacian_stencils[1]),squeezeConvResult(mass_stencils[1]), n_coarse, h_coarse, delta_coarse, m0_s, div.(n_coarse,2),div.(n_coarse,2))
    
#     push!(lis_solvers, lis_solver_uniform)
# end

# Kmeans - K is based on the same amount of LiS elements in the uniform case
for K=1:10
    # values = collect((minimum(m)/maximum(m)):delta:1) .* maximum(m)
    # K = length(values)
    # println(K)
    push!(names, "Kmeans_K=$(K)")
    centers = kmeans_pixel_centers(m_coarse ./ maximum(m_coarse), K) .* maximum(m_coarse)
    # centers = kmeans_pixel_centers(m_coarse, K)

    m0_s = (omega)^2 .*  centers .* (1 - im*(gamma_0+shift))
    lis_solver_uniform = getLiS_solver(squeezeConvResult(laplacian_stencils[1]),squeezeConvResult(mass_stencils[1]), n_coarse, h_coarse, delta_coarse, m0_s, div.(n_coarse,2),div.(n_coarse,2))
    
    push!(lis_solvers, lis_solver_uniform)
end

SPAI_optimization_forPLOT(
    joinpath(exp_folder,"Eigenvalues_Kmeans.png"),
    solver_matrices,
    laplacian_stencils[1],
    mass_stencils[1],
    n_coarse,
    h_coarse,
    lis_solvers,
    names
)





# f = jldopen("$(@__DIR__)/dump/Marmousi/512_2048_DD_4_16.jld2", "r")
# # f = jldopen("$(@__DIR__)/dump/Marmousi/1024_4096_DD_4_16.jld2", "r")

# for key in keys(f)
#     println("IN $(key)")
#     m_coarse, (sl_matrix, helmholtz_matrix, correction) = f[key]
#     println(size(m_coarse))
#     n_coarse = collect(size(m_coarse)) # for m in mat file
#     delta_coarse = zeros(ComplexF64, n_coarse...)
#     delta_coarse[div(n_coarse[1],2)+2,div(n_coarse[2],2)+2] = 1.0

#     solver_matrices = (sl_matrix, helmholtz_matrix, correction)

#     m0_s = (omega)^2 .* [minimum(m_coarse), mean(m_coarse), maximum(m_coarse)] .* (1 - im*(gamma_0+shift))
#     lis_solver = getLiS_solver(squeezeConvResult(laplacian_stencils[1]),squeezeConvResult(mass_stencils[1]), n_coarse, h_coarse, delta_coarse, m0_s, div.(n_coarse,2),div.(n_coarse,2))
    
#     values = collect((minimum(m_coarse)/maximum(m_coarse)):0.2:1) .* maximum(m_coarse)
#     # values = collect(minimum(m_coarse):0.1:maximum(m_coarse))
#     println(length(values))
#     m0_s_uniform = (omega)^2 .*  values .* (1 - im*(gamma_0+shift))
#     lis_solver_uniform = getLiS_solver(squeezeConvResult(laplacian_stencils[1]),squeezeConvResult(mass_stencils[1]), n_coarse, h_coarse, delta_coarse, m0_s_uniform, div.(n_coarse,2),div.(n_coarse,2))
    
#     SPAI_optimization_forPLOT(key, solver_matrices, laplacian_stencils[1], mass_stencils[1], n_coarse, h_coarse, lis_solver_uniform, lis_solver)
# end

# close(f)

