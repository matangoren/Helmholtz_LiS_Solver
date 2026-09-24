using Distributed
using ArgParse
using MAT
using JLD2


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--DD_n"
            help = "NumCells"
            arg_type = Int
            default = 1

        "--model"
            help = "model type"
            arg_type = String
            default = "tunnel"

		"--coarse_solver"
            help = "coarse solver type"
            arg_type = String
            default = "LiS"
    end

    return parse_args(ARGS, s)
end

args = parse_commandline()

DD_n   = args["DD_n"]
model = args["model"]
coarse_solver_type = args["coarse_solver"]

coarse_gmres_maxIter=20
coarse_gmres_restart=10
if coarse_solver_type == "GMRES_1_4"
	coarse_gmres_maxIter=1
	coarse_gmres_restart=4
elseif coarse_solver_type == "GMRES_1_10"
	coarse_gmres_maxIter=1
	coarse_gmres_restart=10
elseif coarse_solver_type == "GMRES_1_20"
	coarse_gmres_maxIter=1
	coarse_gmres_restart=20
end


# mat_filename_jdl = "$(@__DIR__)/dump/Marmousi/1024_4096_DD_4_16.jld2"
mat_filename_jdl = "$(@__DIR__)/dump/Marmousi/512_2048_DD_4_16.jld2"

# const mat_counter = Ref(0)
# const f = jldopen(mat_filename_jdl, "w")

# function save_submatrix(m, solver_matrices)

#     mat_counter[] += 1
#     name = "B_$(mat_counter[])"
# 	if mat_counter[] <= 64
# 		f[name] = (m, solver_matrices)
# 	end

# end


@everywhere begin
	coarse_times = [];
	include("./test_import.jl")
end




# 2D
# NumCells = [1,1];
# overlap = [0,0];
# if model == "marmousi"
# 	NumCells = [max(1,div(DD_n,4)),DD_n];
# else
# 	NumCells = [DD_n,DD_n];
# end

# if DD_n == 1
# 	overlap = [0,0]
# else
# 	overlap = [4,4].*4;
# end


# 3D
NumCells = [1,1,1];
overlap = [0,0,0];
NumCells = [DD_n,DD_n, DD_n];
if DD_n == 1
	overlap = [0,0,0]
else
	overlap = [4,4,4].*4;
end

# # Now add fresh workers

println("========= Workers $(workers()) =========")
println("========= Workers $(length(workers())) =========")

@everywhere begin
	coarse_times = [];
	include("./test_setup.jl")
end

@everywhere function getTest(n;model="linear")

	println(model)

	if model == "marmousi"
		A = readdlm("$(@__DIR__)/GeoModels/MarmousiVp_small.dat", data_type);
		m = expandModelNearest(A*1e-3, size(A),n);
		m = 1 ./ (m.^2)
		m = Matrix(transpose(m))
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

@everywhere function getSubParams(Hparam, M::RegularMesh,i::Array{Int64},NumCells::Array{Int64},Overlap::Array{Int64})
		subMesh   = getSubMeshOfCell(NumCells,Overlap,i,M);
		IIp, IIp_shape = getNodalIndicesOfCell(NumCells,Overlap,i,M.n);
		if length(M.n) == 2
			code 	  = [i[1]!=1 i[1]!=NumCells[1]; i[2]!=1 i[2]!=NumCells[2];];
		else
			code 	  = [i[1]!=1 i[1]!=NumCells[1]; i[2]!=1 i[2]!=NumCells[2]; i[3]!=1 i[3]!=NumCells[3];];
		end

		subgamma  = getABL(subMesh.n.+1,i[end]==1,Overlap.+4,2.0./(M.h[1]),code) #.+0.001*Hparam.omega;
		t = add_sommerfeld
		if Overlap[1] > overlap[1]
			subgamma = zeros(ComplexF64,IIp_shape)
			# t = false
		end
		Hparam = HelmholtzParam(subMesh,Hparam.gamma[IIp] + subgamma[:],Hparam.m[IIp],Hparam.omega,false,t)

		return Hparam;
	end






	sizes = [1] #[0,0,1,2,3]
if DD_n == 8
	sizes = [1,1,2,3]
end
if DD_n == 16
	sizes = [2,2,3]
end

if model == "marmousi"
	sizes = [3,4] #[2,2,3,4]
end

if model == "tunnel"
	sizes = [1]
end


# coarse_times = []
# coarse_iter = []
# coarse_err = []

sizes = [32,64,128]



for i in sizes
	# global coarse_times
	# global coarse_iter
	# global coarse_err
	
	n = i #256*(2^i)
	if model == "marmousi"
		n = [n,div(n,4)] .+ 1;
	else
		n = [n,n,n] .+ 1;
	end

	if model == "tunnel"
		n = [129,129,129]
	end


	# if device == gpu
	# 	result_filename = "$(join(n,'_'))_$(join(NumCells,'_'))_$(model)_$(coarse_solver_type)_gpu"
	# else
	# 	result_filename = "$(join(n,'_'))_$(join(NumCells,'_'))_$(model)_$(coarse_solver_type)_cpu"
	# end
	# coarse_times = []
	# coarse_iter = []
	# coarse_err = []
	println("############## $(model) ##############")

	m,Minv,h,omega,gamma,b = getTest(n;model=model)

	println("$(coarse_solver_type) --- $(model) --- $(size(m)) --- $(NumCells)")




	Hparam = HelmholtzParam(Minv,gamma,m,omega,false,add_sommerfeld)
	q = vec(b)



	if T == "Low"
		H, Lap, MM = GetHelmholtzOperator(Hparam,orderNeumannBC);
	else
		beta = 2/3
		if length(n) == 3
			beta = [1/3;1/2]
		end
		H, Lap, MM,Ms = GetHelmholtzOperatorHO(Hparam,beta,orderNeumannBC);
	end

	HrT = sparse(H')

	sl_matrix, helmholtz_matrix, correction = getHelmholtzMatrices(m, omega, gamma, h, fine_laplacian_stencil,fine_mass_stencil, alpha=alphas[end], add_sommerfeld=add_sommerfeld,BC=orderNeumannBC)

	AmatVec = v->vec(HelmholtzOperator(reshape(v, size(b)...,1,1), helmholtz_matrix,correction, h, fine_laplacian_stencil, fine_mass_stencil))

	solver = getConvGeometricMultigridSolver(;laplace_type=T,mass_type=T,level=level,gmres_maxIter=coarse_gmres_maxIter,gmres_restart=coarse_gmres_restart, interType=T, dim=length(n),add_sommerfeld=add_sommerfeld, orderNeumannBC=orderNeumannBC,coarse_solver_type=coarse_solver_type)
	DDparam = getDomainDecompositionParam(ComplexF64,Int64,Minv,NumCells,overlap,getNodalIndicesOfCell,solver);

	println("##### workers: $(workers()) #####")
	println("Performing Absorbing+Neumann Setup")
	println("NumCells = ",NumCells," overlap = ",overlap);


	# when using ConvGeometricMultigridSolver
	Ctor = DomainDecompositionOperatorConstructor{ComplexF64,Int64}(Hparam,getSubParams,identity,identity);
	if length(workers()) <= 1
		DDparam = setupDDSerial(Ctor,DDparam);
	else
		println("### IN PARALLEL ###")
		DDparam = setupDDParallel(Ctor,DDparam, workers());
	end

	println("Performing DD Solution with $(coarse_solver_type)")
	x = copy(q); 
	x[:] .= 0.0;

	# println("SETUP completed")
	# close(f);

	# return;

	x = solveLinearSystem!(AmatVec,q,x,DDparam)[1];
	# x = @time solveLinearSystem!(HrT,q,x,DDparam)[1];
	println("$(coarse_solver_type) --- $(model) --- $(size(m)) --- $(NumCells)")
	println("Outside error: ", norm(HrT'*x - q)/norm(q))
	println("Outside error: ", norm(vec(AmatVec(reshape(x,n...,1,1))) - q)/norm(q))
	# println(length(coarse_iter))
	
	# writedlm("$(@__DIR__)/Results/Linear/$(result_filename).csv", hcat(coarse_times,coarse_iter,coarse_err),',')
	
	# for i=10:10:120
	# 	close("all")
	# 	imshow(real.(reshape(x,n...)[:,:,i])); colorbar();
	# 	savefig("$(@__DIR__)/dump/slices/xy_$(i).png");

	# 	close("all")
	# 	imshow(real.(reshape(x,n...)[:,i,:])); colorbar();
	# 	savefig("$(@__DIR__)/dump/slices/xz_$(i).png");

	# 	close("all")
	# 	imshow(real.(reshape(x,n...)[i,:,:])); colorbar();
	# 	savefig("$(@__DIR__)/dump/slices/yz_$(i).png");
	# end
	
end

# sl_matrix, helmholtz_matrix, correction = getHelmholtzMatrices(m, omega, gamma, h, fine_laplacian_stencil,fine_mass_stencil, alpha=alphas[end], add_sommerfeld=add_sommerfeld,BC=orderNeumannBC)

# AmatVec = v->vec(HelmholtzOperator(reshape(v, size(b)...,1,1), helmholtz_matrix,correction, h, fine_laplacian_stencil, fine_mass_stencil))

# Hparam = HelmholtzParam(Minv,gamma,m,omega,false,add_sommerfeld)
# q = vec(b)

# if T == "Low"
# 	H, Lap, MM = GetHelmholtzOperator(Hparam,orderNeumannBC);
# else
# 	beta = 2/3
# 	if length(n) == 3
# 		beta = [1/3;1/2]
# 	end
# 	H, Lap, MM,Ms = GetHelmholtzOperatorHO(Hparam,beta,orderNeumannBC);
# end

# # H = GetHelmholtzOperator(Hparam)[1
# HrT = sparse(H')
# # HrT = H

# #Shift = GetHelmholtzShiftOP(m, omega,0.1);
# #Shift = convert(SparseMatrixCSC{ComplexF64,spIndType},Shift);





# # solver = getJuliaSolver()
# solver = getConvGeometricMultigridSolver(;laplace_type=T,mass_type=T,level=level, interType=T, dim=length(n),add_sommerfeld=add_sommerfeld, orderNeumannBC=orderNeumannBC)

# DDparam = getDomainDecompositionParam(ComplexF64,Int64,Minv,NumCells,overlap,getNodalIndicesOfCell,solver);

# println("##### workers: $(workers()) #####")
# println("Performing Absorbing+Neumann Setup")
# println("NumCells = ",NumCells," overlap = ",overlap);

# @everywhere function getSubParams(Hparam, M::RegularMesh,i::Array{Int64},NumCells::Array{Int64},Overlap::Array{Int64})
# 	subMesh   = getSubMeshOfCell(NumCells,Overlap,i,M);
# 	IIp, IIp_shape = getNodalIndicesOfCell(NumCells,Overlap,i,M.n);
# 	if length(M.n) == 2
# 		code 	  = [i[1]!=1 i[1]!=NumCells[1]; i[2]!=1 i[2]!=NumCells[2];];
# 	else
# 		code 	  = [i[1]!=1 i[1]!=NumCells[1]; i[2]!=1 i[2]!=NumCells[2]; i[3]!=1 i[3]!=NumCells[3];];
# 	end

# 	subgamma  = getABL(subMesh.n.+1,i[end]==1,Overlap.+4,2.0./(M.h[1]),code) #.+0.001*Hparam.omega;
# 	t = add_sommerfeld
# 	if Overlap[1] > overlap[1]
# 		subgamma = zeros(ComplexF64,IIp_shape)
# 		# t = false
# 	end
#     Hparam = HelmholtzParam(subMesh,Hparam.gamma[IIp] + subgamma[:],Hparam.m[IIp],Hparam.omega,false,t)

# 	return Hparam;
# end


# # getDDMass = (ddp,hp,i)->(0.0.*Vector(diag(GetHelmholtzShiftOP(hp.m,0.0,0.0))));

# # function getDirichletMassNodalMesh(DDparam::DomainDecompositionParam,problem_param::HelmholtzParam,i::Array{Int64})
# # 	d = getDirichletMassNodal(DDparam.numDomains,DDparam.overlap,i,DDparam.Mesh.n);
# # 	d.*=(0.1*4.0)/prod(DDparam.Mesh.h);
# # 	return d;
# # end
# # getDDMass = getDirichletMassNodalMesh

# # in comment: working example (From Eran's code)
# # Ctor = DomainDecompositionOperatorConstructor{ComplexF64,Int64}(Hparam,getSubParams,GetHelmholtzOperator,identity);
# # DDparam = setupDDSerial(HrT,DDparam);

# # when using ConvGeometricMultigridSolver
# Ctor = DomainDecompositionOperatorConstructor{ComplexF64,Int64}(Hparam,getSubParams,identity,identity);
# if length(workers()) <= 1
# 	DDparam = setupDDSerial(Ctor,DDparam);
# else
# 	println("### IN PARALLEL ###")
# 	DDparam = setupDDParallel(Ctor,DDparam, workers());
# end

# println("Performing DD Solution with GMRES")
# x = copy(q); 
# x[:] .= 0.0;


# x = solveLinearSystem!(AmatVec,q,x,DDparam)[1];
# # x = @time solveLinearSystem!(HrT,q,x,DDparam)[1];
# println("$(n) --- $(NumCells)")
# println("Outside error: ", norm(HrT'*x - q)/norm(q))
# println("Outside error: ", norm(vec(AmatVec(reshape(x,n...,1,1))) - q)/norm(q))


# writedlm("$(@__DIR__)/Results/$(result_filename).dat", coarse_times)

# if length(n) == 2
# 	figure()
# 	imshow(reshape(abs.(real(HrT'*x - q)),n...)); colorbar();
# 	figure()
# 	imshow(reshape((real(x)),n...)); colorbar();
# end
# x, = solveDDSerial(HrT,q,zeros(ComplexF64,size(q)),DDparam,20);
# println("Outside error: ", norm(HrT'*x - q)/norm(q))