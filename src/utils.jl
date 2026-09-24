# Assuming test_import is including this file - device is defined there.

function JacobiDiagonalCorrection2D(shape, stencil; code=nothing)
    if code == nothing
        code = ones(Bool, 2*length(shape))
    end
    s = div(size(stencil,1),2)
    correction = zeros(shape)
    mid = stencil[s+1,s+1]
 
    if code[1] # top row
        correction[1,:] .+= sum(stencil[1:s,s+1])
        if code[3] # top-left corner
            correction[1,1] += sum(stencil[1:s,1:s])
        end
        if code[4] # top-right corner
            correction[1,end] += sum(stencil[1:s,end-s+1:end])
        end
    end
    if code[2] # bottom row
        correction[end,:] .+= sum(stencil[end-s+1:end,s+1])
        if code[3] # bottom-left corner
            correction[end,1] += sum(stencil[end-s+1:end,1:s])
        end
        if code[4] # bottom-right corner
            correction[end,end] += sum(stencil[end-s+1:end,end-s+1:end])
        end
    end
    if code[3] # left column
        correction[:,1] .+= sum(stencil[s+1,1:s])
    end
    if code[4] # right column
        correction[:,end] .+= sum(stencil[s+1,end-s+1:end])
    end

    return correction
end


function JacobiDiagonalCorrection3D(shape, stencil; code=nothing)
    if code == nothing
        code = ones(Bool, 2*length(shape))
    end
    s = div(size(stencil,1),2)
    correction = zeros(shape)
    mid = stencil[s+1,s+1,s+1]

    if code[1] # top layer
        correction[1,:,:] .+= sum(stencil[1:s,s+1,s+1])
        if code[3] && code[5]
            correction[1,1,1] += sum(stencil[1:s,1:s,1:s])
        end
        if code[3] && code[6]
            correction[1,1,end] += sum(stencil[1:s,1:s,end-s+1:end])
        end
        if code[4] && code[5]
            correction[1,end,1] += sum(stencil[1:s,end-s+1:end,1:s])
        end
        if code[4] && code[6]
            correction[1,end,end] += sum(stencil[1:s,end-s+1:end,end-s+1:end])
        end
    end
    if code[2] # bottom layer
        correction[end,:,:] .+= sum(stencil[end-s+1:end,s+1,s+1])
        if code[3] && code[5]
            correction[end,1,1] += sum(stencil[end-s+1:end,1:s,1:s])
        end
        if code[3] && code[6]
            correction[end,1,end] += sum(stencil[1:s,1:s,end-s+1:end])
        end
        if code[4] && code[5]
            correction[end,end,1] += sum(stencil[end-s+1:end,end-s+1:end,1:s])
        end
        if code[4] && code[6]
            correction[end,end,end] += sum(stencil[end-s+1:end,end-s+1:end,end-s+1:end])
        end
    end
    if code[3] # left layer
        correction[:,1,:] .+= sum(stencil[s+1,1:s,s+1])
    end
    if code[4] # right layer
        correction[:,end,:] .+= sum(stencil[s+1,end-s+1:end,s+1])
    end
    if code[5] # front layer
        correction[:,:,1] .+= sum(stencil[s+1,s+1,1:s])
    end
    if code[6] # back layer
        correction[:,:,end] .+= sum(stencil[s+1,s+1,end-s+1:end])
    end

    return correction
end

# function getJacobiDiagonal(h, matrix, correction, laplacian_stencil, mass_stencil)
#     lap_mid = div(size(laplacian_stencil,1),2)+1
#     mass_mid = div(size(mass_stencil,1),2)+1
#     if length(h) == 2
#         D = laplacian_stencil[lap_mid, lap_mid]/(h[1]^2) .+ mass_stencil[mass_mid,mass_mid].*matrix #(-matrix .+ correction[1])
#     else
#         D = laplacian_stencil[lap_mid, lap_mid,lap_mid]/(h[1]^2) .+ mass_stencil[mass_mid,mass_mid,lap_mid].*matrix #(-matrix .+ correction[1])
#     end

#     D .+= correction[1] .+ correction[2].*matrix
    
#     return D
# end

function getHelmholtzMatrices(m, omega, gamma, h, laplacian_stencil, mass_stencil; alpha=0.1, add_sommerfeld=true, BC=1,code=nothing)
    sommerfeld = zeros(ComplexF64, size(m))
    if code == nothing
        code = ones(Bool, 2*length(h))
    end
    if add_sommerfeld
        if ndims(sommerfeld) == 2
            if code[1]
                sommerfeld[1,:]  .+= ((BC/h[1]) .* im*omega*sqrt.(m[1,:]))
            end
            if code[2]
                sommerfeld[end,:]  .+=  ((BC/h[1]) .* im*omega*sqrt.(m[end,:]))
            end
            if code[3]
                sommerfeld[:,1]  .+=   ((BC/h[2]) .* im*omega*sqrt.(m[:,1]))
            end
            if code[4]
                sommerfeld[:,end] .+=  ((BC/h[2]) .* im*omega*sqrt.(m[:,end]))
            end
        else
            if code[1]
                sommerfeld[1,:,:]  .+= ((BC/h[1]) .* im*omega*sqrt.(m[1,:,:]))
            end
            if code[2]
                sommerfeld[end,:,:]  .+= ((BC/h[1]) .* im*omega*sqrt.(m[end,:,:]))
            end
            if code[3]
                sommerfeld[:,1,:]  .+= ((BC/h[2]) .* im*omega*sqrt.(m[:,1,:]))
            end
            if code[4]
                sommerfeld[:,end,:] .+= ((BC/h[2]) .* im*omega*sqrt.(m[:,end,:]))
            end
            if code[5]
                sommerfeld[:,:,1]  .+=  ((BC/h[3]) .* im*omega*sqrt.(m[:,:,1]))
            end
            if code[6]
                sommerfeld[:,:,end] .+= ((BC/h[3]) .* im*omega*sqrt.(m[:,:,end]))
            end
        end
    end

  
    if length(h) == 2
        laplacian_correction = JacobiDiagonalCorrection2D(size(m), laplacian_stencil ./ (h[1]^2); code=code)
        mass_correction = JacobiDiagonalCorrection2D(size(m),mass_stencil; code=code)
    else
        laplacian_correction = JacobiDiagonalCorrection3D(size(m), laplacian_stencil ./ (h[1]^2); code=code)
        mass_correction = JacobiDiagonalCorrection3D(size(m),mass_stencil; code=code)
    end

    helmholtz_matrix = -(omega.^2).*(m).*(1.0.-1im.*gamma/omega) .+ sommerfeld
    sl_matrix = -(omega.^2).*(m).*(1.0.-1im.*(gamma/omega .+ alpha)) .+ sommerfeld

    correction = [laplacian_correction,mass_correction]
    # sl_matrix_Jacobi_D = getJacobiDiagonal(h, sl_matrix, correction, laplacian_stencil, mass_stencil)

    return sl_matrix, helmholtz_matrix, correction
end

function squeezeConvResult(x)
    n_dims = ndims(x)

    # Specify the last two dimensions to drop
    dims_to_drop = (n_dims - 1, n_dims)

    # Drop the dimensions
    return dropdims(x, dims=dims_to_drop)
end

function get_coarse_m(m, level, Rs_Grid)
    m_coarse = reshape(copy(m), size(m)...,1,1)
    for i=(level-1):-1:1
        m_coarse = Rs_Grid[i](m_coarse)
    end
    
    return squeezeConvResult(m_coarse)
end


function getMatrices(Rs_grid, m, omega, gamma, h,laplacian_stencils, mass_stencils; level=3, alphas=0.1, add_sommerfeld=true, BC=1, code=nothing)
    if !isa(alphas,AbstractVector)
        alphas = ones(data_type,level).*0.1
    end
    matrices = Vector{}(undef, level)
    m = copy(m)
    gamma = copy(gamma)
    h = copy(h)
    matrices[level] = getHelmholtzMatrices(m, omega, gamma, h, laplacian_stencils[level], mass_stencils[level]; alpha=alphas[level], add_sommerfeld=add_sommerfeld, BC=BC, code=code)
    for i=(level-1):-1:1
        R_grid = Rs_grid[i]
        m = squeezeConvResult(R_grid(reshape(m,size(m)...,1,1)))
        gamma = squeezeConvResult(R_grid(reshape(gamma,size(gamma)...,1,1)))
        h.*=2
        matrices[i] = getHelmholtzMatrices(m, omega, gamma, h, laplacian_stencils[i], mass_stencils[i]; alpha=alphas[i], add_sommerfeld=add_sommerfeld, BC=BC, code=code)
    end
    return matrices
end

