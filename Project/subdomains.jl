include("auxiliary.jl");


function wedge_9_subdomains(bottom, top, n)
    ratios = wedge_grid_ratio(bottom, top, 3 * n)
    ratios11 = ratios[1:n , 1:n];
    ratios12 = ratios[1:n , n+1:2*n];
    ratios13 = ratios[1:n , 2*n+1:3*n];
    ratios21 = ratios[n+1:2*n , 1:n];
    ratios22 = ratios[n+1:2*n ,n+1:2*n];
    ratios23 = ratios[n+1:2*n , 2*n+1:3*n];
    ratios31 = ratios[2*n+1:3*n , 1:n];
    ratios32 = ratios[2*n+1:3*n , n+1:2*n];
    ratios33 = ratios[2*n+1:3*n , 2*n+1:3*n];

	return ratios11, ratios12, ratios13, ratios21, ratios22, ratios23, ratios31, ratios32, ratios33;
end

function wedge_16_subdomains(bottom, top, n)
    ratios = wedge_grid_ratio(bottom, top, 4*n)
    k = n;
    ratios11 = ratios[1:k , 1:k];
    ratios12 = ratios[1:k , k+1:2*k];
    ratios13 = ratios[1:k , 2*k+1:3*k];
    ratios14 = ratios[1:k , 3*k+1:4*k];
    ratios21 = ratios[k+1:2*k , 1:k];
    ratios22 = ratios[k+1:2*k ,k+1:2*k];
    ratios23 = ratios[k+1:2*k , 2*k+1:3*k];
    ratios24 = ratios[k+1:2*k , 3*k+1:4*k];
    ratios31 = ratios[2*n+1:3*n , 1:n];
    ratios32 = ratios[2*n+1:3*n , n+1:2*n];
    ratios33 = ratios[2*n+1:3*n , 2*n+1:3*n];
    ratios34 = ratios[2*n+1:3*n , 3*n+1:4*n];
    ratios41 = ratios[3*n+1:4*n , 1:n];
    ratios42 = ratios[3*n+1:4*n , n+1:2*n];
    ratios43 = ratios[3*n+1:4*n , 2*n+1:3*n];
    ratios44 = ratios[3*n+1:4*n , 3*n+1:4*n];

	return ratios11, ratios12, ratios13, ratios14, ratios21, ratios22, ratios23, ratios24, ratios31, ratios32, ratios33, ratios34, ratios41, ratios42, ratios43, ratios44;
end



# Not working since we need a matrix of matrices
###############################################
# function wedge_subdomains(bottom, top, n, k)
#     wedge = zeros(k,k);
#     ratios = wedge_grid_ratio(bottom, top, k * n)
#     for i = 0:k-1
#         for j = 0:k-1
#             wedge[i,j] = ratios[1+i*n:(i+1)*n , 1+j*n:(j+1)*n];
#         end
#     end

# 	return wedge;
# end