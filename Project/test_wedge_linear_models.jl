using Printf
include("auxiliary.jl")

n = 256

WedgeModel = wedge_grid_ratio(0.25, 1, n)
LinearModel = linear_grid_ratio(0.25, 1, n)

using PyPlot
figure()
# img = imshow(WedgeModel)
img = imshow(LinearModel)
colorbar(img)
show()