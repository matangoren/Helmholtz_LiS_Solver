using Printf
include("auxiliary.jl")

n = 256

WedgeModel = wedge_grid_ratio(0.5, 1, n)

using PyPlot
figure()
img = imshow(WedgeModel)
colorbar(img)
show()