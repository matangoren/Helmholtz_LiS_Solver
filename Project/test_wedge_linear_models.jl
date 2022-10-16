using Printf
include("auxiliary.jl")
include("subdomains.jl")

n = 256

WedgeModel = wedge_grid_ratio(0.25, 1, n);
LinearModel = linear_grid_ratio(0.25, 1, n);

wedge11, wedge12, wedge13, wedge21, wedge22, wedge23, wedge31, wedge32, wedge33 = wedge_9_subdomains(0.25, 1, n);

using PyPlot
figure()
img = imshow(WedgeModel)
colorbar(img)

figure()
img = imshow(LinearModel)
colorbar(img)

# Plot all subdomains together
figure()

subplot(3,3,1)
img = imshow(wedge11)

subplot(3,3,2)
img = imshow(wedge12)

subplot(3,3,3)
img = imshow(wedge13)

subplot(3,3,4)
img = imshow(wedge21)

subplot(3,3,5)
img = imshow(wedge22)

subplot(3,3,6)
img = imshow(wedge23)

subplot(3,3,7)
img = imshow(wedge31)

subplot(3,3,8)
img = imshow(wedge32)

subplot(3,3,9)
img = imshow(wedge33)

show()

