
using PyPlot
using FFTW
using SparseArrays
using LinearAlgebra
close("all");

function solveConstHelm_temp(hop,n,b,m::ComplexF64)
    # h = [-1 2 -1]/(h^2) - [0 m 0];
    pad = Int(n/2); #n ; #div(n,2);
    bext = zeros(ComplexF64,n+2*pad);
    bext[(pad+1):(pad+n)].= b;
    hvec = zeros(ComplexF64, n+2*pad);
    hvec[1] = hop[2];
    hvec[2] = hop[3];
    hvec[n] = hop[1];
    hath = fft(hvec);
    hatb = fft(bext);
    hatu = hatb./hath;
    uext = ifft(hatu);
    u = uext[(pad+1):(pad+n)];
    return u;
    end

function solveConstHelm(hop,n,b,m::ComplexF64)
# h = [-1 2 -1]/(h^2) - [0 m 0];
pad = Int(n/2); #n ; #div(n,2);
bext = zeros(ComplexF64,n+2*pad);
bext[1:n] .= b;
# bext[(pad+1):(pad+n)].= b;
hvec = zeros(ComplexF64, n+2*pad);
hvec[1] = hop[2];
hvec[2] = hop[3];
hvec[n+2*pad] = hop[1];
hath = fft(hvec);
hatb = fft(bext);
hatu = hatb./hath;
uext = ifft(hatu);
#u = uext[(pad+1):(pad+n)];
u = uext[1:n];
return u;
end

function solveConstHelmNLA(hop,n,b,m::ComplexF64)
# h = [-1 2 -1]/(h^2) - [0 m 0];
pad = Int(2*n) # Int(n/2) # trunc(Int, n/16);
bext = zeros(ComplexF64,n+2*pad);
bext[(pad+1):(pad+n)].= b;

n_w_pad = n + 2*pad;            # n with padding.

gamma = zeros(n_w_pad);

gamma[1]   = -10 * sqrt(real(m))*(1.0/h);             # Sommerfeld radiation condition. Soft ends.
gamma[end] = -10 * sqrt(real(m))*(1.0/h);

# H = spdiagm(-1=>hop[1]*ones(n-1), 0=>hop[2]*ones(n), 1=>hop[3]*ones(n-1));
H = spdiagm(-1=>hop[1]*ones(n_w_pad-1), 0=>hop[2]*ones(n_w_pad), 1=>hop[3]*ones(n_w_pad-1));        # Like lap1D. 

H = H + spdiagm(0=>1im*gamma);

# H[1,end] = hop[1];            # Periodic BC.
# H[end,1] = hop[3];
# u = H\b;
u = H\bext;
u_internal = u[(pad+1):(pad+n)];
return u_internal;
end

n = 200;
# pad = 20;
# n = n+2*pad
h = 2.0/n;
m = (0.1/(h^2))*(1.0 + 1im*0.01)          # m = k^2. In this case it is constant through space (x).

# m is more or less k^2
hop = [-1 2 -1]/(h^2) - [0 m 0];

# In 2D: Lap2D - m*Id. This is after we add lines (62, 63) to each of the 1D Laplacians.

# b is f(x)
b = zeros(ComplexF64,n);
b[div(n,2)] = 1.0;
# In 2D: 1 in the middle of the matrix.

u2 = solveConstHelmNLA(hop,n,b,m::ComplexF64)

# println(norm(u2-u))
figure()
plot(real(u2))
# savefig("no_padding_w_gamma_constNLA.png")



# error()
u1 = solveConstHelm(hop,n,b,m);

# figure()
plot(real(u1))
show()
error()

figure()
plot(abs.(u1-u2))

show()


# hath = fft(hvec);
# hatb = fft(b);
# hatu = hatb./hath;
# u = ifft(hatu);
# println(typeof(u))
# plot(real(u))











