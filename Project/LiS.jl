
using PyPlot
using FFTW
using SparseArrays
using LinearAlgebra
close("all");

function solveConstHelm(hop,n,b,m::ComplexF64)
# h = [-1 2 -1]/(h^2) - [0 m 0];
pad = 0; #n ; #div(n,2);
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

function solveConstHelmNLA(hop,n,b,m::ComplexF64)
# h = [-1 2 -1]/(h^2) - [0 m 0];
pad = trunc(Int, n/2); #n ; #div(n,2);
bext = zeros(ComplexF64,n+2*pad);
bext[(pad+1):(pad+n)].= b;
gamma = zeros(n);
#gamma[1]   = -sqrt(real(m))*(1.0/h^2);             # Sommerfeld radiation condition.
#gamma[end] = -sqrt(real(m]))*(1.0/h^2);
H = spdiagm(-1=>hop[1]*ones(n-1), 0=>hop[2]*ones(n), 1=>hop[3]*ones(n-1));
H = H + spdiagm(0=>1im*gamma);
H[1,end] = hop[1];
H[end,1] = hop[3];
u = H\b;
u_internal = u[(pad+1):(pad+n)];
return u_internal;
end




n = 400;
# pad = 20;
# n = n+2*pad
h = 4.0/n;
m = (0.015/(h^2))*(1.0 + 1im*0.00)

# m is more or less k^2
hop = [-1 2 -1]/(h^2) - [0 m 0];



# b is f(x)
b = zeros(ComplexF64,n);
b[div(n,2)] = 1.0;


u2 = solveConstHelmNLA(hop,n,b,m::ComplexF64)



# println(norm(u2-u))
figure()
plot(real(u2))

# error()
u1 = solveConstHelm(hop,n,b,m);

figure()
plot(real(u1))

figure()
plot(abs.(u1-u2))

show()


# hath = fft(hvec);
# hatb = fft(b);
# hatu = hatb./hath;
# u = ifft(hatu);
# println(typeof(u))
# plot(real(u))









