"""
The Gauss Seidel method
"""
function GS(A,b,x,epsilon,n_iter)
r = b - A*x; n0 = norm(r);
LD = tril(A);
for k=1:n_iter
    x = x + LD\r;
    r = b - A*x;
    nr = norm(r);
    if nr/n0 < epsilon
    break;
    end
end
return x,norms;
end