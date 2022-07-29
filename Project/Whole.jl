A  = sprandn(100,100,.1) + SparseMatrixCSC(10.0I, 100, 100)
n  = size(A,2)
D  = diag(A)
M2 = x -> Vector(D.\x)
rhs = randn(100)
tol = 1e-6;

# test printing and behaviour for early stopping
xtt = fgmres(A,rhs ,3,tol=1e-12,maxIter=3,out=2,storeInterm=true)

