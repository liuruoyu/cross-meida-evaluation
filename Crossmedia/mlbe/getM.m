function m = getM(dim)
% computer the M matrix

M0 = ones(dim,dim);
ids = find(triu(M0)>0);
m = zeros(dim*dim,1);
m(ids) = 1;
% Mmat = diag(m);