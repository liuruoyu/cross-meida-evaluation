function [xx,p,m] = whitenPCA (x)
% PCA based whitening. cov(y) = I

m = mean(x);
x = bsxfun(@minus, x, m);
[u,~,d] = princomp(x);
p = u/sqrt(diag(d));
xx = x*p;
% cov(xx)
