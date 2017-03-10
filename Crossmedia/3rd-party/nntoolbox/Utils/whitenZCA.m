function [out, P, M] = whitenZCA(x,damping)
% applies zero-phase whitening

if nargin < 2
    damping = 0.1;
end

C = cov(x);
M = mean(x);
[V,D] = eig(C);
P = V * diag(sqrt(1./(diag(D) + damping))) * V';
out = bsxfun(@minus, x, M) * P;
