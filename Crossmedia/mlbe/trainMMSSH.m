function [Wx, Wy] = trainMMSSH(X, Y, dMMSSH)
%
% Input
%   X = features matrix [Nfeatures, Nsamples]
%   Y = features matrix [Nfeatures, Nsamples]
%   SHparam.nbits = number of bits (nbits do not need to be a multiple of 8)
%

[Ndimx Nsamples] = size(X); [Ndimy] = size(Y,1);

% algorithm
% option = struct('disp',0);
Wx = zeros(Ndimx, dMMSSH); 
Wy = zeros(Ndimy, dMMSSH);
vw = ones(Nsamples,1)/Nsamples;
for m = 1:dMMSSH
    % fast way to construct C
    X1 = bsxfun(@times, X, sqrt(vw(:)'));
    Y1 = bsxfun(@times, Y, sqrt(vw(:)'));
    C = X1*Y1' / sum(vw);
    
    % slow way to construct C
%     C = X*diag(vw)*Y';
    if issparse(C)
        [U, S, V] = svds(C,1);
    else
        [U, S, V] = svd(C,'econ');
        [~,id] = max(diag(S));
        U = U(:,id); V = V(:,id);
    end
    
    % update weights
    hx = sign(U'*X); hy = sign(V'*Y); h = hx.*hy;
    vw = vw.*exp(-h)';   vw = vw/sum(vw);
    % update Wx, Wy
    Wx(:,m) = U; 
    Wy(:,m) = V;
end
