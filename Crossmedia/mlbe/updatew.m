function [W] = updatew(U, S, hyper)
% this function can be used for updating W^{x} or W^{y}

do_sampling = 0;
if do_sampling
    % subsampling U and S for practical use
    randinds = randperm(size(U,2));
    U = U(:,randinds(1:hyper.refsize));
    S = S(randinds(1:hyper.refsize), randinds(1:hyper.refsize));
end
% learn W
[K,N] = size(U);
A = kron(U,U);
indexOfRU = rightupperindex_rowwise(K,0);
A1 = A(indexOfRU, :);
[IRU, JRU] = ind2sub_rowwise([K,K], indexOfRU);
indexOfRUA = sub2ind_rowwise([K,K], JRU, IRU);
A2 = A(indexOfRUA, :);
indictor = zeros(K*(K+1)/2,1);
indictor(diagindex_rowwise([K,K],indexOfRU)) = 1;
A3 = diag(indictor)*A1;
Ah = A1+A2-A3;

% indU = leftlowerindex(size(U,2));
indU = rightupperindex_rowwise(size(U,2));
s = S(indU');
Ah = Ah(:,indU);
% indW = diagnoalindex(size(U,1));
% m = zeros(K*K,1); m(indW') = 1;
% w = inv(Ah*Ah'+(hyper.theta/(4*hyper.phi))*(eye(K*K)+diag(m)))*Ah*s;
% W = reshape(w,K,K);
% W = (W+W')/2;
% w = inv(Ah*Ah'+(hyper.theta/(4*hyper.phi))*(size(Ah,1)))*Ah*s;
w = (Ah*Ah'+(hyper.theta/(2*hyper.phi))*(eye(size(Ah,1))))\(Ah*s);
W = reformv2m(w,K);
end