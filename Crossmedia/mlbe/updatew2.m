function [W] = updatew2(U, S, hyper)
% this is new version based on clearer interpretation (1-16-2012)
% this function can be used for updating W^{x} or W^{y}

% learn W
[K,N] = size(U);
A = kron(U',U');
m1 = getM(K);
m2 = getM(N);
% indexOfRU = rightupperindex_rowwise(K,0);
% A1 = A(indexOfRU, :);'
% [IRU, JRU] = ind2sub_rowwise([K,K], indexOfRU);
% indexOfRUA = sub2ind_rowwise([K,K], JRU, IRU);
% A2 = A(indexOfRUA, :);
% indictor = zeros(K*(K+1)/2,1);
% indictor(diagindex_rowwise([K,K],indexOfRU)) = 1;
% A3 = diag(indictor)*A1;
% Ah = A1+A2-A3;

% indU = leftlowerindex(size(U,2));
% indU = rightupperindex_rowwise(size(U,2));
% s = S(indU');
% Ah = Ah(:,indU);
% indW = diagnoalindex(size(U,1));
% m = zeros(K*K,1); m(indW') = 1;
% w = inv(Ah*Ah'+(hyper.theta/(4*hyper.phi))*(eye(K*K)+diag(m)))*Ah*s;
% W = reshape(w,K,K);
% W = (W+W')/2;
% w = inv(Ah*Ah'+(hyper.theta/(4*hyper.phi))*(size(Ah,1)))*Ah*s;
Ah = A(m2>0,:);
Sv = S(:);
Sh = Sv(m2>0);
noise = 1e-6;
w = (noise*eye(K*K) + Ah'*Ah + (hyper.theta/hyper.phi)*diag(m1))\(Ah'*Sh);
W = reshape(w,K,K);
W = (W+W')/2;
end