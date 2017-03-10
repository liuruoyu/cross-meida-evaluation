function SHparam = trainSpecH(X, dSpecH)
%
% Input
%   X = features matrix [Nsamples, Nfeatures], each point is a row
%   SHparam.nbits = number of bits (nbits do not need to be a multiple of 8)
%
%
% Spectral Hashing
% Y. Weiss, A. Torralba, R. Fergus. 
% Advances in Neural Information Processing Systems, 2008.

[Nsamples Ndim] = size(X);
SHparam.nbits = dSpecH;
% npca = Ndim;

% algo:
% 1) PCA
% npca = min(SHparam.nbits, Ndim);
% opts.disp = 0;
% [pc, l] = eigs(cov(X), npca,'la',opts);
% X = X * pc; % no need to remove the mean
pc = 1;
npca = Ndim;

% 2) fit uniform distribution
mn = prctile(X, 5);  mn = min(X)-eps;
mx = prctile(X, 95);  mx = max(X)+eps;


% 3) enumerate eigenfunctions
R=(mx-mn);
maxMode=ceil((SHparam.nbits+1)*R/max(R));

nModes=sum(maxMode)-length(maxMode)+1;
modes = ones([nModes npca]);
m = 1;
for i=1:npca
    modes(m+1:m+maxMode(i)-1,i) = 2:maxMode(i);
    m = m+maxMode(i)-1;
end
modes = modes - 1;
omega0 = pi./R;
omegas = modes.*repmat(omega0, [nModes 1]);
eigVal = -sum(omegas.^2,2);
[yy,ii]= sort(-eigVal);
modes=modes(ii(2:SHparam.nbits+1),:);


% 4) store paramaters
SHparam.pc = pc;
SHparam.mn = mn;
SHparam.mx = mx;
SHparam.modes = modes;
