function Kxy = rbfkernel(X, Y, sigma,modename)
% RBFKERNEL		Calculate RBF kernel between data matrix X and Y
%
%	DESCRIPTION
%	Calculate RBF kernel w.r.t. each data point in matrix X and Y. The
%	calculatioin is K(x,y) = exp(-1/sigma^2 * norm(x-y)).
%
%	INPUT
%		X: K x N matrix where each column vector is a data point
%		Y: K x M matrix where each column vector is a data point
%		SIGMA: The parameter in the kernel calculation
%
%	OUTPUT
%		K: N by M matrix, each entry K(i,j) is the kernel function on data
%		points X(i,:) and Y(j,:)
%


switch modename
    case 'normal1'
        N = size(X,2); M = size(Y, 2);
        Kxy = (-2*X'*Y+repmat(diag(X'*X),1,M)+repmat(diag(Y'*Y)',N,1));      
    case 'normal2'
        N = size(X,2); M = size(Y, 2);
        Kxy = sum((X.^2), 1)' * ones(1, M) + ones(N, 1) * sum((Y.^2),1) - 2.*(X'*(Y));
    case 'fast'
        Kxy = -2*X'*Y;
        Kxy = bsxfun(@plus, Kxy, sum(X.^2,1)');
        Kxy = bsxfun(@plus, Kxy, sum(Y.^2,1));
    otherwise
        disp('The mode name is not supproted.');
end
Kxy = exp(-0.5 * Kxy / sigma^2);

