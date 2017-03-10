function [label, energy, medoid] = kmedoids(X,m)
% X: d x n data matrix
% m: k (1 x 1) or label (1 x n, 1<=label(i)<=k) or medoid index(1 x k)
% Written by Mo Chen (mochen@ie.cuhk.edu.hk). March 2009.
%% initialization
n = size(X,2);
s = length(m);
if s == 1
    k = m;
    medoid = randsample(n,k);
elseif s < n
    k = s;
    medoid = m;
elseif s == n
    k = max(m);
    label = m;
    medoid = zeros(k,1);
else
    error('ERROR: m is not valid.');
end

D = sqdistance(X);
if s ~= n
    [val,label] = min(D(medoid,:));
end
%% main algorithm
last = 0;
while any(label ~= last)
    for i = 1:k
        idx = (label==i);
        [~,tmp] = min(sum(D(idx,idx),1));
        idx = find(idx);
        medoid(i) = idx(tmp);
    end  
    last = label;
    [val,label] = min(D(medoid,:));
end
energy = sum(val);
%% simpler but slower
% last = 0;
% S = repmat(1:k,n,1);
% while any(label ~= last)
%     [val,medoid] = min(D*bsxfun(@eq,S,label'));
%     last = label;
%     [val,label] = min(D(medoid,:));
% end
% energy = sum(val);