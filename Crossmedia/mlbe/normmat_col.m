function [normv] = normmat_col(X)

normv = zeros(1,size(X,2));
for i = 1:size(X,2)
    normv(i) = norm(X(:,i));
end