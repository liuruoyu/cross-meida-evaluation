function [normv] = normmat_row(X)

normv = zeros(size(X,1),1);
for i = 1:size(X,1)
    normv(i) = norm(X(i,:));
end