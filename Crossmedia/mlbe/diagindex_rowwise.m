function [ind] = diagindex_rowwise(siz,ind)
% find index of diagonals of ind
    I = floor((ind-1)/siz(2))+1;
    J = ind - (I-1)*siz(2);
    ind = find(I==J);
end