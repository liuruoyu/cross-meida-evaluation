function [I,J] = ind2sub_rowwise(siz,ind)
    I = floor((ind-1)/siz(2))+1;
    J = ind - (I-1)*siz(2);
end