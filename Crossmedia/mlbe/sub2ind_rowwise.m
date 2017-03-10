function [ind] = sub2ind_rowwise(siz,I,J)
    ind = siz(2)*(I-1)+J;
end