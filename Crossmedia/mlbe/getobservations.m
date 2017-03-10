function [O] = getobservations(nrow, ncol, p)
% approach 1: element-wise generation
%     O = sparse(zeros(nrow,ncol));
%     for j = 1:ncol
%         for i = 1:nrow
%             O(i,j) = randbinom(p, 1);
%         end
%     end
% approach 2: rand matrix + thresholding
    O = spalloc(nrow,ncol,floor(nrow*ncol*p)+ncol);
    for c = 1:ncol
        OV = rand(nrow, 1);
        O(:,c) = sparse(bsxfun(@le, OV, p));
    end
end