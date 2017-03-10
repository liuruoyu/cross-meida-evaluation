function [ind] = leftlowerindex_rowwise(N,k)
% get the linear index of left lower parts (k=1, excluding the diagonal) of an
% N x N matrix
    if nargin<2
        k = 1;
    end
    
    I = []; J = [];
    for j=1:N
        I = [I j+k:N];
        J = [J ones(1,N-j-k+1)*j];
    end
    ind = sub2ind_rowwise([N,N],I,J);
end

% function [ind] = leftlowerindex1(N)
%     ind = [];
%     for j=1:N
%         for i = j+1:N
%             ind(end+1) = sub2ind([N,N],i,j);
%         end
%     end
% end