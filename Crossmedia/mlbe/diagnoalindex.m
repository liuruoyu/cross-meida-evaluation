function [ind] = diagnoalindex(N)
% get the linear index of diagonal elements of an N x N matrix
    ind = sub2ind([N,N],[1:N],[1:N]);
%     ind = [];
%     for j=1:N
%         ind(end+1) = sub2ind(j,j);
%     end
end