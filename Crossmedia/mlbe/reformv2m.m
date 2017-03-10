function [M] = reformv2m(v, N)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reform a vector to an N x N symmetric matrix (diagonal included)
% v is a vector consisting of entries of right-upper triangle (length: N*(N+1)/2)
% procedure:
% first, pad v to right-upper part of an NxN matrix row-wise
% then, copy other entries
% example:
% v = 1:10, N=4
% M = [1 2 3 4]
%     [2 5 6 7]
%     [3 6 8 9]
%     [4 7 9 10]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    M1 = zeros(N,N);
    ind = rightupperindex(N,0);
    M1(ind) = v;
    M = M1+M1'-diag(diag(M1));
end