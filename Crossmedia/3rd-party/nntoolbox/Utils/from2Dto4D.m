function y = from2Dto4D(x, siz)
%FROM2DTO4D converts the 2D matrix x into the 4D tensor y where each
%element is reshaped according to siz.
%The number of rows in x is the number of samples and hence the resulting
%matrix will be as follows:
%   (N, M) -> (siz(1), siz(2), siz(3), N)    where M = prod(siz)

%   This file is part of netlabExtensions.
%   June 2011
%	Jonathan Masci <jonathan@idsia.ch>

y = permute(reshape(x,size(x,1),siz(1),siz(2),siz(3)),[2 3 4 1]);
