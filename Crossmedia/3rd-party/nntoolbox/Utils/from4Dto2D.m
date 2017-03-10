function y = from4Dto2D(x)
%FROM2DTO4D converts the 2D matrix x into the 4D tensor y where each
%element is reshaped according to siz.
%The number of rows in x is the number of samples and hence the resulting
%matrix will be as follows:
%   (N, M) -> (siz(1), siz(2), siz(3), N)    where M = prod(siz)

y = permute(reshape(x,size(x,1)*size(x,2)*size(x,3),size(x,4)),[2 1]);
