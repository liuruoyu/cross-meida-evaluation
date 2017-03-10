function out = fromCellto4D(c, siz)
%FROMCELLTO4D convert a cell c, where each element is a sample into a 4D
%tensor where the last dimension is the sample index (rows, cols, channels,
%sample).
%
%   siz: a 3D vector for the image size

out = zeros(siz(1),siz(2),siz(3),length(c));
for i=1:length(c)
    out(:,:,:,i) = reshape(c{i},siz);
end