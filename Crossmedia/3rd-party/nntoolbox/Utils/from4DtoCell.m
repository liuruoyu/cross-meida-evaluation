function c = from4DtoCell(x)
%FROM4DTOCELL converts a 4D tensor x into a cell array where each element
%of the 4th dim is a cell item. The content is made of the first 3 dims.

%   This file is part of netlabExtensions.
%   September 2011
%	Jonathan Masci <jonathan@idsia.ch>

c = cell(size(x,4),1);
for i=1:size(x,4)
    c{i} = x(:,:,:,i);
end