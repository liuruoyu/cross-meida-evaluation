function out = normL2(data, type, batchSize)
% Performs normalization using L2 norm.
%   type: rows,cols

%   This file is part of netlabExtensions.
%   Nov 2011
%   Jonathan Masci <jonathan@idsia.ch>


switch lower(type)
    case {'rows'}
        nBatches = ceil(size(data,1)/batchSize);
        out = zeros(size(data));
        for i=1:nBatches
            from = (i-1)*batchSize + 1;
            to = min(i*batchSize,size(data,1));
            
            out(from:to,:) = bsxfun(@rdivide, data(from:to,:), sqrt(sum(data(from:to,:).^2, 2)));
        end
    case {'cols'}
        nBatches = ceil(size(data,2)/batchSize);
        out = zeros(size(data));
        for i=1:nBatches
            from = (i-1)*batchSize + 1;
            to = min(i*batchSize,size(data,2));
            
            out(:,from:to) = bsxfun(@rdivide, data(:,from:to), sqrt(sum(data(:,from:to).^2, 1)));
        end
    otherwise
        disp('Wrong argument provided for type');
end
