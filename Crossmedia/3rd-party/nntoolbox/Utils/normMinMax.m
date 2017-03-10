function out = normMinMax(data, minv, maxv, type, mindata, maxdata)
% Performs normalization of the data in order to have all values bounded by
% minv and maxv.
%   type: rows,cols,all specifies which normalization has to be used

%   This file is part of netlabExtensions.
%   July 2011
%   Jonathan Masci <jonathan@idsia.ch>

if nargin < 4 || isempty(type)
    type = 'all';
end

switch lower(type)
    case {'rows'}
        if nargin < 5
            out = bsxfun(@rdivide, bsxfun(@minus, data, min(data,[],2)), max(data,[],2) - min(data,[],2)) * (maxv - minv) + minv;
        else
            out = bsxfun(@rdivide, bsxfun(@minus, data, mindata), maxdata - mindata) * (maxv - minv) + minv; 
        end
    case {'cols'}
        if nargin < 5
            out = bsxfun(@rdivide, bsxfun(@minus, data, min(data,[],1)), max(data,[],1) - min(data,[],1)) * (maxv - minv) + minv;
        else
            out = bsxfun(@rdivide, bsxfun(@minus, data, mindata), maxdata - mindata) * (maxv - minv) + minv;
        end
    case {'all'}
        if nargin < 5
            out = (data - min(data(:)))/(max(data(:) - min(data(:)))) * (maxv - minv) + minv;
        else
            out = (data - mindata)/(maxdata - mindata) * (maxv - minv) + minv;
        end
    otherwise
        disp('Wrong argument provided for type');
end
