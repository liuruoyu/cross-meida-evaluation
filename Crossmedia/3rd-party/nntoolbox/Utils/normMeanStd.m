function [out M S] = normMeanStd(data, type, damp, M, S)
%function  [out M S] = normMeanStd(data, type, damp, M, S)
%
% normalizes the data to zero mean unit variance and gives opitionally back
% the mean and the std of the data (which HAS to be used to normalize another
% data set in the same way) and takes damp, mean and standard deviatio as optinal
% inputs. The damp parameter is added to the variance in order to avoid
% division by small numbers...
% normalize in order to have data with 0 mean and 1 std (if damp = 0)

% if damp is not provided then it is set to 1 by default

if nargin < 3
    damp = 1;
end


switch lower(type)
    case {'rows'}
        if nargin <= 3
            M = mean(data,2);
            S = sqrt(var(data,[],2)+damp);
        end
        out = bsxfun(@rdivide, bsxfun(@minus, data, M), S);
    case {'cols'}
        if nargin <= 3
            M = mean(data,1);
            S = sqrt(var(data,[],1)+damp);
        end
        out = bsxfun(@rdivide, bsxfun(@minus, data, M), S);
    case {'all'}
        if nargin <= 3
            M = mean(data(:));
            S = sqrt(var(data(:))+damp);
        end
        out = bsxfun(@rdivide, bsxfun(@minus, data, M), S);
    otherwise
        disp('Wrong argument provided for type: rows, cols, all');
end
