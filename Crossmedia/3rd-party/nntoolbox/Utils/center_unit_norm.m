function [Y, X_m] = center_unit_norm(X, type, datatype, X_m)

switch lower(type)
    case {'rows'}
        % Every row is a sample
        if nargin < 4
            X_m = mean(X);
        end
        X = bsxfun(@plus, feval(datatype, X), - X_m);
        Y = feval(datatype, normL2(double(X), 'rows', 1000));
   
    case {'cols'}
        if nargin < 4
            X_m = mean(X, 2);
        end
        X = bsxfun(@plus, feval(datatype, X), - X_m);
        Y = feval(datatype, normL2(double(X), 'cols', 1000));

    otherwise
        disp('Wrong argument provided for type');
end
