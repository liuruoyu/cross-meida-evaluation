classdef lossMCCE < ILossFun
    properties
        toDeriv = false;
    end
    
    methods
        function obj = lossMCCE()
        end
        
        function y = derivOut(obj)
            y = obj.toDeriv;
        end
        
        function [cost, deltas] = apply(obj, x, y)
            datatype = class(x);
            x = double(x);
            y = double(y);
            
            deltas = feval(datatype, x - y);
            % Ensure that log(x) is computable
            x(x < realmin) = realmin;
            cost = feval(datatype, - sum(y(:) .* log(x(:))));
        end
        
        function [cost, deltas] = applyCell(obj, x, y)
            [cost, deltas] = obj.apply(x{1}, y{1});
        end
    end
end
