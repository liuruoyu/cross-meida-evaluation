classdef lossCE < ILossFun
    properties
        toDeriv = false;
    end
    
    methods
        function obj = lossCE()
        end
        
        function y = derivOut(obj)
            y = obj.toDeriv;
        end
        
        function [cost, deltas] = apply(obj, x, y)
            datatype = class(x);
            x = double(x);
            y = double(y);

            deltas = feval(datatype, x - y);

            cost = feval(datatype, - sum(sum(y(:).*log(x(:)) + (1 - y(:)).*log(1 - x(:)))));
        end
        
        function [cost, deltas] = applyCell(obj, x, y)
            [cost, deltas] = obj.apply(x{1}, y{1});
        end
    end
end
