classdef lossMSE < ILossFun
    properties
        toDeriv = true;
    end
    
    methods
        function obj = lossMSE()
        end
        
        function y = derivOut(obj)
            y = obj.toDeriv;
        end
        
        function [cost, deltas] = apply(obj, x, y)
            deltas = x - y;
            cost = 0.5 * sum((deltas(:)).^2);
        end
        
        function [cost, deltas] = applyCell(obj, x, y)
            [cost, deltas] = obj.apply(x{1}, y{1});
        end
    end
end
