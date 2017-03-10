classdef lossMSEperPixelNormalization < ILossFun
    properties
        toDeriv = true;
    end
    
    methods
        function obj = lossMSEperPixelNormalization()
        end
        
        function y = derivOut(obj)
            y = obj.toDeriv;
        end
        
        function [cost, deltas] = apply(obj, x, y)
            nPixels = size(x,1) * size(x,2);
            deltas = x - y;
            
            cost = 0.5 * sum((deltas(:)).^2);
            cost = cost/nPixels;
            deltas = deltas/nPixels;
        end
        
        function [cost, deltas] = applyCell(obj, x, y)
            [cost, deltas] = obj.apply(x{1}, y{1});
        end
    end
end
