classdef lossHingeMultiClass < ILossFun
    properties
        toDeriv = true
        M
    end
    
    methods
        function obj = lossHingeMultiClass(M)
            obj.M = M;
        end
        
        function y = derivOut(obj)
            y = obj.toDeriv;
        end
        
        function [cost, deltas] = apply(obj, x, y)
            nSamples = size(x,1);
            
            margin = max(0, obj.M - y .* x);
            
            deltas = -2/nSamples * margin .* y;
            
            cost = sum(margin(:).^2) / nSamples;
        end
        
        function [cost, deltas] = applyCell(obj, x, y)
            [cost, deltas] = obj.apply(x{1}, y{1});
        end
    end
end
