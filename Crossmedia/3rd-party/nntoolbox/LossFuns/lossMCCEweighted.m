classdef lossMCCEweighted < ILossFun
    properties
        alphas % weigths for classes
        normPerClass
        toDeriv = false;
    end
    
    methods
        function obj = lossMCCEweighted(alphas, normPerClass)
            obj.alphas = alphas;
            obj.normPerClass = normPerClass;
        end
        
        function y = derivOut(obj)
            y = obj.toDeriv;
        end
        
        function [cost, deltas] = apply(obj, x, y)
            datatype = class(x);
            x = double(x);
            y = double(y);
            
            [~, cl] = max(y,[],2);
            
            % Make a big alphas vector
            ealpha = zeros(size(y, 1), 1);
            
            % Get #samples per class and set alphas
            n = zeros(length(obj.alphas), 1);
            for i = 1 : length(obj.alphas)
                n(i) = sum(cl == i);
                ealpha(cl == i) = obj.alphas(i);
            end
            
            deltas = feval(datatype, x - y);
            deltas = bsxfun(@times, deltas, ealpha);
            if obj.normPerClass
                for i = 1 : length(obj.alphas)
                    deltas(cl==i,:) = deltas(cl==i,:) / n(i);
                end
            end
            
            % Ensure that log(x) is computable
            x(x < realmin) = realmin;
            
            cost = feval(datatype, - sum(y .* log(x), 2));
            cost = bsxfun(@times, cost, ealpha);
            if obj.normPerClass
                for i = 1 : length(obj.alphas)
                    cost(cl==i,:) = cost(cl==i,:) / n(i);
                end
            end
            cost = sum(cost(:));
        end
        
        function [cost, deltas] = applyCell(obj, x, y)
            [cost, deltas] = obj.apply(x{1}, y{1});
        end
    end
end
