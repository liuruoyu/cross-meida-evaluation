classdef lossMCCEbinaryWeighted < ILossFun
    properties
        alpha % weigth for class 1
        normPerClass
        toDeriv = false;
    end
    
    methods
        function obj = lossMCCEbinaryWeighted(alpha, normPerClass)
            obj.alpha = alpha;
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
            
            n1 = sum(cl == 1);
            n2 = sum(cl == 2);
            
            deltas = feval(datatype, x - y);
            deltas(cl==1,:) = deltas(cl==1,:) * obj.alpha;
            if obj.normPerClass
                deltas(cl==1,:) = deltas(cl==1,:) / n1;
                deltas(cl==2,:) = deltas(cl==2,:) / n2;
            end
            
            % Ensure that log(x) is computable
            x(x < realmin) = realmin;
            
            cost = feval(datatype, - sum(y .* log(x), 2));
            cost(cl==1,:) = cost(cl==1,:) * obj.alpha;
            if obj.normPerClass
                cost(cl==1,:) = cost(cl==1,:) / n1;
                cost(cl==2,:) = cost(cl==2,:) / n2;
            end
            cost = sum(cost(:));
        end
        
        function [cost, deltas] = applyCell(obj, x, y)
            [cost, deltas] = obj.apply(x{1}, y{1});
        end
    end
end
