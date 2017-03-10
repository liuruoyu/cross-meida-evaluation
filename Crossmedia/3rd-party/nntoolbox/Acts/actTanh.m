classdef actTanh < IAct
    properties
        beta = 1;
    end
    
    methods
        function obj = actTanh(beta)
            if nargin < 1
                obj.beta = 1;
            else
                obj.beta = beta;
            end
        end
        
        function y = apply(obj, x)
            y = tanh(obj.beta * x);
        end
        
        function y = deriv1(obj, x, da)
            y = da .* (obj.beta * (1 - x.^2));
        end
        
        function y = deriv2(obj, x, dx, da, xprime)
            if nargin < 5
                xprime = obj.deriv1(x, dx);
            end
            
            y = da .* (-2.0 * x .* xprime);
        end
    end
end

