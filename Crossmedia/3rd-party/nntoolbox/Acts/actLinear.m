classdef actLinear < IAct
    properties
        beta = 1;
    end
    
    methods
        function obj = actLinear()
        end
        
        function y = apply(obj, x)
            y = x;
        end
        
        function y = deriv1(obj, x, da)
            if isa(x, 'parallel.gpu.GPUArray')
                datatype = class(gather(x));
            else
                datatype = class(x);
            end
            y = da;
        end
        
        function y = deriv2(obj, x, dx, da, xprime)
            if isa(x, 'parallel.gpu.GPUArray')
                datatype = class(gather(x));
            else
                datatype = class(x);
            end
            y = zeros(size(x), datatype);
        end
    end
end

