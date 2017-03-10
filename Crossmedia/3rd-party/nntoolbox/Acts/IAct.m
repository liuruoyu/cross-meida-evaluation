classdef IAct < handle
    methods (Abstract)
        y = apply(x)
        y = deriv1(x, da)
        y = deriv2(x, dx, da, xprime)
    end
end
