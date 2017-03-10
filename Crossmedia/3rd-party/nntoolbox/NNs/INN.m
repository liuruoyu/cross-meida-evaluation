classdef INN < handle
    methods (Abstract)
        y = fwd(obj, x)
        [cost, g] = grad(obj, x, y)
        w = getParams(obj)
        setParams(obj, w)
        n = nParams(obj)
        g = computeGradOnly(obj)
        g = computeGradOnlyCell(obj)
        o = clone(obj)
    end
end
