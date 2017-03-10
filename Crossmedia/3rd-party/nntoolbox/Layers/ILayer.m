classdef ILayer < handle
    methods (Abstract)
        y = fwd(obj, x)
        y = bkp(obj, x)
        g = grad(obj, x, y)
        w = getParams(obj)
        setParams(obj, w)
        n = nParams(obj);
        
        y = fwdCell(obj, x)
        y = bkpCell(obj, x)
        g = gradCell(obj, x, y)
        w = getParamsCell(obj)
        setParamsCell(obj, w)
        
        n = getNOut(obj);
        
        o = clone(obj);
    end
end
