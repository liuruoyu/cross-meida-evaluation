classdef ILossFun < handle
    methods (Abstract)
        y = derivOut(obj)
        [cost, deltas] = apply(obj, x, y)
        [cost, deltas] = applyCell(obj, x, y)
    end
end
