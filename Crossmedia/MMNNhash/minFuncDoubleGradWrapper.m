function [cost,grad] = minFuncDoubleGradWrapper(gfun)

[cost, grad] = gfun();
cost = double(cost);
grad = double(grad);