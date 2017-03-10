function [x, t] = sliceSiamData(x,idxs,ndims,datatype)

if ndims == 2
    x.x1 = x.x1(idxs,:);
    x.x2 = x.x2(idxs,:);
else
    x.x1 = x.x1(:,:,:,idxs);
    x.x2 = x.x2(:,:,:,idxs);    
end

x.x1 = feval(datatype, x.x1);
x.x2 = feval(datatype, x.x2);
x.t = feval(datatype, x.t(idxs,:));
t = x.t;