function [x, t] = sliceMMData(x,idxs,ndims)

if ndims == 2
    x.M1.x1 = x.M1.x1(idxs,:);
    x.M1.x2 = x.M1.x2(idxs,:);
    x.M1.t = x.M1.t(idxs,:);
    x.M2.x1 = x.M2.x1(idxs,:);
    x.M2.x2 = x.M2.x2(idxs,:);
    x.M2.t = x.M2.t(idxs,:);
    x.CM.x1 = x.CM.x1(idxs,:);
    x.CM.x2 = x.CM.x2(idxs,:);
    x.CM.t = x.CM.t(idxs,:);
else
    x.M1.x1 = x.M1.x1(:,:,:,idxs);
    x.M1.x2 = x.M1.x2(:,:,:,idxs);
    x.M1.t = x.M1.t(:,:,:,idxs);
    x.M2.x1 = x.M2.x1(:,:,:,idxs);
    x.M2.x2 = x.M2.x2(:,:,:,idxs);
    x.M2.t = x.M2.t(:,:,:,idxs);
    x.CM.x1 = x.CM.x1(:,:,:,idxs);
    x.CM.x2 = x.CM.x2(:,:,:,idxs);
    x.CM.t = x.CM.t(:,:,:,idxs);
end
t = x.CM.t;