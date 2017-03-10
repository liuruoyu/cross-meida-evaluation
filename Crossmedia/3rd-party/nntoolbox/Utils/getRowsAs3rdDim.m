function out = getRowsAs3rdDim(x, imsize, mergePatterns)
%GETROWSAS3RDDIM returns the 4D tensor where each line in X is rearranged
%as 3rd dimension.
%It is the inverse mapping for get3rdDimAsRows
%
%   imsize: (rows, cols)

%   This file is part of netlabExtensions.
%   August 2011
%	Jonathan Masci <jonathan@idsia.ch>

if nargin < 3
    mergePatterns = 1;
end

onGpu = false;
if isa(x, 'parallel.gpu.GPUArray')
    onGpu = true;
    x = gather(x);
end 

if mergePatterns
    nsamples = size(x,1) / prod(imsize);
    out = permute(reshape(x',size(x,2),imsize(1),imsize(2),nsamples),[2 3 1 4]);
else
    nsamples = size(x,1);
    out = permute(reshape(x,nsamples,imsize(1),imsize(2),size(x,3)),[2 3 4 1]);
end

if onGpu
    out = gpuArray(out);
end