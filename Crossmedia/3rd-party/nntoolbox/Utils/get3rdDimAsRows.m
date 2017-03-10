function out = get3rdDimAsRows(x, mergePatterns)
%GET3RDDIMASROWS returns the 3rd dimension of a 4d tensor shaped as
% (rows cols nchannels ndata)
%and returns the column of channels as rows in a 3d tensor shaped as
%   (ndata rows*cols nchannels) if mergePatterns is 0
%   (ndata*rows*cols nchannels) if mergePatterns is 1
%
% For a given position returns the k values in the maps of the various
% channels.

%   This file is part of netlabExtensions.
%   July 2011
%	Jonathan Masci <jonathan@idsia.ch>

if nargin < 2
    mergePatterns = 1;
end

onGpu = false;
if isa(x, 'parallel.gpu.GPUArray')
    x = gather(x);
    onGpu = true;
end

if mergePatterns
    out = reshape(permute(x,[3 1 2 4]),size(x,3),size(x,1)*size(x,2)*size(x,4))';
else
    out = reshape(permute(x,[4 1 2 3]),size(x,4),size(x,1)*size(x,2),size(x,3));
end

if onGpu
    out = gpuArray(out);
end