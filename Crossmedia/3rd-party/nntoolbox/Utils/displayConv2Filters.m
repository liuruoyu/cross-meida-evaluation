function imout = displayConv2Filters(w,fighandle,siz,localNormalise,bgval)
%DISPLAYCONV2FILTERS takes a 4D tensor and displays the filters. If the
%third dimension is 3 then it plots them in colours.
%
%   fighandle: figure handle where to plot, if not needed use []
%   siz: shape of the tiled image (ntiles_rows,ntiles_cols). if not given
%       or [] it is automatically computed using sqrt(nfilters)
%   localNormalise: whether to normalise each filter onto 0..1
%   bgval: background value

%   This file is part of netlabExtensions.
%   July 2011
%	Jonathan Masci <jonathan@idsia.ch>


if nargin < 2 || isempty(fighandle)
    fighandle = figure();
end

if nargin < 4
    localNormalise = true;
end

if nargin < 5
    bgval = 0;
end

if ndims(w) == 3
    w = reshape(w,size(w,1),size(w,2),1,size(w,3));
end

H = size(w,1);
W = size(w,2);
nchannels = size(w,3);
nfilters = size(w,4);

if nargin < 3 || isempty(siz)
    cols = round(sqrt(nfilters));
    rows = ceil(nfilters / cols);
else
    cols = siz(2);
    rows = siz(1);
end
image = zeros(rows * (H+1) + 1, cols * (W+1) + 1, nchannels) + bgval;

for i=1:nfilters
    r = floor((i-1) / cols);
    c = mod(i-1, cols);
    
    if localNormalise
        image((r*(H+1)+1)+1:((r+1)*(H+1)),(c*(W+1)+1)+1:((c+1)*(W+1)),:) = normMinMax(w(:,:,:,i),0,1);
    else
        image((r*(H+1)+1)+1:((r+1)*(H+1)),(c*(W+1)+1)+1:((c+1)*(W+1)),:) = w(:,:,:,i);
    end
end

figure(fighandle);
if localNormalise
    imshow(image)
else
    mn=-1.5;
    mx=+1.5;
    image = (image - mn) / (mx - mn);
    imshow(image)
%     imshow(image,[])
end

drawnow

if nargout == 1
    imout = image;
end