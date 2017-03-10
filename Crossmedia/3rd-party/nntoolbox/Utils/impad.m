function y = impad(im,siz,val,centre)
%IMPAD pad the image with val values if the dimensions are smaller than the
%one given in siz.
%center is a boolean value which indicates whether to center or not the
%resulting image

if nargin < 3
    val = 0;
end

if nargin < 4
    centre = true;
end

[rows, cols, channels,samples] = size(im);
outrows = rows;
outcols = cols;
rowstart = 0;
colstart = 0;

if rows < siz(1)
    outrows = siz(1);
end

if cols < siz(2)
    outcols = siz(2);
end

if centre
    rowstart = round(floor(outrows - rows)/2);
    colstart = round(floor(outcols - cols)/2);
end

y = zeros(outrows,outcols,channels,samples,class(im)) + val;
y(rowstart+1:rowstart+rows,colstart+1:colstart+cols,:,:) = im;

