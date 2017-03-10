function cell2idx(data, siz, nchannels, fname)
% siz: edge, only square
% performs resizing long edge using siz

train = zeros(siz*siz*nchannels,length(data),class(data{1}));

for i=1:length(data)
    tmp = zeros(siz,siz,nchannels);
    im = imresizeLongEdge(data{i}, siz);
    tmp(1:size(im,1),1:size(im,2),:) = im;
    for c = 1:size(im,3)
        tmp(:,:,c) = tmp(:,:,c)';
    end
    train(:,i) = normMinMax(tmp(:),0,255,'all');
end

fid = fopen(fname, 'w', 'b');
fwrite(fid, 2051, 'int');
fwrite(fid, length(data), 'int');
fwrite(fid, siz, 'int');
fwrite(fid, siz, 'int');
fwrite(fid, train(:), 'uchar')
fclose(fid);
