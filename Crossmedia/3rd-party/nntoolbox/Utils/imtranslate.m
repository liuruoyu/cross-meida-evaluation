function im = imtranslate(i,perc)
%perc: 0..1
if nargin < 2
    perc = .2
end

siz = size(i);
% sizout = floor(siz * .79);
sizout = floor(siz * (1.01 - perc));

% o_x = randi(floor(siz(1) * .1),1,1);
% o_y = randi(floor(siz(2) * .1),1,1);
o_x = randi(floor(siz(1) * perc/2),1,1);
o_y = randi(floor(siz(2) * perc/2),1,1);

if ndims(i) == 3
    im = i(o_x:o_x + sizout(1), o_y:o_y + sizout(2), :);
else
    im = i(o_x:o_x + sizout(1), o_y:o_y + sizout(2));
end
