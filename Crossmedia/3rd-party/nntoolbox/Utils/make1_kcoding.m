function targets=make1_kcoding(class, n_out)
%function targets=make1_kcoding(class)
%
% takes the a vector of classes and makes the matrix
% with 1 for the corresponding class

cc = unique(class);
if isempty(n_out)
    targets = zeros(length(class),length(unique(cc)));
else
    targets = zeros(length(class),n_out);
end

for i=1:length(class)
    targets(i,cc == class(i)) = 1;
end