%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Description
%  compute MAP

%Input
%  queryset     n*dim_a data matrix 
%  targetset     n*dim_b data matrix
%  test_Y       n*1 label vector

%Output
%  map   MAP score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function map = calculateMAP( queryset, targetset, test_Y )
    
    Dist = CosineDist(queryset, targetset);
    [asDist index] = sort(Dist, 2, 'ascend');
    classIndex = test_Y(index);
    ntest = size(unique(test_Y));
    AP = [];
    
    [num c] = size( queryset );
    for k = 1:num
        reClassIndex = find(classIndex(k, :) == test_Y(k));
        relength = length(reClassIndex);
        counts = [1:relength];
        AP =[AP sum(counts./reClassIndex)/relength];
    end
    
    map = mean (AP);
end

