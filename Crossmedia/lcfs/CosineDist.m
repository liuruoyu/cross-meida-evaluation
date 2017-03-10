%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Description
%  compute cosine distance

%Input
%  fea_a     n*dim_a data matrix 
%  fea_b     n*dim_b data matrix

%Output
%  D         1 - CosineSimlarity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  D = CosineDist( fea_a,fea_b )

    temp_a = fea_a;
    temp_b = fea_b;
    
    temp_a = temp_a.*temp_a;
    temp_b = temp_b.*temp_b;
    
    tempaa = sqrt(sum(temp_a, 2));
    tempbb = sqrt(sum(temp_b, 2));
    
    tempab = tempaa*tempbb';
    
    D = 1 - (fea_a * fea_b')./max(tempab, 1e-6);
end

