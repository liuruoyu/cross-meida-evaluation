function [ hcode ] = code2hash_mlbe( icode )

% Version: 0.1.0  -- 2013.12.28
% Authors: Liu Ruoyu

[k,n] = size( icode );
hcode = zeros(k*8,n);
for i = 1:n
    for j = 1:k
        hcode((j*8 - 7):(j*8),i) = bitget(icode(j,i),8:-1:1)';
    end
end


end

