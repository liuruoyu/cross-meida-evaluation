function D = fastEuclideanDist(A, B, batchSize)
%{
A = rand(4754,1024);
B = rand(6800,1024);

tic
D = pdist2(A,B,'euclidean');
toc

tic
DD = sqrt( bsxfun(@plus,sum(A.^2,2),sum(B.^2,2)') - 2*(A*B') );
toc
%}

if nargin < 3
    batchSize = -1;
end

if batchSize > 0
    nBatches = ceil(size(A, 1) / batchSize);
    
    D = zeros(size(A, 1), size(B, 1), class(A));
    for i = 1 : nBatches
        from = (i - 1) * batchSize + 1;
        to = min(i * batchSize, size(A, 1));
%         D(from:to, :) = fastEuclideanDist(A(from:to, :), B, -1);
        D(from:to, :) = sqrt(bsxfun(@plus, sum(A(from:to, :).^2,2), sum(B.^2,2)') - 2*(A(from:to, :)*B'));
    end
else
    D = sqrt(bsxfun(@plus, sum(A.^2,2), sum(B.^2,2)') - 2*(A*B'));
end
