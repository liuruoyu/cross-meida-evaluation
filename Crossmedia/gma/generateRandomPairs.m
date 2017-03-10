function C = generateRandomPairs(labelCell,numPair)

nV = length(labelCell);
C = zeros(numPair,nV);
labLength = zeros(nV,1);

for i = 1:nV
    labLength(i) = length(labelCell{i,1});
end

pair = 0;
while pair <=numPair
    r = rand(nV,1);
    ind = ceil(r.*labLength);
    tmp = labelCell{1,1}(ind(1));
    flagBreak = 0;
    for i = 2:nV
        if ~(tmp == labelCell{i,1}(ind(i)))
            flagBreak = 1;
            break;
        end
    end
    if ~flagBreak
        pair = pair + 1;
        C(pair,:) = ind;
    end
end

C = unique(C,'rows');