function [mapv] = meanap(retrieved_gnd)

ps = zeros(length(retrieved_gnd),1);
changeidx = find(retrieved_gnd>0);
total_good = length(changeidx);
if total_good >0
    for r = 1:length(changeidx)
        ps(changeidx(r)) = sum(retrieved_gnd(1:changeidx(r)))/changeidx(r);
    end
    mapv = sum(ps)/total_good;
else
    mapv = 0;
end