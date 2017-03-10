function [map, aps, res] = mAPEvaluate(q, db, gnd_q, gnd_db, R, distmed)

map = 0;
[qn,~] = size(q);
[dbn,~] = size(db);
[qidx,~] = knnsearch( db, q, 'K', dbn, 'Distance', distmed );

aps = zeros(1,qn); % 1*N
res = qidx'; % K*N

for i = 1:qn
    catq = gnd_q(:,i);
    idx = qidx(i,1:R);
    catdb = gnd_db(:,idx);
    ridx = (catq'*catdb)>0;
    ranks = 1:R;
    ranks = ranks(ridx);
    gIdxCatnum = length(ranks);
    ap = score_ap_from_ranks1( ranks, gIdxCatnum );
    aps(i) = ap;
    map = map + ap;   
end

map = map / qn;

end

