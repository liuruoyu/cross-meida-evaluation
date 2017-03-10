function cmc = CMCcurve(res, gnd_q, gnd_db)
% param: res, each column for each query
% param: gnd_q, each column for each query
% param: gnd_db, each column for each query

num_q = size(res,2);
num_db = size(res,1);
isfind = zeros(num_q,num_db);
S = (gnd_q'*gnd_db)>0;

for i = 1:num_q
    tmp_idx = res(:,i);
    tmp_S = S(i,tmp_idx);
    f_pos = find(tmp_S,1);
    isfind(i,f_pos:end) = 1;
end

cmc = sum(isfind,1)./num_q;

end
