function [Xtrainidx, Xtestidx, Xrefidx, Ytrainidx, Ytestidx, Yrefidx]...
    = generateRDIDX(Ntrain,Ntest,Nref,Nrepeat)

% % for ucsd data
% Ntrain = 2173;
% Ntest = 693;
% Nref = 200;
% Nrepeat = 10;

Xtrainidx = zeros(Ntrain-Nref,Nrepeat);% each column is a repeat
Ytrainidx = zeros(Ntrain-Nref,Nrepeat);
Xrefidx = zeros(Nref,Nrepeat);
Yrefidx = zeros(Nref,Nrepeat);
Xtestidx = [Ntrain+1:(Ntrain+Ntest)]';
Ytestidx = [Ntrain+1:(Ntrain+Ntest)]';
for i = 1:10
    rp = randperm(Ntrain);
    Xrefidx(:,i) = rp(1:Nref);
    Xtrainidx(:,i) = setdiff([1:Ntrain]',Xrefidx(:,i));
%     rp = randperm(Ntrain);
    Yrefidx(:,i) = rp(1:Nref);
    Ytrainidx(:,i) = setdiff([1:Ntrain]',Yrefidx(:,i));
end

save(sprintf('../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_RDIDXMatTOY%d',Nref),...
    'Xtrainidx','Xtestidx','Xrefidx','Ytrainidx','Ytestidx','Yrefidx');


