function [U] = updateu(U, V, Wx, wxy, Sx, Sxy, O, hyper)
% this function can be used for updating U or V
% U: K x N, code matrix for X, each column is a point
% V: K x M, code matrix for Y, each column is a point
% W: K x K, weight matrix
% Sx:N x N, pairwise similarity matrix
% Sxy:  N x M, pairwise relation matrix
% O:  N x M, pairwise observation matrix
[K, N] = size(U); M = size(V,2);

% to speed up, we just choose several random points to update
% randids1 = randperm(N);
npoints = floor(N);
% flipU = zeros(size(U));
for id = 1:npoints
%     tic
    i = id;%randids1(id);
%     randids2 = randperm(K);
    nbits = floor(K);%
    for ik = 1:nbits
        ui = U(:,i);
        k = ik; %randids2(ik);
        ui1 = ui;ui1(k) = 1; ui2 = ui; ui2(k) =-1;
        ud = zeros(size(ui)); ud(k)=2;
        
        D1 = ud'*Wx*U;
        D2 = Sx(i,:).*D1;
        lp = -2*(sum(D2) - D2(i));
        
%         Wxa = Wx*U;
%         Wxi = Wxa'*(ui1*ud'+ud*ui2')*Wxa;
%         lp = lp + trace(Wxi)- Wxi(i,i);
        psW = Wx*(ui1*ud'+ud*ui2')*Wx;
%         tic
        for j=1:N
            if j ~= i
                uj = U(:,j);
                lp = lp + uj'*psW*uj;
            end
        end
        
        lp = lp + ((Sx(i,i)-ui1'*Wx*ui1)^2-(Sx(i,i)-ui2'*Wx*ui2)^2);
%         toc
%         tic 
        Th1 = logsig(wxy*ui1'*V); Th2 = logsig(wxy*ui2'*V); 
        Th3 = Sxy(i,:).*log(Th1./Th2);
        Th1 = 1 - Th1; Th2 = 1 - Th2; 
        Th4 = (1-Sxy(i,:)).*log(Th1./Th2);
        lb = O(i,:)*(Th3 + Th4)';
%         toc
        
        cost = log(hyper.alpha/hyper.beta) - lp/(2*hyper.theta) + lb;
%         oldUki = U(k,i);
        if cost>0
            U(k,i) = 1;
%             if oldUki == -1, disp('flopped');end
        else
            U(k,i) = -1;
%             if oldUki == 1, disp('flopped');end
        end
    end
%     toc
end
% U = U.*flipU;
end