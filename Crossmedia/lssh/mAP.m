function map = mAP(sim_x,L_tr,L_te,mark)
%sim_x(i,j) denote the sim bewteen query j and database i 
%
% Reference:
% Jile Zhou, GG Ding, Yuchen Guo
% "Latent Semantic Sparse Hashing for Cross-modal Similarity Search"
% ACM SIGIR 2014
% (Manuscript)
%
% Version1.0 -- Nov/2013
% Written by Jile Zhou (zhoujile539@gmail.com)
%

tn = size(sim_x,2);
APx = zeros(tn,1);
R = 100;
%L_tr = [L_tr;L_tr];
for i = 1 : tn
    Px = zeros(R,1);
    deltax = zeros(R,1);
    label = L_te(i);
    if mark == 0
        [~,inxx] = sort(sim_x(:,i),'descend');
    elseif mark == 1
        [~,inxx] = sort(sim_x(:,i));
    end
    Lx = length([L_tr(inxx(1:R)) == label]);
    for r = 1 : R
        Lrx = length([L_tr(inxx(1:r)) == label]);
        if label == L_tr(inxx(r))
            deltax(r) = 1;
        end
        Px(r) = Lrx/r;
    end
    if Lx ~=0
        APx(i) = sum(Px.*deltax)/Lx;
    end
end
map = mean(APx);

