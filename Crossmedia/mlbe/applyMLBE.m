%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Distributed under GNU General Public License (see license.txt for details).
%
% Copyright (c) 2012 Linus ZHEN Yi
% All Rights Reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get hash code for out-of-sample points
% Input:
%   flag    : modality indicator
%   Sx      : similarity matrix of current modality
%   Sxy     : similarity matrix of cross-modality similarity
%   O       : indicator matrix
%   params  : parameters
%   hyper   : hyper parameters
%   clength : code length
%   U       : initialization of hash code
% output:
%   Bx      : the compact code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Bx] = applyMLBE(flag, Sx, Sxy, O, params, hyper, clength, U)

    if isempty(Sx) || isempty(Sxy)
        Bx = [];
    else
        [N,~] = size(Sx);
        if nargin < 8, U = initializeU(N,clength); end
        UR = params.UR; VR = params.VR; Wx = params.Wx; Wxy = params.wxy;
        for np = 1:N
            if rem(np,10000) == 0
                fprintf('points processed: %d', np);
            end
            u = U(:,np);
            for iter = 1:1% 1 iteration is enough, since it is nearly closed-form
                if strcmp(flag, 'x')
                    ihyper.alpha = hyper.alphax; 
                    ihyper.beta = hyper.betax; 
                    ihyper.theta = hyper.thetax;
                elseif strcmp(flag,'y') 
                    ihyper.alpha = hyper.alphay; 
                    ihyper.beta = hyper.betay; 
                    ihyper.theta = hyper.thetay;
                end
                U(:,np) = updateu2(u, UR, VR, Wx, Wxy, Sx(np,:), Sxy(np,:), O(np,:), ihyper);
            end
        end
        fprintf('All points processed!');
        Bx = compactbit(U>0);% each point is a column
    end
    
end

function [U] = updateu2(U, UR, VR, Wx, Wxy,  Sx, Sxy, O, hyper)
    % this function can be used for updating U or V, only one vector
    % U: K x 1, code matrix for X, each column is a point
    % VR: K x LM, code matrix for Y, each column is a point
    % Wx: K x K, weight matrix
    % Sx: 1 x LM, pairwise similarity vector
    % Sxy:  1 x LM, pairwise relation vector
    % O:  1 x LM, pairwise observation matrix
    [K, N] = size(U); R = size(UR,2); %M = size(VR,2);

    if N>1, disp('wrong');end
    npoints = floor(N);
    WxUR = Wx*UR;
    for id = 1:npoints
        i = id;%randids1(id);
        nbits = floor(K);%
        for ik = 1:nbits
            ui = U(:,i);
            k = ik; %randids2(ik);
            ui1 = ui; ui1(k) = 1; ui2 = ui;ui2(k) = -1;
            ud = zeros(size(ui)); ud(k)=2;

            D1 = ud'*WxUR;
            D2 = Sx.*D1;
            lp = -2*sum(D2);

            psW = Wx*(ui1*ud'+ud*ui2')*Wx;
            for j=1:R
                uj = UR(:,j);
                lp = lp + uj'*psW*uj;
            end

            Th1 = logsig(Wxy*ui1'*VR); 
            Th2 = logsig(Wxy*ui2'*VR); 
            Th3 = Sxy.*log(Th1./Th2);
            Th1 = 1 - Th1; Th2 = 1 - Th2; 
            Th4 = (1-Sxy).*log(Th1./Th2);
            lb = O*(Th3 + Th4)';

            cost = log(hyper.alpha/hyper.beta) - lp/(2*hyper.theta) + lb;
            if cost>0
                U(k,i) = 1;
            else
                U(k,i) = -1;
            end
        end
    end
end


function [U] = initializeU(N,K)
    U = randsrc(K,N);
end
