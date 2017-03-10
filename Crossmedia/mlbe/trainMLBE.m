%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Distributed under GNU General Public License (see license.txt for details).
%
% Copyright (c) 2012 Linus ZHEN Yi
% All Rights Reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function for training MLBE codes
% Input:
%   Sx      : similarity matrix of X modality
%   Sy      : similarity matrix of Y modality
%   Sxy     : similarity matrix of cross-modality similarity
%   O       : indicator matrix
%   hyper   : hyper parameters
%   clength : code length
%   U       : initialization of hash code of X modality
%   V       : initialization of hash code of Y modality
% output:
%   U       : hash codes of X modality
%   V       : hash codes of Y modality
%   Wx      : weight matrix of X modality
%   Wy      : weight matrix of Y modality
%   wxy     : weight of cross-modality term
%   hyper   : hyper parameters
%   loglikelihood: objective value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U, V, Wx, Wy, wxy, hyper, loglikelihood] = trainMLBE(Sx, Sy, Sxy, O, hyper, clength, U, V)

    show_tips = 0;

    [N,M] = size(Sxy);
    if nargin < 7
        [U, V] = initializeUV(N, M, clength);        
    end    
    Wx = initializeW(clength);
    Wy = initializeW(clength);
    wxy = 0.1;

    loglikelihood = evaluate_obj(Sx, Sy, Sxy, O, U, V, Wx, Wy, wxy, hyper);
    fprintf('Initial total cost: %.4f\n', loglikelihood);
    do_updateWx = 1; do_updatephix = 1; do_updateU = 1; do_updatethetax = 1;
    do_updateWy = 1; do_updatephiy = 1; do_updateV = 1; do_updatethetay = 1;
    do_updatewxy = 1; do_updatephixy = 1;
    covergecond = do_updateWx+do_updateU+do_updateWy+do_updateV+do_updatephix...
        +do_updatephiy+do_updatethetax+do_updatethetay+do_updatewxy+do_updatephixy;
    converge = 0;
%     isconverge = 0;
%     loglikelihood = -inf;
    maxIter = 30; % the maximum number of iterations
    for i = 1:maxIter
        fprintf('iteration: %d\tobj: %f\n', i-1, loglikelihood);

        % for Wx
        if do_updateWx
            ihyper.theta = hyper.thetax;
            ihyper.phi = hyper.phix;
            Wx2 = updatew2(U, Sx, ihyper);
            loglikelihood2 = evaluate_obj(Sx, Sy, Sxy, O, U, V, Wx2, Wy, wxy, hyper);
            if loglikelihood2 - loglikelihood > 1e-2 
                s = 'good'; converge = 0; loglikelihood = loglikelihood2;
                Wx = Wx2;
            elseif loglikelihood2-loglikelihood>=0
                s = 'neutral'; Wx = Wx2; converge = converge + 1; 
                loglikelihood = loglikelihood2;
                if converge == covergecond,disp('Converged!');break;end
            else
                s = 'bad'; 
                converge = converge + 1; 
                if converge == covergecond,disp('Converged!');break;end
            end
            clear Wx2;
            if show_tips, fprintf('total cost: %.4f; %s\n', loglikelihood,s );end
        end

        % for phix
        if do_updatephix
            iphi = hyper.phix;
            hyper.phix = 4*(hyper.bphi+0.5*sum(sum(triu(Wx.^2))))/(clength*(clength+1)+4*(hyper.aphi-1));
            loglikelihood2 = evaluate_obj(Sx, Sy, Sxy, O, U, V, Wx, Wy, wxy, hyper);
            if loglikelihood2 - loglikelihood > 1e-2 
                s = 'good'; converge = 0; loglikelihood = loglikelihood2;
            elseif loglikelihood2 - loglikelihood >= 0
                s = 'neutral'; converge = converge + 1; 
                loglikelihood = loglikelihood2;
                if converge == covergecond,disp('Converged!');break;end
            else
                s = 'bad'; hyper.phix = iphi;
                converge = converge + 1; 
                if converge == covergecond,disp('Converged!');break;end
            end
            if show_tips, fprintf('total cost: %.4f; %s\n', loglikelihood,s );end
        end
        
        % for U
        if do_updateU
            ihyper.alpha = hyper.alphax; 
            ihyper.beta = hyper.betax; 
            ihyper.theta = hyper.thetax;
            U2 = updateu(U, V, Wx, wxy, Sx, Sxy, O, ihyper);
            loglikelihood2 = evaluate_obj(Sx, Sy, Sxy, O, U2, V, Wx, Wy, wxy, hyper);
            if loglikelihood2 - loglikelihood > 1e-2 
                s = 'good'; converge = 0; loglikelihood = loglikelihood2;
                U = U2;
            elseif loglikelihood2 - loglikelihood >= 0
                s = 'neutral'; converge = converge + 1; 
                loglikelihood = loglikelihood2;
                U = U2;
                if converge == covergecond,disp('Converged!');break;end
            else
                s = 'bad'; 
                converge = converge + 1; 
                if converge == covergecond,disp('Converged!');break;end
            end
            clear U2;
            if show_tips, fprintf('total cost: %.4f; %s U positive: %.4f\n', loglikelihood, s, length(find(U>0))/(size(U,1)*size(U,2)));end
        end

        % for thetax
        if do_updatethetax
            itheta = hyper.thetax;
            hyper.thetax = 4*(hyper.btheta+0.5*sum(sum(triu((Sx-U'*Wx*U).^2))))/(size(Sx,2)*(size(Sx,2)+1)+4*(hyper.atheta-1));
            loglikelihood2 = evaluate_obj(Sx, Sy, Sxy, O, U, V, Wx, Wy, wxy, hyper);
            if loglikelihood2 - loglikelihood > 1e-2 
                s = 'good'; converge = 0; loglikelihood = loglikelihood2;
            elseif loglikelihood2 - loglikelihood >= 0
                s = 'neutral'; converge = converge + 1; 
                loglikelihood = loglikelihood2;
                if converge == covergecond,disp('Converged!');break;end
            else
                s = 'bad'; hyper.thetax = itheta;
                converge = converge + 1; 
                if converge == covergecond,disp('Converged!');break;end
            end
            if show_tips, fprintf('total cost: %.4f; %s\n', loglikelihood, s);end
        end
        
        % for Wy
        if do_updateWy
            ihyper.theta = hyper.thetay;
            ihyper.phi = hyper.phiy;
            Wy2 = updatew2(V, Sy, ihyper);
            loglikelihood2 = evaluate_obj(Sx, Sy, Sxy, O, U, V, Wx, Wy2, wxy, hyper);
            if loglikelihood2 - loglikelihood > 1e-2 
                s = 'good'; converge = 0; loglikelihood = loglikelihood2;
                Wy = Wy2;
            elseif loglikelihood2-loglikelihood>=0
                s = 'neutral'; Wy = Wy2; converge = converge + 1; 
                loglikelihood = loglikelihood2;
                if converge == covergecond,disp('Converged!');break;end
            else
                s = 'bad'; 
                converge = converge + 1; 
                if converge == covergecond,disp('Converged!');break;end
            end
            clear Wy2;
            if show_tips, fprintf('total cost: %.4f; %s\n', loglikelihood,s );end
        end

        % for phiy
        if do_updatephiy
            iphi = hyper.phiy;
            hyper.phiy = 4*(hyper.dphi+0.5*sum(sum(triu(Wy.^2))))/(clength*(clength+1)+4*(hyper.cphi-1));
            loglikelihood2 = evaluate_obj(Sx, Sy, Sxy, O, U, V, Wx, Wy, wxy, hyper);
            if loglikelihood2 - loglikelihood > 1e-2 
                s = 'good'; converge = 0; loglikelihood = loglikelihood2;
            elseif loglikelihood2 - loglikelihood >= 0
                s = 'neutral'; converge = converge + 1; 
                loglikelihood = loglikelihood2;
                if converge == covergecond,disp('Converged!');break;end
            else
                s = 'bad'; hyper.phiy = iphi;
                converge = converge + 1; 
                if converge == covergecond,disp('Converged!');break;end
            end
            if show_tips, fprintf('total cost: %.4f; %s\n', loglikelihood,s );end
        end
        
        % for V
        if do_updateV
            ihyper.alpha = hyper.alphay; 
            ihyper.beta = hyper.betay; 
            ihyper.theta = hyper.thetay;
            V2 = updateu(V, U, Wy, wxy, Sy, Sxy',O', ihyper); %initializeU(M,clength); %
            loglikelihood2 = evaluate_obj(Sx, Sy, Sxy, O, U, V2, Wx, Wy, wxy, hyper);
            if loglikelihood2 - loglikelihood > 1e-2 
                s = 'good'; converge = 0; loglikelihood = loglikelihood2;
                V = V2;
            elseif loglikelihood2 - loglikelihood >= 0
                s = 'neutral'; converge = converge + 1; 
                loglikelihood = loglikelihood2;
                V = V2;
                if converge == covergecond,disp('Converged!');break;end
            else
                s = 'bad'; 
                converge = converge + 1; 
                if converge == covergecond,disp('Converged!');break;end
            end
            clear V2;
            if show_tips, fprintf('total cost: %.4f; %s V positive: %.4f\n', loglikelihood, s,length(find(V>0))/(size(V,1)*size(V,2)));end
        end

        % for thetay
        if do_updatethetay
            itheta = hyper.thetay;
            hyper.thetay = 4*(hyper.dtheta+0.5*sum(sum(triu((Sy-V'*Wy*V).^2))))/(size(Sy,2)*(size(Sy,2)+1)+4*(hyper.ctheta-1));
            loglikelihood2 = evaluate_obj(Sx, Sy, Sxy, O, U, V, Wx, Wy, wxy, hyper);
            if loglikelihood2 - loglikelihood > 1e-2 
                s = 'good'; converge = 0; loglikelihood = loglikelihood2;
            elseif loglikelihood2 - loglikelihood >= 0
                s = 'neutral'; loglikelihood = loglikelihood2;
                converge = converge + 1; 
                if converge == covergecond,disp('Converged!');break;end
            else
                s = 'bad'; hyper.thetay = itheta;
                converge = converge + 1; 
                if converge == covergecond,disp('Converged!');break;end
            end
            if show_tips, fprintf('total cost: %.4f; %s\n', loglikelihood,s );end
        end
        
        % for wxy
        if do_updatewxy
            iparams.phi = 1/hyper.phixy;
            wxy2 = updatew_scalar_bern(wxy, U, V, Sxy, O, iparams);
            loglikelihood2 = evaluate_obj(Sx, Sy, Sxy, O, U, V, Wx, Wy, wxy2, hyper);
            if loglikelihood2 - loglikelihood > 1e-2 
                s = 'good'; converge = 0; loglikelihood = loglikelihood2;
                wxy = wxy2;
            elseif loglikelihood2-loglikelihood>=0
                s = 'neutral'; wxy = wxy2; converge = converge + 1; 
                loglikelihood = loglikelihood2;
                if converge == covergecond,disp('Converged!');break;end
            else
                s = 'bad'; 
                converge = converge + 1; 
                if converge == covergecond,disp('Converged!');break;end
            end
            clear wxy2;
            if show_tips, fprintf('total cost: %.4f; %s\n', loglikelihood,s );end
        end
        
        % for phixy
        if do_updatephixy
            iphi = hyper.phixy;
            hyper.phixy = (2*hyper.fphi + wxy.^2)/(2*hyper.ephi - 1);
            loglikelihood2 = evaluate_obj(Sx, Sy, Sxy, O, U, V, Wx, Wy, wxy, hyper);
            if loglikelihood2 - loglikelihood > 1e-2 
                s = 'good'; converge = 0; loglikelihood = loglikelihood2;
            elseif loglikelihood2 - loglikelihood >= 0
                s = 'neutral'; converge = converge + 1; 
                loglikelihood = loglikelihood2;
                if converge == covergecond,disp('Converged!');break;end
            else
                s = 'bad'; hyper.phixy = iphi;
                converge = converge + 1; 
                if converge == covergecond,disp('Converged!');break;end
            end
            if show_tips, fprintf('total cost: %.4f; %s\n', loglikelihood,s );end
        end
    end
end



function [U, V] = initializeUV(N,M,K)
    U = randsrc(K,N);
    V = randsrc(K,M);
end














