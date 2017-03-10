function llh = evaluate_obj(SX, SY, SXY, OXY, U, V, Wx, Wy, wxy, hyper)

    phix = 1/hyper.phix;
    phiy = 1/hyper.phiy;
    phixy = 1/hyper.phixy;
    thetax = 1/hyper.thetax;
    thetay = 1/hyper.thetay;
    
    [I, J] = size(SXY);
    K = size(U,1);
    % 1. lnP(U)
    nPos = length(find(U>0));
    nNeg = size(U,1)*size(U,2) - nPos;
    llh1 = nPos*log(hyper.alphax/(hyper.alphax+hyper.betax))...
        + nNeg*log(hyper.betax/(hyper.alphax+hyper.betax));
    % 2. lnP(V)
    nPos = length(find(V>0));
    nNeg = size(V,1)*size(V,2) - nPos;
    llh2 = nPos*log(hyper.alphay/(hyper.alphay+hyper.betay))...
        + nNeg*log(hyper.betay/(hyper.alphay+hyper.betay));
    % 3. lnP(Wx)
    llh3 = (K*(K+1)/4)*log(phix/(2*pi))-0.5*phix*sum(sum(triu(Wx.^2)));
    % 4. lnP(Wy)
    llh4 = (K*(K+1)/4)*log(phiy/(2*pi))-0.5*phiy*sum(sum(triu(Wy.^2)));
    % 5. lnP(Sx)
    llh5 = (I*(I+1)/4)*log(thetax/(2*pi))...
        -0.5*thetax*sum(sum(triu((SX-U'*Wx*U).^2)));
    % 6. lnP(Sy)
    llh6 = (J*(J+1)/4)*log(thetay/(2*pi))...
        -0.5*thetay*sum(sum(triu((SY-V'*Wy*V).^2)));
    % 7. lnP(Sxy)
    llh7 = 0;
    id = find(OXY>0);
    Th = logsig(wxy*(U'*V));
    [nrows,ncols] = ind2sub([I,J],id);
    for i = 1:length(id)        
%         p = logsig(U(:,nrow)'*V(:,ncol));
        p = Th(nrows(i),ncols(i));
        if SXY(nrows(i), ncols(i))>0
            llh7 = llh7 + log(p);
        else
            llh7 = llh7 + log(1 - p);
        end
    end    
    % 8. lnP(thetax)
    llh8 = (hyper.atheta-1)*log(thetax)-hyper.btheta*thetax;
    
    % 9. lnP(thetay)
    llh9 = (hyper.ctheta-1)*log(thetay)-hyper.dtheta*thetay;
    
    % 10. lnP(phix)
    llh10 = (hyper.aphi-1)*log(phix)-hyper.bphi*phix;
    
    % 11. lnP(phiy)
    llh11 = (hyper.cphi-1)*log(phiy)-hyper.dphi*phiy;
    
    % 12. lnP(wxy)
    llh12 = 0.5*log(2*phixy/pi)-0.5*phixy*wxy*wxy;
    
    % 13. lnP(phixy)
    llh13 = (hyper.ephi-1)*log(phixy)-hyper.fphi*phixy;
    
%     [llh1 llh2 llh3 llh4 llh5 llh6 llh7 llh8 llh9 llh10 llh11]
    
    llh = llh1+llh2+llh3+llh4+llh5+llh6+llh7+llh8+llh9+llh10+llh11+llh12+llh13;
end