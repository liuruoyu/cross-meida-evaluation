function llh = evaluate_likelihood(SX, SY, SXY, OX, OY, O, U, V, Wx, Wy, hyper)
    [N, M] = size(SXY);
    % 1. P(U)
    nPos = length(find(U>0));
    nNeg = size(U,1)*size(U,2) - nPos;
    llh1 = nPos*(hyper.alphax/(hyper.alphax+hyper.betax)) + nNeg*(hyper.betax/(hyper.alphax+hyper.betax));
    % 2. P(V)
    nPos = length(find(V>0));
    nNeg = size(V,1)*size(V,2) - nPos;
    llh2 = nPos*(hyper.alphay/(hyper.alphay+hyper.betay)) + nNeg*(hyper.betay/(hyper.alphay+hyper.betay));
    % 3. P(Wx)
%     llh3 = -sum(sum(tril(Wx.^2,-1)))/(2*hyper.phix);
    llh3 = -sum(sum(triu(Wx.^2,1)))/(2*hyper.phix);
    % 4. P(Wy)
%     llh4 = -sum(sum(tril(Wy.^2,-1)))/(2*hyper.phiy);
    llh4 = -sum(sum(triu(Wy.^2,1)))/(2*hyper.phiy);
    % 5. P(Sx)
%     llh5 = 0;
%     tic
    SXE = U'*Wx*U;
    DX = OX.*((SX-SXE).^2);
%     llh5 = sum(sum(tril(DX,-1)));
    llh5 = sum(sum(triu(DX,1)));
%     toc
%     tic
%     llh5 = 0
%     for j=1:N
%         for i=j+1:N
%             llh5 = llh5 + (SX(i,j) - U(:,i)'*Wx*U(:,j))^2;
%         end
%     end
    llh5 = -llh5/(2*hyper.thetax);
%     toc
    % 6. P(Sy)
%     llh6 = 0;
%     for j=1:N
%         for i=j+1:N
%             llh6 = llh6 + (SY(i,j) - V(:,i)'*Wy*V(:,j))^2;
%         end
%     end
    SYE = V'*Wy*V;
    DY = OY.*((SY-SYE).^2);
%     llh6 = sum(sum(tril(DY,-1)));
    llh6 = sum(sum(triu(DY,1)));
    llh6 = -llh6/(2*hyper.thetay);
    % 7. P(Sxy)
    llh7 = 0;
    id = find(O>0);
    Th = logsig(U'*V);
    [nrows,ncols] = ind2sub([N,M],id);
    for i = 1:length(id)        
%         p = logsig(U(:,nrow)'*V(:,ncol));
        p = Th(nrows(i),ncols(i));
        if SXY(nrows(i), ncols(i))>0
            llh7 = llh7 + p;
        else
            llh7 = llh7 + (1 - p);
        end
    end
    llh = llh1+llh2+llh3+llh4+llh5+llh6+llh7;
end