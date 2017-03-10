function mh_process_data()

% mtype


addpath utils;
% addpath lightspeed;

do_ucsd = 0;
do_nus = 0;
do_toyucsd = 1;
do_fake = 0;
% for ucsd
if do_ucsd
datafile = '../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_TT';
outputfile = '../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_SimMat.mat';
end
% for nus
if do_nus
datafile = '../../MH_TMM11/Python-1/HashExp/data/nus/SNUS10_TT';
outputfile = '../../MH_TMM11/Python-1/HashExp/data/nus/SNUS10_SimMat.mat';
end
if do_toyucsd
datafile = '../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_TT';
outputfile = '../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_TOY3.mat';
end
if do_fake
datafile = '../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_TT';
outputfile = '../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_FAKE.mat';
end


% neighborfile = '../..//MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_NBM';
load(datafile)% gnd_tr, gnd_te, feax_tr, feax_te, feay_tr, feay_te
% load(ProbParam.neighborfile);% load neighborhood file: NBM [# of test x # of train]

% % each point is a columen
% [DimX nTrain] = size(feax_tr);
% [DimY nTest] = size(feay_te);
% % 1. mean-center the data (no need, the data have been mean-centered)
% X = [feax_tr feax_te];
% Y = [feay_tr feay_te];
% gnd = [gnd_tr gnd_te];
% mX = mean(X,2);
% mY = mean(Y,2);
% X = bsxfun(@minus, X, mX);%X = X - ones(1, size(X,2))*mean(X);
% Y = bsxfun(@minus, Y, mY);
% 
% % 2. normlize data
% nX = sqrt(sum(X.^2,1));
% nY = sqrt(sum(Y.^2,1));
% X = bsxfun(@rdivide, X, nX); 
% Y = bsxfun(@rdivide, Y, nY);


if do_toyucsd<1
    % 3. pca
    % for X
%     npcaX = 50; npcaX = min(npcaX, DimX); 
%     Xtr = X(:,1:nTrain); Xte = X(:, nTrain+1:end);
%     opts.disp = 0;
%     [pc, ~] = eigs(cov(Xtr'),npcaX,'LM',opts);
%     Xtr2 = pc'*Xtr; Xte2 = pc'*Xte;
%     X2 = [Xtr2 Xte2];
%     % for Y, if ucsd, no need for PCA
%     if do_nus
%     npcaY = 100; npcaY = min(npcaY, DimY); 
%     Ytr = Y(:,1:nTrain); Yte = Y(:, nTrain+1:end);
%     [pc, ~] = eigs(cov(Ytr'),npcaY,'LM',opts);
%     Ytr2 = pc'*Ytr; Yte2 = pc'*Yte;
%     Y2 = [Ytr2 Yte2];
%     end
% 
%     % 4. compute the similarity
%     % for X (image), euclidean
%     SX = distMat(X2);
%     maxDist = max(SX(:));
%     if maxDist > 1
%         SX = SX / maxDist;% scale to [0,1]
%     end
%     SX = 1 - SX; % change to similarity
%     % for Y (text), cosine 
%     % SY = Y'*Y;% Y is normalized
%     % for Y (text), euclidean
%     if do_nus
%         SY = distMat(Y2);
%     end
%     if do_ucsd
%         SY = distMat(Y);
%     end
%     maxDist = max(SY(:));
%     if maxDist > 1
%         SY = SY / maxDist;% scale to [0,1]
%     end

else% make toy
    % each point is a columen
    Ntotal = 500;Ntrain = 300;Ntest=200;Nref = 300;
    X = feax_tr(:,1:Ntotal);
    Y = feay_tr(:,1:Ntotal);
    L = gnd_tr(:,1:Ntotal);
%     [DimX nTrain] = size(feax_tr);
%     [DimY nTest] = size(feay_te);
    % 1. mean-center the data (no need, the data have been mean-centered)
%     X = [feax_tr feax_te];
%     Y = [feay_tr feay_te];
%     gnd = [gnd_tr gnd_te];
    mX = mean(X,2); X = bsxfun(@minus, X, mX);%X = X - ones(1, size(X,2))*mean(X);
    mY = mean(Y,2); Y = bsxfun(@minus, Y, mY);

    % 2. normlize data
    nX = sqrt(sum(X.^2,1));
    nY = sqrt(sum(Y.^2,1));
    X = bsxfun(@rdivide, X, nX); 
    Y = bsxfun(@rdivide, Y, nY);
    
    % 3. compute the similarity
    do_euc = 1;
    do_cos = 1;
    if do_euc == 1
        % for X (image), euclidean
        SX0 = distMat(X);
        maxDist = max(SX0(:));
        if maxDist > 1
            SX0 = SX0 / maxDist;% scale to [0,1]
        end
        SX.euc = 1 - SX0; % change to similarity
        % for Y (text), cosine 
        % SY = Y'*Y;% Y is normalized
        % for Y (text), euclidean
        SY0 = distMat(Y);
        maxDist = max(SY0(:));
        if maxDist > 1
            SY0 = SY0 / maxDist;% scale to [0,1]
        end
        SY.euc = 1 - SY0;
        
        % construct observations on X
    end
    if do_cos == 1
        SX0 = X'*X;% X is normalized
        maxDist = max(SX0(:));
        if maxDist > 1, SX.cos = SX0 / maxDist;% scale to [0,1]
        end
        SY0 = Y'*Y;% Y is normalized
        maxDist = max(SY0(:));
        if maxDist > 1, SY.cos = SY0 / maxDist;% scale to [0,1]
        end
    end
    
    % for cross similarity
    SXY = L'*L>0;
    DNeig = SXY(Ntrain+1:end,1:Ntrain);
    
    % for observation
    op = 0.001;
    nob = 10;
    OXY = cell(nob,1);
    for i = 1:nob
    OXY{i} = getobservations(Ntotal, Ntotal, op);
    end

    % for index, this is generated to RDIDX file
%     Nrepeat = 10;
%     [Xtrainidx, Xtestidx, Xrefidx, Ytrainidx, Ytestidx, Yrefidx]...
%     = generateRDIDX(Ntrain,Ntest,Nref,Nrepeat);

end


% 5. save to file
save(outputfile,'SX','SY','SXY','OXY','DNeig','X','Y','L');

end