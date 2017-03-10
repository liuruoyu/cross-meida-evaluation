function makedata_flickr

addpath utils

do_split = 1;
do_sim = 0;
do_nbm = 0;
do_landmark = 0;
do_observation = 0;
do_centers = 0;

% we use largest 5 classes, for each class 80% as training, 20% as testing
nclass = 10;
lmperClass = 50;

if do_split == 1
    disp('Generating training and testing.');
    load('../../MH_TMM11/Python-1/HashExp/data/nus/SNUS10_TT');
    % we have feax_tr, feax_te, feay_tr, feay_te, gnd_tr, gnd_te
    % each column is a point
    unique(feax_tr)
    feax = [feax_tr feax_te];
    feay = [feay_tr feay_te];
    gnd = [gnd_tr gnd_te];

    np = sum(gnd,2);

    [val ind] = sort(np,'descend');

    feax_tr = [];
    feax_te = [];
    feay_tr = [];
    feay_te = [];
    gnd_tr = [];
    gnd_te = [];

    for i = 1:nclass
%         clabel = zeros(nclass,1);
%         clabel(i) = 1;
        cind = find(gnd(ind(i),:)>0);% index of points belonging to classi ind(i)
        rind = randperm(length(cind));
        trsize = floor(length(cind)*0.99);
        feax_tr = [feax_tr feax(:,cind(rind(1:trsize)))];
        feax_te = [feax_te feax(:,cind(rind(trsize+1:end)))];
        feay_tr = [feay_tr feay(:,cind(rind(1:trsize)))];
        feay_te = [feay_te feay(:,cind(rind(trsize+1:end)))];
        gnd_tr = [gnd_tr gnd(:,cind(rind(1:trsize)))];
        gnd_te = [gnd_te gnd(:,cind(rind(trsize+1:end)))];
        feax(:,cind) = [];
        feay(:,cind) = [];
        gnd(:,cind) = [];
    end

%     save(sprintf('../../MH_TMM11/Python-1/HashExp/data/nus/NUS_TT%d',nclass),...
%         'feax_tr', 'feax_te', 'feay_tr', 'feay_te', 'gnd_tr', 'gnd_te');
end
if do_sim == 1
    disp('Generating similarities and neighbourhood.');
%     lmperClass = 30;
    load(sprintf('../../MH_TMM11/Python-1/HashExp/data/nus/NUS_TT%d',nclass));
    % we have feax_tr, feax_te, feay_tr, feay_te, gnd_tr, gnd_te
    % each column is a point
    % we compute the similarity between all points
%     feax = [feax_tr feax_te];
%     feay = [feay_tr feay_te];
%     gnd = [gnd_tr gnd_te];

    load(sprintf('../../MH_TMM11/Python-1/HashExp/data/nus/NUS_LMIDX%d_%d', nclass, lmperClass));
    % we first normlize each point to have norm 1
    feax_tr = normEachCol(feax_tr);
    feax_te = normEachCol(feax_te);
    feay_tr = normEachCol(feay_tr);
    feay_te = normEachCol(feay_te);
    
    % seperate lm, tr and te
    iXlmidx = Xlmidx(:,1);
    iXtridx = Xtridx(:,1);
    iYlmidx = Ylmidx(:,1);
    iYtridx = Ytridx(:,1);
    
    feaxlm = feax_tr(:,iXlmidx);
    feaxtr = feax_tr(:,iXtridx);
    feaylm = feay_tr(:,iYlmidx);
    feaytr = feay_tr(:,iYtridx);
    
    
    
%     KNN = 50;
    sigma = 1;% the best
%     simtype = 'EUC';% cos, pearson
    disp('For X...');
    % for X
    [EXlm] = distMat(feaxlm, feaxlm); 
    SXlm = exp(-0.5*(EXlm.^2)/sigma);clear EXlm; %NUS_SIMEUC%d
    [EXtr] = distMat(feaxtr, feaxlm); 
    SXtrlm = exp(-0.5*(EXtr.^2)/sigma);clear EXtr; %NUS_SIMEUC%d
    [EXte] = distMat(feax_te, feaxlm); 
    SXtelm = exp(-0.5*(EXte.^2)/sigma);clear EXte; %NUS_SIMEUC%d

    disp('For Y...');
    % for X
    [EYlm] = distMat(feaylm, feaylm); 
    SYlm = exp(-0.5*(EYlm.^2)/sigma);clear EYlm; %NUS_SIMEUC%d
    [EYtr] = distMat(feaytr, feaylm); 
    SYtrlm = exp(-0.5*(EYtr.^2)/sigma);clear EYtr; %NUS_SIMEUC%d
    [EYte] = distMat(feay_te, feaylm); 
    SYtelm = exp(-0.5*(EYte.^2)/sigma);clear EYte; %NUS_SIMEUC%d
        
    save(sprintf('../../MH_TMM11/Python-1/HashExp/data/nus/NUS_SIMEUC%d_%d',nclass,lmperClass),...
       'SXlm', 'SYlm','SXtelm', 'SYtelm','SXtrlm', 'SYtrlm');
end

if do_nbm == 1
    % this part is not necessary. We have defined SXY_te as the NBM above.
    disp('Generating neighborhood.');
    load(sprintf('../../MH_TMM11/Python-1/HashExp/data/nus/NUS_TT%d',nclass));
    % we have feax_tr, feax_te, feay_tr, feay_te, gnd_tr, gnd_te
    % each column is a point
    % we compute the neighbor relations between train and test
    
    NBM = gnd_te'*gnd_tr;
    
    save(sprintf('../../MH_TMM11/Python-1/HashExp/data/nus/NUS_NBM%d',nclass),'NBM');
end

if do_landmark ==1
    load(sprintf('../../MH_TMM11/Python-1/HashExp/data/nus/NUS_TT%d',nclass));
    % 'Xlmidx'+'Xtridx' = training database for X, lmidx and tridx are
    % exclusive
    % 'Ylmidx','Ytridx' = training database for X
    % we generate 20 random splits, each split is a column
%     lmperClass = 50;
    lmsize = nclass*lmperClass;
    Xlmidx = []; Xtridx = []; Ylmidx = []; Ytridx = [];
    for i =1:20
        rdidxX = randperm(size(feax_tr,2));
        Xlmidx = [Xlmidx rdidxX(1:lmsize)']; Xtridx = [Xtridx rdidxX(1+lmsize:end)'];
        rdidxY = randperm(size(feay_tr,2));
        Ylmidx = [Ylmidx rdidxY(1:lmsize)']; Ytridx = [Ytridx rdidxY(1+lmsize:end)'];
    end
    save(sprintf('../../MH_TMM11/Python-1/HashExp/data/nus/NUS_LMIDX%d_%d', nclass, lmperClass),'Xlmidx','Xtridx','Ylmidx','Ytridx');
end

if do_observation ==1
    load(sprintf('../../MH_TMM11/Python-1/HashExp/data/nus/NUS_TT%d',nclass));
    % we generate 20 random observations
    obratio = 1e-3;
%     lmperClass = 50;
%     nclass = 10
    OXYs = cell(2,1);% observations on all training
    nX = size(feax_tr,2);
    nY = size(feay_tr,2);
    nlm = nclass*lmperClass;
    OXYs{1} = getobservations2(nlm, nlm, obratio);
    OXYs{2} = getobservations2(nX-nlm, nlm, obratio);
%     for i =1:1
%         OXYs{i} = getobservations2(nX, nY,obratio);
%     end
    save(sprintf('../../MH_TMM11/Python-1/HashExp/data/nus/NUS_OB%d_%d_small.mat', nclass,lmperClass),'OXYs','obratio');
end

if do_centers == 1
    load(sprintf('../../MH_TMM11/Python-1/HashExp/data/nus/NUS_TT%d',nclass));
%     load(sprintf('../../MH_TMM11/Python-1/HashExp/data/nus/NUS_SIMEUC%d',nclass));
%     lmperClass = 30;
    lmsize = nclass*lmperClass;%
    [ndim np] = size(feax_tr);
    nprocessed = 0;
    blocksize = 1000;
    blockcentersize = 10;
    Xcaidx = []; Ycaidx = [];
    while nprocessed < np
        fprintf('iteration: %d\n', floor(nprocessed/blocksize)+1);
        if nprocessed+blocksize>np
            iprocessedidx = nprocessed+1:np;
            nprocessed = np;
        else
            iprocessedidx = nprocessed+1:nprocessed+blocksize;
            nprocessed = nprocessed + blocksize;
        end
        [lable e iXcaidx] = kmedoids(feax_tr(:,iprocessedidx),blockcentersize);
        [lable e iYcaidx] = kmedoids(feay_tr(:,iprocessedidx),blockcentersize);
        Xcaidx = [Xcaidx iprocessedidx(iXcaidx)];
        Ycaidx = [Ycaidx iprocessedidx(iYcaidx)];        
    end
    [lable e iXlmidx] = kmedoids(feax_tr(:,Xcaidx),lmsize);
    Xlmidx = Xcaidx(iXlmidx);Xtridx = setdiff(1:np,Xlmidx)';
    [lable e iYlmidx] = kmedoids(feay_tr(:,Ycaidx),lmsize);
    Ylmidx = Ycaidx(iYlmidx);Ytridx = setdiff(1:np,Ylmidx)';
    save(sprintf('../../MH_TMM11/Python-1/HashExp/data/nus/NUS_CTIDX%d_%d', nclass, lmperClass),'Xlmidx','Xtridx','Ylmidx','Ytridx');
end

disp('Program ends.');
end

function X1 = normEachCol(X)
    normv = normmat_col(X);
    normv(normv==0) = 0.1;
    X1 = double(bsxfun(@rdivide, X, normv));
%     if ~isempty(find(normv<=0))
%         disp('wrong points!');
%     else
%         X1 = double(bsxfun(@rdivide, X, normv));
%     end
end