function makedata_wiki

addpath utils

do_split = 0;
do_sim = 1;
do_nbm = 0;
do_landmark = 0;
do_observation = 0;
do_centers = 0;

% we use largest 5 classes, for each class 80% as training, 20% as testing
nclass = 10;

if do_split == 1
    disp('Generating training and testing.');
    load('../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_TT');
    % we have feax_tr, feax_te, feay_tr, feay_te, gnd_tr, gnd_te
    % each column is a point
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
        clabel = zeros(nclass,1);
        clabel(i) = 1;
        cind = find(gnd(ind(i),:)>0);% index of points belonging to classi ind(i)
        rind = randperm(length(cind));
        trsize = floor(length(cind)*0.8);
        feax_tr = [feax_tr feax(:,cind(rind(1:trsize)))];
        feax_te = [feax_te feax(:,cind(rind(trsize+1:end)))];
        feay_tr = [feay_tr feay(:,cind(rind(1:trsize)))];
        feay_te = [feay_te feay(:,cind(rind(trsize+1:end)))];
        gnd_tr = [gnd_tr repmat(clabel,1,trsize)];
        gnd_te = [gnd_te repmat(clabel,1,length(cind)-trsize)];
    end

    save(sprintf('../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_TT%d',nclass),...
        'feax_tr', 'feax_te', 'feay_tr', 'feay_te', 'gnd_tr', 'gnd_te');
    
end

if do_sim == 1
    disp('Generating similarities and neighbourhood.');
    load(sprintf('../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_TT%d',nclass));
    % we have feax_tr, feax_te, feay_tr, feay_te, gnd_tr, gnd_te
    % each column is a point
    % we compute the similarity between all points
%     feax = [feax_tr feax_te];
%     feay = [feay_tr feay_te];
%     gnd = [gnd_tr gnd_te];
    
%     % since feax is SIFT, we normalize it to make each point has norm 1
%     normv = normmat_col(feax);
%     if length(find(normv<=0))>0
%         disp('wrong points!');
%     else
%         feax1 = double(bsxfun(@rdivide, feax, normv));
%     end
    npca = 10;
    [pc, l] = eigs(cov(feax_tr'), npca);
    feax_tr = pc'* feax_tr; % no need to remove the mean
    feax_te = pc'* feax_te; % no need to remove the mean

    % we first normlize each point to have norm 1
%     feax_tr = normEachCol(feax_tr);
%     feax_te = normEachCol(feax_te);
%     feay_tr = normEachCol(feay_tr);
%     feay_te = normEachCol(feay_te);
    
%     KNN = 50;
    sigma = 1;% the best
%     simtype = 'EUC';% cos, pearson
    % for X
    [EX_tr] = distMat(feax_tr, feax_tr); 
    SX_tr = exp(-0.5*(EX_tr.^2)/sigma);%UCSD_SIMEUC%d
    simx = prctile(SX_tr(:),95);% use top 5 as NN
    [EX_te] = distMat(feax_te, feax_tr); 
    SX_te = exp(-0.5*(EX_te.^2)/sigma);
    NX_tr = (SX_tr)>simx;
    NX_te = (SX_te)>simx;
%     [EX_tr, NX_tr] = distMat(feax_tr, feax_tr, KNN); 
%     NX_tr = (NX_tr+NX_tr')>0; 
%     SX_tr = (1+EX_tr).^(-1);%UCSD_SIM%d
    
%     [EX_te, NX_te] = distMat(feax_te, feax_tr, KNN); 
%     SX_te = (1+EX_te).^(-1);
    

    % for Y
    [EY_tr] = distMat(feay_tr, feay_tr); 
    SY_tr = exp(-0.5*(EY_tr.^2)/sigma);%UCSD_SIMEUC%d
    simy = prctile(SY_tr(:),95);% use top 5 as NN
    [EY_te] = distMat(feay_te, feay_tr); 
    SY_te = exp(-0.5*(EY_te.^2)/sigma);
    NY_tr = (SY_tr)>simy;
    NY_te = (SY_te)>simy;
%     [EY_tr, NY_tr] = distMat(feay_tr, feay_tr, KNN); 
%     NY_tr = (NY_tr+NY_tr')>0; 
% %     SY_tr = (1+EY_tr).^(-1);
%     SY_tr = exp(-0.5*(EY_tr.^2)/sigma);%UCSD_SIMEUC%d
%     [EY_te, NY_te] = distMat(feay_te, feay_tr, KNN); 
% %     SY_te = (1+EY_te).^(-1);
%     SY_te = exp(-0.5*(EY_te.^2)/sigma);
    
    % for XY
    
    SXY_tr = (NX_tr + NY_tr)>0;
    SXY_te = (NX_te + NY_te)>0;    
    
    GID_te = find(sum(SXY_te,2)>=50);
    
%     EX = distMat(feax, feax); 
%     % change distance to similarity
%     SX = (1+EX).^(-1);
% %     SX1 = feax1'*feax1;
%     EY = distMat(feay, feay);
%     % change distance to similarity
%     SY = (1+EY).^(-1);
%     SXY = gnd'*gnd;
%     save(sprintf('../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_SIMEUC%d_RNNG',nclass),...
%         'SX_tr', 'SY_tr', 'SX_te', 'SY_te', 'SXY_tr', 'SXY_te','NX_tr','NX_te','NY_tr','NY_te','GID_te');
end

if do_nbm == 1
    % this part is not necessary. We have defined SXY_te as the NBM above.
    disp('Generating neighborhood.');
    load(sprintf('../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_TT%d',nclass));
    % we have feax_tr, feax_te, feay_tr, feay_te, gnd_tr, gnd_te
    % each column is a point
    % we compute the neighbor relations between train and test
    
    NBM = gnd_te'*gnd_tr;
    
    save(sprintf('../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_NBM%d',nclass),'NBM');
end

if do_landmark ==1
    load(sprintf('../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_TT%d',nclass));
    % 'Xlmidx'+'Xtridx' = training database for X, lmidx and tridx are
    % exclusive
    % 'Ylmidx','Ytridx' = training database for X
    % we generate 20 random splits, each split is a column
    lmperClass = 40;
    lmsize = nclass*lmperClass;
    Xlmidx = []; Xtridx = []; Ylmidx = []; Ytridx = [];
    for i =1:20
        rdidxX = randperm(size(feax_tr,2));
        Xlmidx = [Xlmidx rdidxX(1:lmsize)']; Xtridx = [Xtridx rdidxX(1+lmsize:end)'];
        rdidxY = randperm(size(feay_tr,2));
        Ylmidx = [Ylmidx rdidxY(1:lmsize)']; Ytridx = [Ytridx rdidxY(1+lmsize:end)'];
    end
    save(sprintf('../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_LMIDX%d_%d', nclass, lmperClass),'Xlmidx','Xtridx','Ylmidx','Ytridx');
end

if do_observation ==1
    load(sprintf('../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_TT%d',nclass));
    % we generate 20 random observations
    obratio = 0.001;
    OXYs = cell(20,1);% observations on all training
    nX = size(feax_tr,2);
    nY = size(feay_tr,2);
    for i =1:20
        OXYs{i} = getobservations(nX, nY,obratio);
    end
    save(sprintf('../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_OB%d_%.4f.mat', nclass, obratio),'OXYs');
end

if do_centers == 1
    load(sprintf('../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_TT%d',nclass));
%     load(sprintf('../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_SIMEUC%d',nclass));
    lmperClass = 5;
    lmsize = nclass*lmperClass;%
    [lable e Xlmidx] = kmedoids(feax_tr,lmsize);Xtridx = setdiff(1:size(feax_tr,2),Xlmidx)';
    [lable e Ylmidx] = kmedoids(feay_tr,lmsize);Ytridx = setdiff(1:size(feay_tr,2),Ylmidx)';
    save(sprintf('../../MH_TMM11/Python-1/HashExp/data/ucsd/UCSD_CTIDX%d_%d', nclass, lmperClass),'Xlmidx','Xtridx','Ylmidx','Ytridx');
end

disp('Program ends.');
end

function X1 = normEachCol(X)
    normv = normmat_col(X);
    if ~isempty(find(normv<=0))
        disp('wrong points!');
    else
        X1 = double(bsxfun(@rdivide, X, normv));
    end
end