function results = SEPHkm(settings)

addpath(genpath('./3rd-party/markSchmidt'));
addpath('./seph');
results = cell(length(settings.bmarks),max(1,length(settings.codelens)));

for i = 1:size(results,1) 
    load(fullfile(settings.bmark_path,settings.bmarks{i},'img_caffenet.mat'),...
        'ifea_tr','ifea_te');
    load(fullfile(settings.bmark_path,settings.bmarks{i},'txt_word2vec.mat'),...
        'tfea_tr','tfea_te');
    load(fullfile(settings.bmark_path,settings.bmarks{i},'gnd.mat'),...
        'gnd_tr','gnd_te');
    load(fullfile(settings.bmark_path,settings.bmarks{i},'cat.mat'),...
        'catIdx1','catIdx2');
    MAPsi2t1 = zeros(max(1,length(settings.codelens)),settings.iterN);
    MAPst2i1 = zeros(max(1,length(settings.codelens)),settings.iterN);
    MAPsi2t2 = zeros(max(1,length(settings.codelens)),settings.iterN);
    MAPst2i2 = zeros(max(1,length(settings.codelens)),settings.iterN);
    cmcsi2t1 = cell(max(1,length(settings.codelens)),1);
    cmcst2i1 = cell(max(1,length(settings.codelens)),1);
    cmcsi2t2 = cell(max(1,length(settings.codelens)),1);
    cmcst2i2 = cell(max(1,length(settings.codelens)),1);
    
    for j = 1:settings.iterN
        cat_idx1 = catIdx1{j};
        cat_idx2 = catIdx2{j};
        tr_idx1 = find(sum(gnd_tr(cat_idx1,:),1)>0);
        tr_idx2 = find(sum(gnd_tr(cat_idx2,:),1)>0);
        te_idx1 = find(sum(gnd_te(cat_idx1,:),1)>0);
        te_idx2 = find(sum(gnd_te(cat_idx2,:),1)>0);
        ifea_tr1 = ifea_tr(:,tr_idx1);
        tfea_tr1 = tfea_tr(:,tr_idx1);
        gnd_tr1 = gnd_tr(cat_idx1,tr_idx1);
        ifea_te1 = ifea_te(:,te_idx1);
        tfea_te1 = tfea_te(:,te_idx1);
        gnd_te1 = gnd_te(cat_idx1,te_idx1);
        ifea_tr2 = ifea_tr(:,tr_idx2);
        tfea_tr2 = tfea_tr(:,tr_idx2);
        gnd_tr2 = gnd_tr(cat_idx2,tr_idx2);
        ifea_te2 = ifea_te(:,te_idx2);
        tfea_te2 = tfea_te(:,te_idx2);
        gnd_te2 = gnd_te(cat_idx2,te_idx2);
        
        ifea_tr1 = ifea_tr1';
        tfea_tr1 = tfea_tr1';
        ifea_te1 = ifea_te1';
        tfea_te1 = tfea_te1';
        ifea_tr2 = ifea_tr2';
        tfea_tr2 = tfea_tr2';
        ifea_te2 = ifea_te2';
        tfea_te2 = tfea_te2';
        
        Meani = mean(ifea_tr1,1);
        Meant = mean(tfea_tr1,1);
        
        ifea_tr1 = bsxfun(@minus, ifea_tr1, Meani);
        tfea_tr1 = bsxfun(@minus, tfea_tr1, Meant);
        ifea_te1 = bsxfun(@minus, ifea_te1, Meani);
        tfea_te1 = bsxfun(@minus, tfea_te1, Meant);
        ifea_tr2 = bsxfun(@minus, ifea_tr2, Meani);
        tfea_tr2 = bsxfun(@minus, tfea_tr2, Meant);
        ifea_te2 = bsxfun(@minus, ifea_te2, Meani);
        tfea_te2 = bsxfun(@minus, tfea_te2, Meant);
        
        Model = {};
        Model.alpha = 1e-2;
        
        num_tr = size(ifea_tr1,1);
        idx_tr = 1:num_tr;
        ifea = ifea_tr1;
        tfea = tfea_tr1;
        if num_tr > 5000
            idx_tr = sort(randperm(num_tr,5000));
            ifea = ifea_tr1(idx_tr,:);
            tfea = tfea_tr1(idx_tr,:);
            num_tr = 5000;
        end
        P = zeros(num_tr,num_tr);
        trainLabels = gnd_tr';
        trainLabels = trainLabels(idx_tr,:);
        if size(trainLabels, 2) > 1
            % cosine similarity for multi-label cases
            num1 = 1 ./ sqrt(sum(trainLabels .* trainLabels, 2));
            num1(isinf(num1) | isnan(num1)) = 1;
            trainLabels = diag(num1) * trainLabels;
            P = trainLabels * trainLabels';
        else
            % binary similarity (i.e. 0 and 1) for multi-class cases
            for ti = 1 : trainNum
                P(ti, :) = double(trainLabels == trainLabels(ti))';
            end
        end
        
        % RBF Kernel
        zi = ifea * ifea';
        zi = repmat(diag(zi), 1, num_tr) + repmat(diag(zi)', num_tr, 1) - 2 * zi;
        ki = {};
        ki.type = 0;
        ki.param = mean(zi(:));
        
        zt = tfea * tfea';
        zt = repmat(diag(zt), 1, num_tr) + repmat(diag(zt)', num_tr, 1) - 2 * zt;
        kt = {};
        kt.type = 0;
        kt.param = mean(zt(:));
        
        kernelSampleNum = 500;
        if kernelSampleNum > num_tr
            kernelSampleNum = num_tr;
        end
        opts = statset('Display', 'off', 'MaxIter', 100);
        [INX, C] = kmeans(ifea, kernelSampleNum, 'Start', 'sample', 'EmptyAction',...
            'singleton', 'Options', opts, 'OnlinePhase', 'off');
        ifea_kn = C;
        
        [INX, C] = kmeans(tfea, kernelSampleNum, 'Start', 'sample', 'EmptyAction',...
            'singleton', 'Options', opts, 'OnlinePhase', 'off');
        tfea_kn = C;
                
        K0i = kernelMatrix(ifea_kn, ifea_kn, ki);
        K0t = kernelMatrix(tfea_kn, tfea_kn, kt);
        trainKi = kernelMatrix(ifea, ifea_kn, ki);
        trainKt = kernelMatrix(tfea, tfea_kn, kt);
        testKi_tr1 = kernelMatrix(ifea_tr1, ifea_kn, ki);
        testKt_tr1 = kernelMatrix(tfea_tr1, tfea_kn, kt);
        testKi_te1 = kernelMatrix(ifea_te1, ifea_kn, ki);
        testKt_te1 = kernelMatrix(tfea_te1, tfea_kn, kt);
        testKi_tr2 = kernelMatrix(ifea_tr2, ifea_kn, ki);
        testKt_tr2 = kernelMatrix(tfea_tr2, tfea_kn, kt);
        testKi_te2 = kernelMatrix(ifea_te2, ifea_kn, ki);
        testKt_te2 = kernelMatrix(tfea_te2, tfea_kn, kt);
        
        if isempty(settings.codelens)
            settings.codelens = size(gnd_tr1,1);
        end
        
        for k = 1:size(results,2)
            try
            bit = settings.codelens(k);
            
            ydata = minKLD(Model.alpha/bit/num_tr, P, bit);
            trainH = sign(ydata);
            
            SEPHrndi_tr1 = zeros(size(ifea_tr1,1), bit);
            SEPHrndt_tr1 = zeros(size(tfea_tr1,1), bit);
            SEPHrndi_te1 = zeros(size(ifea_te1,1), bit);
            SEPHrndt_te1 = zeros(size(tfea_te1,1), bit);
            SEPHrndi_tr2 = zeros(size(ifea_tr2,1), bit);
            SEPHrndt_tr2 = zeros(size(tfea_tr2,1), bit);
            SEPHrndi_te2 = zeros(size(ifea_te2,1), bit);
            SEPHrndt_te2 = zeros(size(tfea_te2,1), bit);
            
            options.Display = 'final';
            C = 0.01;
            
            for l = 1:bit
                tH = trainH(:, l);
                % View 1 (Image View)
                funObj = @(u)LogisticLoss(u, trainKi, tH);
                w = minFunc(@penalizedKernelL2, zeros(size(K0i, 1),1), options, K0i, funObj, C);
                SEPHrndi_tr1(:, l) = sign(testKi_tr1 * w);
                SEPHrndi_te1(:, l) = sign(testKi_te1 * w);
                SEPHrndi_tr2(:, l) = sign(testKi_tr2 * w);
                SEPHrndi_te2(:, l) = sign(testKi_te2 * w);

                % View 2 (Text View)
                funObj = @(u)LogisticLoss(u, trainKt, tH);
                w = minFunc(@penalizedKernelL2, zeros(size(K0t, 1),1), options, K0t, funObj, C);
                SEPHrndt_tr1(:, l) = sign(testKt_tr1 * w);
                SEPHrndt_te1(:, l) = sign(testKt_te1 * w);
                SEPHrndt_tr2(:, l) = sign(testKt_tr2 * w);
                SEPHrndt_te2(:, l) = sign(testKt_te2 * w);
            end
            
            num_tr1 = size(ifea_tr1,1);
            num_tr2 = size(ifea_tr2,1);
            
            [MAPi2t1,~,resi2t1] = mAPEvaluate(SEPHrndi_te1,SEPHrndt_tr1,gnd_te1,gnd_tr1,num_tr1,'hamming');
            [MAPt2i1,~,rest2i1] = mAPEvaluate(SEPHrndt_te1,SEPHrndi_tr1,gnd_te1,gnd_tr1,num_tr1,'hamming');
            [MAPi2t2,~,resi2t2] = mAPEvaluate(SEPHrndi_te2,SEPHrndt_tr2,gnd_te2,gnd_tr2,num_tr2,'hamming');
            [MAPt2i2,~,rest2i2] = mAPEvaluate(SEPHrndt_te2,SEPHrndi_tr2,gnd_te2,gnd_tr2,num_tr2,'hamming');
            
            MAPsi2t1(k,j) = MAPi2t1;
            MAPst2i1(k,j) = MAPt2i1;
            MAPsi2t2(k,j) = MAPi2t2;
            MAPst2i2(k,j) = MAPt2i2;
            
            cmci2t1 = CMCcurve(resi2t1, gnd_te1, gnd_tr1);
            cmct2i1 = CMCcurve(rest2i1, gnd_te1, gnd_tr1);
            cmci2t2 = CMCcurve(resi2t2, gnd_te2, gnd_tr2);
            cmct2i2 = CMCcurve(rest2i2, gnd_te2, gnd_tr2);
            
            if j == 1
                cmcsi2t1{k} = [cmcsi2t1{k};cmci2t1];
                cmcst2i1{k} = [cmcst2i1{k};cmct2i1];
                cmcsi2t2{k} = [cmcsi2t2{k};cmci2t2];
                cmcst2i2{k} = [cmcst2i2{k};cmct2i2];
            else
                num_min = min(size(cmcsi2t1{k},2),size(cmci2t1,2));
                cmcsi2t1{k} = [cmcsi2t1{k}(:,1:num_min);cmci2t1(1:num_min)];
                cmcst2i1{k} = [cmcst2i1{k}(:,1:num_min);cmct2i1(1:num_min)];
                num_min = min(size(cmcsi2t2{k},2),size(cmci2t2,2));
                cmcsi2t2{k} = [cmcsi2t2{k}(:,1:num_min);cmci2t2(1:num_min)];
                cmcst2i2{k} = [cmcst2i2{k}(:,1:num_min);cmct2i2(1:num_min)];    
            end
            
            catch ME
                disp(ME);
                MAPsi2t1(k,j) = 0;
                MAPst2i1(k,j) = 0;
                MAPsi2t2(k,j) = 0;
                MAPst2i2(k,j) = 0;
            end
        end   
    end
    
    for j = 1:size(results,2)
        result.bmark = settings.bmarks{i};
        result.method = 'SEPHkm';
        result.codelen = settings.codelens(j);
        result.MAPi2t1 = mean(MAPsi2t1(j,:));
        result.MAPt2i1 = mean(MAPst2i1(j,:));
        result.MAPi2t2 = mean(MAPsi2t2(j,:));
        result.MAPt2i2 = mean(MAPst2i2(j,:));
        result.cmci2t1 = mean(cmcsi2t1{j},1);
        result.cmct2i1 = mean(cmcst2i1{j},1);
        result.cmci2t2 = mean(cmcsi2t2{j},1);
        result.cmct2i2 = mean(cmcst2i2{j},1);
        
        disp(result);
        results{i,j} = result;
    end
end

rmpath(genpath('./3rd-party/markSchmidt'));
rmpath('./seph');

end

