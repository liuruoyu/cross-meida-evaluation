function results = MMNN_m(settings)

addpath(genpath('./3rd-party/nntoolbox'));
addpath('./MMNNhash');
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
        
        S = gnd_tr1'*gnd_tr1>0;
        num_tr = size(ifea_tr1,2);
        k_pos = find(S == 1);
        k_pos = k_pos(randperm(length(k_pos)));
        k_neg = find(S == 0);
        k_neg = k_neg(randperm(length(k_neg)));
        [Xp, Yp] = ind2sub(size(S),k_pos);
        [Xn, Yn] = ind2sub(size(S),k_neg);
        [ifea, i_Mean] = center_unit_norm(ifea_tr1,'cols','double');
        [tfea, t_Mean] = center_unit_norm(tfea_tr1,'cols','double');
        
        ifea_tr1 = bsxfun(@plus, feval('double', ifea_tr1), - i_Mean);
        ifea_tr1 = feval('double', normL2(double(ifea_tr1), 'cols', 1000));
        tfea_tr1 = bsxfun(@plus, feval('double', tfea_tr1), - t_Mean);
        tfea_tr1 = feval('double', normL2(double(tfea_tr1), 'cols', 1000));
        ifea_te1 = bsxfun(@plus, feval('double', ifea_te1), - i_Mean);
        ifea_te1 = feval('double', normL2(double(ifea_te1), 'cols', 1000));
        tfea_te1 = bsxfun(@plus, feval('double', tfea_te1), - t_Mean);
        tfea_te1 = feval('double', normL2(double(tfea_te1), 'cols', 1000));
        ifea_tr2 = bsxfun(@plus, feval('double', ifea_tr2), - i_Mean);
        ifea_tr2 = feval('double', normL2(double(ifea_tr2), 'cols', 1000));
        tfea_tr2 = bsxfun(@plus, feval('double', tfea_tr2), - t_Mean);
        tfea_tr2 = feval('double', normL2(double(tfea_tr2), 'cols', 1000));
        ifea_te2 = bsxfun(@plus, feval('double', ifea_te2), - i_Mean);
        ifea_te2 = feval('double', normL2(double(ifea_te2), 'cols', 1000));
        tfea_te2 = bsxfun(@plus, feval('double', tfea_te2), - t_Mean);
        tfea_te2 = feval('double', normL2(double(tfea_te2), 'cols', 1000));
        
        x.x1 = [Xp;Xn];
        x.x2 = [Yp;Yn];
        x.t = [ones(1, length(Xp),'single'), zeros(1, length(Xn),'single')]';
        trainIdxs = 1:num_tr;
        
        if isempty(settings.codelens)
            settings.codelens = size(gnd_tr1,1);
        end
        
        for k = 1:size(results,2)
            try
            nbit = settings.codelens(k);
            dim_i = size(ifea,1);
            dim_t = size(tfea,1);
            batchSize = round(0.9*length(trainIdxs));
            
            rng(127, 'twister');
            nnG = GNN_L2reg(lossMSE());
            nnG.addLayer(layerMLP(dim_i, nbit), actTanh(1), 1, 2, 0.00);
            nnG.batchSize = batchSize;
            
            snnG = SNN(nnG, 5);
            snnG.batchSize = batchSize;
            
            rng(127, 'twister');
            nnH = GNN_L2reg(lossMSE());
            nnH.addLayer(layerMLP(dim_t, nbit), actTanh(1), 1, 2, 0.00);
            nnH.batchSize = batchSize;
            
            snnH = SNN(nnH, 5);
            snnH.batchSize = batchSize;
            
            mm = MMNN(snnG, snnH, 5, 0.5, 0.5);
            batchSize_ = round(batchSize/2);
            nBatches_ = 10;
            
            rng(127, 'twister');
            options.MaxIter = 5;
            options.Display = 0;
            nEpochs = 50;
            
            for l = 1:nEpochs
                fprintf('Epoch %i/%i\n',l,nEpochs)
                for m = 1:nBatches_
                    prmL = trainIdxs(randperm(length(trainIdxs), batchSize_));
                    prmR = trainIdxs(randperm(length(trainIdxs), batchSize_));
                    
                    x_.M1.x1 = ifea(:, prmL)';
                    x_.M1.x2 = ifea(:, prmR)';
                    x_.M1.t = double(diag(gnd_tr1(:,prmL)'*gnd_tr1(:,prmR))>0);
                    
                    x_.M2.x1 = tfea(:, prmL)';
                    x_.M2.x2 = tfea(:, prmR)';
                    x_.M2.t = x_.M1.t;
                    
                    x_.CM.x1 = x_.M1.x1;
                    x_.CM.x2 = x_.M2.x2;
                    x_.CM.t = x_.M1.t;
                    
                    minFunc(@(w,x,t)minFuncDoubleGradWrapper(@()mm.grad(w,x,t)),...
                        double(mm.getParams), options, x_, x_.M1.t);
                end
            end
            
            MMNNi_tr1 = double(mm.net1.net.fwdOnly(ifea_tr1',2) > 0);
            MMNNt_tr1 = double(mm.net2.net.fwdOnly(tfea_tr1',2) > 0);
            MMNNi_te1 = double(mm.net1.net.fwdOnly(ifea_te1',2) > 0);
            MMNNt_te1 = double(mm.net2.net.fwdOnly(tfea_te1',2) > 0);
            MMNNi_tr2 = double(mm.net1.net.fwdOnly(ifea_tr2',2) > 0);
            MMNNt_tr2 = double(mm.net2.net.fwdOnly(tfea_tr2',2) > 0);
            MMNNi_te2 = double(mm.net1.net.fwdOnly(ifea_te2',2) > 0);
            MMNNt_te2 = double(mm.net2.net.fwdOnly(tfea_te2',2) > 0);
            
            num_tr1 = size(ifea_tr1,2);
            num_tr2 = size(ifea_tr2,2);
            
            [MAPi2t1,~,resi2t1] = mAPEvaluate(MMNNi_te1,MMNNt_tr1,gnd_te1,gnd_tr1,num_tr1,'hamming');
            [MAPt2i1,~,rest2i1] = mAPEvaluate(MMNNt_te1,MMNNi_tr1,gnd_te1,gnd_tr1,num_tr1,'hamming');
            [MAPi2t2,~,resi2t2] = mAPEvaluate(MMNNi_te2,MMNNt_tr2,gnd_te2,gnd_tr2,num_tr2,'hamming');
            [MAPt2i2,~,rest2i2] = mAPEvaluate(MMNNt_te2,MMNNi_tr2,gnd_te2,gnd_tr2,num_tr2,'hamming');
            
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
        result.method = 'MMNN';
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

rmpath(genpath('./3rd-party/nntoolbox'));
rmpath('./MMNNhash');

end

