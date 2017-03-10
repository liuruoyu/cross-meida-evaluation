function results = LCFS(settings)

addpath('./lcfs');
results = cell(length(settings.bmarks),1);

for i = 1:length(results)
    load(fullfile(settings.bmark_path,settings.bmarks{i},'img_caffenet.mat'),...
        'ifea_tr','ifea_te');
    load(fullfile(settings.bmark_path,settings.bmarks{i},'txt_word2vec.mat'),...
        'tfea_tr','tfea_te');
    load(fullfile(settings.bmark_path,settings.bmarks{i},'gnd.mat'),...
        'gnd_tr','gnd_te');
    load(fullfile(settings.bmark_path,settings.bmarks{i},'cat.mat'),...
        'catIdx1','catIdx2');
    MAPsi2t1 = zeros(1,settings.iterN);
    MAPst2i1 = zeros(1,settings.iterN);
    MAPsi2t2 = zeros(1,settings.iterN);
    MAPst2i2 = zeros(1,settings.iterN);
    cmcsi2t1 = [];
    cmcst2i1 = [];
    cmcsi2t2 = [];
    cmcst2i2 = [];
    try
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
            
            ifea1 = [];
            tfea1 = [];
            label1 = [];
            for k = 1:size(gnd_tr1,1)
                ifea1 = [ifea1, ifea_tr1(:,logical(gnd_tr1(k,:)))];
                tfea1 = [tfea1, tfea_tr1(:,logical(gnd_tr1(k,:)))];
                label1 = [label1; k*ones(sum(gnd_tr1(k,:)),1)];
            end
            ifea1 = ifea1';
            tfea1 = tfea1';
            
            ifea_tr1 = ifea_tr1';
            tfea_tr1 = tfea_tr1';
            ifea_te1 = ifea_te1';
            tfea_te1 = tfea_te1';
            
            ifea_tr2 = ifea_tr2';
            tfea_tr2 = tfea_tr2';
            ifea_te2 = ifea_te2';
            tfea_te2 = tfea_te2';
            
            lambda_1 = 0.1;
            lambda_2 = 0.001;
            ite = 5;
            
            [Wi, Wt] = LCFS_ite(ifea1, tfea1, label1, lambda_1, lambda_2, ite);
            
            LCFSi_tr1 = ifea_tr1*Wi;
            LCFSt_tr1 = tfea_tr1*Wt;
            LCFSi_te1 = ifea_te1*Wi;
            LCFSt_te1 = tfea_te1*Wt;
            LCFSi_tr2 = ifea_tr2*Wi;
            LCFSt_tr2 = tfea_tr2*Wt;
            LCFSi_te2 = ifea_te2*Wi;
            LCFSt_te2 = tfea_te2*Wt;
            
            num_tr1 = size(ifea_tr1,1);
            num_tr2 = size(ifea_tr2,1);
            
            [MAPi2t1,~,resi2t1] = mAPEvaluate(LCFSi_te1,LCFSt_tr1,gnd_te1,gnd_tr1,num_tr1,'cosine');
            [MAPt2i1,~,rest2i1] = mAPEvaluate(LCFSt_te1,LCFSi_tr1,gnd_te1,gnd_tr1,num_tr1,'cosine');
            [MAPi2t2,~,resi2t2] = mAPEvaluate(LCFSi_te2,LCFSt_tr2,gnd_te2,gnd_tr2,num_tr2,'cosine');
            [MAPt2i2,~,rest2i2] = mAPEvaluate(LCFSt_te2,LCFSi_tr2,gnd_te2,gnd_tr2,num_tr2,'cosine');
            
            MAPsi2t1(j) = MAPi2t1;
            MAPst2i1(j) = MAPt2i1;
            MAPsi2t2(j) = MAPi2t2;
            MAPst2i2(j) = MAPt2i2;
            
            cmci2t1 = CMCcurve(resi2t1, gnd_te1, gnd_tr1);
            cmct2i1 = CMCcurve(rest2i1, gnd_te1, gnd_tr1);
            cmci2t2 = CMCcurve(resi2t2, gnd_te2, gnd_tr2);
            cmct2i2 = CMCcurve(rest2i2, gnd_te2, gnd_tr2);
            
            if j == 1
                cmcsi2t1 = [cmcsi2t1;cmci2t1];
                cmcst2i1 = [cmcst2i1;cmct2i1];
                cmcsi2t2 = [cmcsi2t2;cmci2t2];
                cmcst2i2 = [cmcst2i2;cmct2i2];
            else
                num_min = min(size(cmcsi2t1,2),size(cmci2t1,2));
                cmcsi2t1 = [cmcsi2t1(:,1:num_min);cmci2t1(1:num_min)];
                cmcst2i1 = [cmcst2i1(:,1:num_min);cmct2i1(1:num_min)];
                num_min = min(size(cmcsi2t2,2),size(cmci2t2,2));
                cmcsi2t2 = [cmcsi2t2(:,1:num_min);cmci2t2(1:num_min)];
                cmcst2i2 = [cmcst2i2(:,1:num_min);cmct2i2(1:num_min)];    
            end
        end
        
        result.bmark = settings.bmarks{i};
        result.method = 'LCFS';
        result.MAPi2t1 = mean(MAPsi2t1);
        result.MAPt2i1 = mean(MAPst2i1);
        result.MAPi2t2 = mean(MAPsi2t2);
        result.MAPt2i2 = mean(MAPst2i2);
        result.cmci2t1 = mean(cmcsi2t1,1);
        result.cmct2i1 = mean(cmcst2i1,1);
        result.cmci2t2 = mean(cmcsi2t2,1);
        result.cmct2i2 = mean(cmcst2i2,1);
        
        disp(result);
        results{i} = result;
        
    catch ME
        disp(ME);
        results{i} = 0;
    end
end

rmpath('./lcfs');

end

