function results = SCMorth(settings)

addpath('./scm');
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
        
        i_mean = mean(ifea_tr1,2);
        t_mean = mean(tfea_tr1,2);
        i_var = sqrt(var(ifea_tr1,0,2));
        t_var = sqrt(var(tfea_tr1,0,2));
        
        ifea_tr1 = double(bsxfun(@minus, ifea_tr1, i_mean));
        tfea_tr1 = double(bsxfun(@minus, tfea_tr1, t_mean));
        ifea_te1 = double(bsxfun(@minus, ifea_te1, i_mean));
        tfea_te1 = double(bsxfun(@minus, tfea_te1, t_mean));
        ifea_tr2 = double(bsxfun(@minus, ifea_tr2, i_mean));
        tfea_tr2 = double(bsxfun(@minus, tfea_tr2, t_mean));
        ifea_te2 = double(bsxfun(@minus, ifea_te2, i_mean));
        tfea_te2 = double(bsxfun(@minus, tfea_te2, t_mean));
        
        ifea_tr1 = ifea_tr1./repmat(i_var,1,size(ifea_tr1,2));
        tfea_tr1 = tfea_tr1./repmat(t_var,1,size(tfea_tr1,2));
        ifea_te1 = ifea_te1./repmat(i_var,1,size(ifea_te1,2));
        tfea_te1 = tfea_te1./repmat(t_var,1,size(tfea_te1,2));
        ifea_tr2 = ifea_tr2./repmat(i_var,1,size(ifea_tr2,2));
        tfea_tr2 = tfea_tr2./repmat(t_var,1,size(tfea_tr2,2));
        ifea_te2 = ifea_te2./repmat(i_var,1,size(ifea_te2,2));
        tfea_te2 = tfea_te2./repmat(t_var,1,size(tfea_te2,2));
        
        ifea_tr1(isnan(ifea_tr1)) = 0;
        tfea_tr1(isnan(tfea_tr1)) = 0;
        ifea_te1(isnan(ifea_te1)) = 0;
        tfea_te1(isnan(tfea_te1)) = 0;
        ifea_tr2(isnan(ifea_tr2)) = 0;
        tfea_tr2(isnan(tfea_tr2)) = 0;
        ifea_te2(isnan(ifea_te2)) = 0;
        tfea_te2(isnan(tfea_te2)) = 0;
        
        ifea_tr1 = ifea_tr1';
        tfea_tr1 = tfea_tr1';
        ifea_te1 = ifea_te1';
        tfea_te1 = tfea_te1';
        ifea_tr2 = ifea_tr2';
        tfea_tr2 = tfea_tr2';
        ifea_te2 = ifea_te2';
        tfea_te2 = tfea_te2';
        
        if isempty(settings.codelens)
            settings.codelens = size(gnd_tr1,1);
        end
        
        for k = 1:size(results,2)
            try
            [Wi, Wt] = trainSCM_orth(ifea_tr1, tfea_tr1, settings.codelens(k), gnd_tr1');
            SCMoi_tr1 = (ifea_tr1*Wi)'>=0;
            SCMot_tr1 = (tfea_tr1*Wt)'>=0;
            SCMoi_te1 = (ifea_te1*Wi)'>=0;
            SCMot_te1 = (tfea_te1*Wt)'>=0;
            SCMoi_tr2 = (ifea_tr2*Wi)'>=0;
            SCMot_tr2 = (tfea_tr2*Wt)'>=0;
            SCMoi_te2 = (ifea_te2*Wi)'>=0;
            SCMot_te2 = (tfea_te2*Wt)'>=0;
            
            num_tr1 = size(ifea_tr1,1);
            num_tr2 = size(ifea_tr2,1);
            
            [MAPi2t1,~,resi2t1] = mAPEvaluate(SCMoi_te1',SCMot_tr1',gnd_te1,gnd_tr1,num_tr1,'hamming');
            [MAPt2i1,~,rest2i1] = mAPEvaluate(SCMot_te1',SCMoi_tr1',gnd_te1,gnd_tr1,num_tr1,'hamming');
            [MAPi2t2,~,resi2t2] = mAPEvaluate(SCMoi_te2',SCMot_tr2',gnd_te2,gnd_tr2,num_tr2,'hamming');
            [MAPt2i2,~,rest2i2] = mAPEvaluate(SCMot_te2',SCMoi_tr2',gnd_te2,gnd_tr2,num_tr2,'hamming');
            
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
        result.method = 'SCMorth';
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

rmpath('./scm');

end

