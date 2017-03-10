function results = IMH(settings)

addpath('./imh');
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
        
        ifea_tr1 = double(bsxfun(@minus, ifea_tr1, i_mean));
        tfea_tr1 = double(bsxfun(@minus, tfea_tr1, t_mean));
        ifea_te1 = double(bsxfun(@minus, ifea_te1, i_mean));
        tfea_te1 = double(bsxfun(@minus, tfea_te1, t_mean));
        ifea_tr2 = double(bsxfun(@minus, ifea_tr2, i_mean));
        tfea_tr2 = double(bsxfun(@minus, tfea_tr2, t_mean));
        ifea_te2 = double(bsxfun(@minus, ifea_te2, i_mean));
        tfea_te2 = double(bsxfun(@minus, tfea_te2, t_mean));
        
        num_tr = size(ifea_tr1,2);
        n2 = round(num_tr/2);
        idx_tr = randperm(num_tr);
        idx_sh = sort(idx_tr(1:n2));
        idx_nsh = sort(idx_tr(n2+1:end));
        X1 = ifea_tr1(:,[idx_nsh,idx_sh]);
        X2 = tfea_tr1(:,[idx_sh,idx_nsh]);
        para.k = 10;
        para.sigma = 1;
        L1 = Laplacian_GK(X1,para);
        L2 = Laplacian_GK(X2,para);
        
        beta = 1;
        lambda = 1;
        [m1, imgNum] = size(X1);
        [m2, textNum] = size(X2);
        n1 = imgNum - n2;
        n3 = textNum - n2;
        tmp1 = diag(ones(1,n1+n2));
        S1 = tmp1(n1+1:n1+n2,:);clear tmp1;
        tmp2 = diag(ones(1,n2+n3));
        S2 = tmp2(1:n2,:);clear tmp2;
        U = 10000*diag(ones(1,n2));
        
        eyemat_1 = eye(imgNum);
        eyemat_2 = eye(textNum);
        eyemat_m_1 = eye(m1);
        eyemat_m_2 = eye(m2);
        M1 = X1*X1' + beta*eyemat_m_1;
        M2 = X2*X2' + beta*eyemat_m_2;
        A1 = eyemat_1 - (X1')/M1*X1; %B in the paper
        A2 = eyemat_2 - (X2')/M2*X2;
        C2 = (A2+lambda*L2+S2'*U*S2)\S2'*U*S2;%E in the paper
        D = A1+C2'*A2*C2 + (S1-S2*C2)'*U*(S1-S2*C2) + lambda*L1+lambda*C2'*L2*C2;%C in the paper
        D=(D+D')/2;
        [v,eigval]=eig(D);
        eigval = diag(eigval);
        [eigval, idx] = sort(eigval);
        
        if isempty(settings.codelens)
            settings.codelens = size(gnd_tr1,1);
        end
        
        for k = 1:size(results,2)
            try 
            Y1 = v(:,idx(2:settings.codelens(k)+1));%F in the paper
            Y2 = C2*Y1;
            eyemat_m1 = eye(m1);
            eyemat_m2 = eye(m2);
            Wi = (X1*X1'+beta*eyemat_m1)\X1*Y1;
            Wt = (X2*X2'+beta*eyemat_m2)\X2*Y2;
            
            IMHi_tr1 = (ifea_tr1'*Wi>0);
            IMHt_tr1 = (tfea_tr1'*Wt>0);
            IMHi_te1 = (ifea_te1'*Wi>0);
            IMHt_te1 = (tfea_te1'*Wt>0);
            IMHi_tr2 = (ifea_tr2'*Wi>0);
            IMHt_tr2 = (tfea_tr2'*Wt>0);
            IMHi_te2 = (ifea_te2'*Wi>0);
            IMHt_te2 = (tfea_te2'*Wt>0);
            
            num_tr1 = size(ifea_tr1,2);
            num_tr2 = size(ifea_tr2,2);
            
            [MAPi2t1,~,resi2t1] = mAPEvaluate(IMHi_te1,IMHt_tr1,gnd_te1,gnd_tr1,num_tr1,'hamming');
            [MAPt2i1,~,rest2i1] = mAPEvaluate(IMHt_te1,IMHi_tr1,gnd_te1,gnd_tr1,num_tr1,'hamming');
            [MAPi2t2,~,resi2t2] = mAPEvaluate(IMHi_te2,IMHt_tr2,gnd_te2,gnd_tr2,num_tr2,'hamming');
            [MAPt2i2,~,rest2i2] = mAPEvaluate(IMHt_te2,IMHi_tr2,gnd_te2,gnd_tr2,num_tr2,'hamming');
            
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
        result.method = 'CVH';
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

rmpath('./imh');

end

