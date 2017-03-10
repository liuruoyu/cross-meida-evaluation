bmark_path = 'E:\Benchmarks\nus_wide';
bmark_path_ = 'E:\Crossmedia(deep)\Benchmarks\nus_wide';
path_prefix = '/home/ruoyu/Benchmarks/nus_wide/images';
path_prefix_ = '/home/ruoyu/Crossmedia(deep)/Benchmarks/nus_wide';

img_files = dir(fullfile(bmark_path,'images','train','*.jpg'));
img_files = {img_files.name};
load(fullfile(bmark_path,'txt_word2vec.mat'),'tfea_tr');
load(fullfile(bmark_path,'gnd.mat'),'gnd_tr');
load(fullfile(bmark_path,'cat.mat'),'catIdx1');

for i = 1:length(catIdx1)
    cat_idx1 = catIdx1{i};
    tr_idx1 = find(sum(gnd_tr(cat_idx1,:),1)>0);
    img_files1 = img_files(tr_idx1);
    gnd_tr1 = gnd_tr(cat_idx1,tr_idx1);
    gnd_tr1 = bsxfun(@rdivide, gnd_tr1, sum(gnd_tr1,1));
    tfea_tr1 = tfea_tr(:,tr_idx1);
    num_tr1 = size(gnd_tr1,2);
    
    iter_epoch = floor(num_tr1/256);
    total_epoch = 60;
    save_iter = 10;
    
    bmark_path_1 = fullfile(bmark_path_,num2str(i));
    if ~exist(bmark_path_1,'dir')
        mkdir(bmark_path_1);
    end
    path_prefix_1 = sprintf('%s/%d',path_prefix_,i);
    
    % training
    ilabelv = [];
    tdata = [];
    tlabelv = [];
    fid = fopen(fullfile(bmark_path_1,'img_list_tr.txt'),'w');
    for j = 1:total_epoch
        tmp_idx = randperm(num_tr1,iter_epoch*256);
        ilabelv = [ilabelv,gnd_tr1(:,tmp_idx)];
        tlabelv = [tlabelv,gnd_tr1(:,tmp_idx)];
        tdata = [tdata, tfea_tr1(:,tmp_idx)];
        for k = 1:length(tmp_idx)
            fprintf(fid,'%s/train/%s %d\n',path_prefix,img_files1{tmp_idx(k)},0);
        end
        if mod(j,save_iter) == 0
            if exist(fullfile(bmark_path_1,sprintf('txt_label_tr_%02d.h5',j/save_iter)), 'file')
                delete(fullfile(bmark_path_1,sprintf('txt_label_tr_%02d.h5',j/save_iter)));
            end
            h5create(fullfile(bmark_path_1,sprintf('txt_label_tr_%02d.h5',j/save_iter)),'/tlabelv',size(tlabelv),'Datatype','single');
            h5create(fullfile(bmark_path_1,sprintf('txt_label_tr_%02d.h5',j/save_iter)),'/tdata',size(tdata),'Datatype','single');
            h5write(fullfile(bmark_path_1,sprintf('txt_label_tr_%02d.h5',j/save_iter)),'/tlabelv',single(tlabelv));
            h5write(fullfile(bmark_path_1,sprintf('txt_label_tr_%02d.h5',j/save_iter)),'/tdata',single(tdata));
            tlabelv = [];
            tdata = [];
        end
    end
    fclose(fid);
    
    if exist(fullfile(bmark_path_1,'img_label_tr.h5'), 'file')
        delete(fullfile(bmark_path_1,'img_label_tr.h5'));
    end
    h5create(fullfile(bmark_path_1,'img_label_tr.h5'),'/ilabelv',size(ilabelv),'Datatype','single');
    h5write(fullfile(bmark_path_1,'img_label_tr.h5'),'/ilabelv',single(ilabelv));
    fid = fopen(fullfile(bmark_path_1,'img_h5_list_tr.txt'),'w');
    fprintf(fid,'%s/img_label_tr.h5\n', path_prefix_1);
    fclose(fid);
    
    fid = fopen(fullfile(bmark_path_1,'txt_h5_list_tr.txt'),'w');
    for j = 1:total_epoch/save_iter
        fprintf(fid,'%s/txt_label_tr_%02d.h5\n', path_prefix_1,j);
    end
    fclose(fid);
    
    % testing
    ilabelv = [];
    tlabelv = [];
    tdata = [];
    fid = fopen(fullfile(bmark_path_1,'img_list_te.txt'),'w');
    for j = 1:total_epoch
        tmp_idx = randperm(num_tr1,iter_epoch*64);
        ilabelv = [ilabelv,gnd_tr1(:,tmp_idx)];
        tlabelv = [tlabelv,gnd_tr1(:,tmp_idx)];
        tdata = [tdata, tfea_tr1(:,tmp_idx)];
        for k = 1:length(tmp_idx)
            fprintf(fid,'%s/train/%s %d\n',path_prefix,img_files1{tmp_idx(k)},0);
        end
        if mod(j,save_iter) == 0
            if exist(fullfile(bmark_path_1,sprintf('txt_label_te_%02d.h5',j/save_iter)), 'file')
                delete(fullfile(bmark_path_1,sprintf('txt_label_te_%02d.h5',j/save_iter)));
            end
            h5create(fullfile(bmark_path_1,sprintf('txt_label_te_%02d.h5',j/save_iter)),'/tlabelv',size(tlabelv),'Datatype','single');
            h5create(fullfile(bmark_path_1,sprintf('txt_label_te_%02d.h5',j/save_iter)),'/tdata',size(tdata),'Datatype','single');
            h5write(fullfile(bmark_path_1,sprintf('txt_label_te_%02d.h5',j/save_iter)),'/tlabelv',single(tlabelv));
            h5write(fullfile(bmark_path_1,sprintf('txt_label_te_%02d.h5',j/save_iter)),'/tdata',single(tdata));
            tlabelv = [];
            tdata = [];
        end
    end
    fclose(fid);
    
    if exist(fullfile(bmark_path_1,'img_label_te.h5'), 'file')
        delete(fullfile(bmark_path_1,'img_label_te.h5'));
    end
    h5create(fullfile(bmark_path_1,'img_label_te.h5'),'/ilabelv',size(ilabelv),'Datatype','single');
    h5write(fullfile(bmark_path_1,'img_label_te.h5'),'/ilabelv',single(ilabelv));
    fid = fopen(fullfile(bmark_path_1,'img_h5_list_te.txt'),'w');
    fprintf(fid,'%s/img_label_te.h5\n', path_prefix_1);
    fclose(fid);
    
    fid = fopen(fullfile(bmark_path_1,'txt_h5_list_te.txt'),'w');
    for j = 1:total_epoch/save_iter
        fprintf(fid,'%s/txt_label_te_%02d.h5\n', path_prefix_1,j);
    end
    fclose(fid);
end