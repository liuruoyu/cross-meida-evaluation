bmark_path = 'E:\Benchmarks\wiki';
bmark_path_ = 'E:\Crossmedia(deep)\Benchmarks\wiki';
path_prefix = '/home/ruoyu/Benchmarks/wiki/images';
path_prefix_ = '/home/ruoyu/Crossmedia(deep)/Benchmarks/wiki';

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
    tfea_tr1 = tfea_tr(:,tr_idx1);
    labels1 = zeros(size(gnd_tr1,2),1);
    for j = 1:size(gnd_tr1,2)
        labels1(j) = double(find(gnd_tr1(:,j))) - 1;
    end
    num_tr1 = size(gnd_tr1,2);
    
    iter_epoch = floor(num_tr1/256);
    total_epoch = 60;
    
    bmark_path_1 = fullfile(bmark_path_,num2str(i));
    if ~exist(bmark_path_1,'dir')
        mkdir(bmark_path_1);
    end
    path_prefix_1 = sprintf('%s/%d',path_prefix_,i);
    
    % training
    labelv = [];
    tdata = [];
    tlabel = [];
    fid = fopen(fullfile(bmark_path_1,'img_list_tr.txt'),'w');
    for j = 1:total_epoch
        tmp_idx = randperm(num_tr1,iter_epoch*256);
        labelv = [labelv,gnd_tr1(:,tmp_idx)];
        tdata = [tdata, tfea_tr1(:,tmp_idx)];
        tlabel = [tlabel, labels1(tmp_idx)'];
        for k = 1:length(tmp_idx)
            fprintf(fid,'%s/train/%s %d\n',path_prefix,img_files1{tmp_idx(k)},labels1(tmp_idx(k)));
        end
    end
    fclose(fid);
    
    if exist(fullfile(bmark_path_1,'img_label_tr.h5'), 'file')
        delete(fullfile(bmark_path_1,'img_label_tr.h5'));
    end
    h5create(fullfile(bmark_path_1,'img_label_tr.h5'),'/ilabelv',size(labelv),'Datatype','single');
    h5write(fullfile(bmark_path_1,'img_label_tr.h5'),'/ilabelv',single(labelv));
    fid = fopen(fullfile(bmark_path_1,'img_h5_list_tr.txt'),'w');
    fprintf(fid,'%s/img_label_tr.h5\n', path_prefix_1);
    fclose(fid);
    
    if exist(fullfile(bmark_path_1,'txt_label_tr.h5'), 'file')
        delete(fullfile(bmark_path_1,'txt_label_tr.h5'));
    end
    h5create(fullfile(bmark_path_1,'txt_label_tr.h5'),'/tlabelv',size(labelv),'Datatype','single');
    h5create(fullfile(bmark_path_1,'txt_label_tr.h5'),'/tdata',size(tdata),'Datatype','single');
    h5create(fullfile(bmark_path_1,'txt_label_tr.h5'),'/tlabel',size(tlabel),'Datatype','single');
    h5write(fullfile(bmark_path_1,'txt_label_tr.h5'),'/tlabelv',single(labelv));
    h5write(fullfile(bmark_path_1,'txt_label_tr.h5'),'/tdata',single(tdata));
    h5write(fullfile(bmark_path_1,'txt_label_tr.h5'),'/tlabel',single(tlabel));
    fid = fopen(fullfile(bmark_path_1,'txt_h5_list_tr.txt'),'w');
    fprintf(fid,'%s/txt_label_tr.h5\n', path_prefix_1);
    fclose(fid);
    
    % testing
    labelv = [];
    tdata = [];
    tlabel = [];
    fid = fopen(fullfile(bmark_path_1,'img_list_te.txt'),'w');
    for j = 1:total_epoch
        tmp_idx = randperm(num_tr1,iter_epoch*64);
        labelv = [labelv,gnd_tr1(:,tmp_idx)];
        tdata = [tdata, tfea_tr1(:,tmp_idx)];
        tlabel = [tlabel, labels1(tmp_idx)'];
        for k = 1:length(tmp_idx)
            fprintf(fid,'%s/train/%s %d\n',path_prefix,img_files1{tmp_idx(k)},labels1(tmp_idx(k)));
        end
    end
    fclose(fid);
    
    if exist(fullfile(bmark_path_1,'img_label_te.h5'), 'file')
        delete(fullfile(bmark_path_1,'img_label_te.h5'));
    end
    h5create(fullfile(bmark_path_1,'img_label_te.h5'),'/ilabelv',size(labelv),'Datatype','single');
    h5write(fullfile(bmark_path_1,'img_label_te.h5'),'/ilabelv',single(labelv));
    fid = fopen(fullfile(bmark_path_1,'img_h5_list_te.txt'),'w');
    fprintf(fid,'%s/img_label_te.h5\n', path_prefix_1);
    fclose(fid);
    if exist(fullfile(bmark_path_1,'txt_label_te.h5'), 'file')
        delete(fullfile(bmark_path_1,'txt_label_te.h5'));
    end
    h5create(fullfile(bmark_path_1,'txt_label_te.h5'),'/tlabelv',size(labelv),'Datatype','single');
    h5create(fullfile(bmark_path_1,'txt_label_te.h5'),'/tdata',size(tdata),'Datatype','single');
    h5create(fullfile(bmark_path_1,'txt_label_te.h5'),'/tlabel',size(tlabel),'Datatype','single');
    h5write(fullfile(bmark_path_1,'txt_label_te.h5'),'/tlabelv',single(labelv));
    h5write(fullfile(bmark_path_1,'txt_label_te.h5'),'/tdata',single(tdata));
    h5write(fullfile(bmark_path_1,'txt_label_te.h5'),'/tlabel',single(tlabel));
    fid = fopen(fullfile(bmark_path_1,'txt_h5_list_te.txt'),'w');
    fprintf(fid,'%s/txt_label_te.h5\n', path_prefix_1);
    fclose(fid);
end