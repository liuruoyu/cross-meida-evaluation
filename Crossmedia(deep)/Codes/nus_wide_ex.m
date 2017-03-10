bmark_path = 'E:\Benchmarks\nus_wide';
bmark_path_ = 'E:\Crossmedia(deep)\Benchmarks\nus_wide';
path_prefix = '/home/ruoyu/Benchmarks/nus_wide/images';
path_prefix_ = '/home/ruoyu/Crossmedia(deep)/Benchmarks/nus_wide';

img_files_tr = dir(fullfile(bmark_path,'images','train','*.jpg'));
img_files_tr = {img_files_tr.name};
img_files_te = dir(fullfile(bmark_path,'images','test','*.jpg'));
img_files_te = {img_files_te.name};

load(fullfile(bmark_path,'gnd.mat'),'gnd_tr','gnd_te');
load(fullfile(bmark_path,'txt_word2vec.mat'),'tfea_tr','tfea_te');
num_tr = size(gnd_tr,2);
num_te = size(gnd_te,2);
num_cat = 0.5*size(gnd_tr,1);
if ~exist(fullfile(bmark_path_,'ex'),'dir')
    mkdir(fullfile(bmark_path_,'ex'));
end
fid = fopen(fullfile(bmark_path_,'ex','img_list_ex.txt'),'w');
for i = 1:num_tr
    fprintf(fid,'%s/train/%s %d\n', path_prefix,img_files_tr{i},0);
end
for i = 1:num_te
    fprintf(fid,'%s/test/%s %d\n', path_prefix,img_files_te{i},0);
end
fclose(fid);

if exist(fullfile(bmark_path_,'ex','img_label_ex.h5'), 'file')
    delete(fullfile(bmark_path_,'ex','img_label_ex.h5'));
end
h5create(fullfile(bmark_path_,'ex','img_label_ex.h5'),'/ilabelv',[num_cat,num_tr+num_te],'Datatype','single');
h5write(fullfile(bmark_path_,'ex','img_label_ex.h5'),'/ilabelv',single(zeros([num_cat,num_tr+num_te])));
fid = fopen(fullfile(bmark_path_,'ex','img_h5_list_ex.txt'),'w');
fprintf(fid,'%s/ex/img_label_ex.h5\n', path_prefix_);
fclose(fid);

if exist(fullfile(bmark_path_,'ex','txt_label_ex.h5'), 'file')
    delete(fullfile(bmark_path_,'ex','txt_label_ex.h5'));
end
h5create(fullfile(bmark_path_,'ex','txt_label_ex.h5'),'/tlabelv',[num_cat,num_tr+num_te],'Datatype','single');
h5create(fullfile(bmark_path_,'ex','txt_label_ex.h5'),'/tdata',size([tfea_tr,tfea_te]),'Datatype','single');
h5create(fullfile(bmark_path_,'ex','txt_label_ex.h5'),'/tlabel',size(zeros(1,num_tr+num_te)),'Datatype','single');
h5write(fullfile(bmark_path_,'ex','txt_label_ex.h5'),'/tlabelv',single(zeros([num_cat,num_tr+num_te])));
h5write(fullfile(bmark_path_,'ex','txt_label_ex.h5'),'/tdata',single([tfea_tr,tfea_te]));
h5write(fullfile(bmark_path_,'ex','txt_label_ex.h5'),'/tlabel',single(zeros(1,num_tr+num_te)));
fid = fopen(fullfile(bmark_path_,'ex','txt_h5_list_ex.txt'),'w');
fprintf(fid,'%s/ex/txt_label_ex.h5\n', path_prefix_);
fclose(fid);