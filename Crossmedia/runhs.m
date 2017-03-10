addpath('./binary_methods');
addpath('./utils');

settings.bmark_path = '../Benchmarks';
% bmarks: wiki, pascal_sentence, nus_wide
settings.bmarks = {'wiki','pascal_sentence','nus_wide'};
% codelens: 8,16,32
settings.codelens = [8,16,32];
% methods: CVH, SCMseq, SCMorth, CMFH, LSSH, SEPHkm, IMH, MMNN
settings.iterN = 5;
test_methods = {'CVH','SEPHseq','SCMorth','CMFH','LSSH','SEPHkm','IMH','MMNN'};
results = struct;

if ismember('CVH',test_methods)
    results.CVH = CVH(settings);
end

if ismember('SCMseq',test_methods)
    results.SCMseq = SCMseq(settings);
end

if ismember('SCMorth',test_methods)
    results.SCMorth = SCMorth(settings);
end

if ismember('CMFH',test_methods)
    results.CMFH = CMFH(settings);
end

if ismember('LSSH',test_methods)
    results.LSSH = LSSH(settings);
end

if ismember('SEPHkm',test_methods)
    results.SEPHkm = SEPHkm(settings);
end

if ismember('IMH',test_methods)
    results.IMH = IMH(settings);
end

if ismember('MMNN',test_methods)
    results.MMNN = MMNN_m(settings);
end

rmpath('./binary_methods');
rmpath('./utils');

save('result_hs.mat','results');