addpath('./real_methods');
addpath('./utils');

settings.bmark_path = '../Benchmarks';
% bmarks: wiki, pascal_sentence, nus_wide
settings.bmarks = {'wiki','pascal_sentence','nus_wide'};
% methods: CM, SM, SCM, PLS, BLM, GMMFA, GMLDA, CCA3V, LCFS, deep_SM
settings.iterN = 5;
test_methods = {'CM','SM','SCM','PLS','BLM','GMMFA','GMLDA','CCA3V','LCFS','deep_SM'};
results = struct;

if ismember('CM',test_methods)
    results.CM = CM(settings);
end

if ismember('SM',test_methods)
    results.SM = SM(settings);
end

if ismember('SCM',test_methods)
    results.SCM = SCM(settings);
end

if ismember('PLS',test_methods)
    results.PLS = PLS(settings);
end

if ismember('BLM',test_methods)
    results.BLM = BLM(settings);
end

if ismember('GMMFA',test_methods)
    results.GMMFA = GMMFA(settings);
end

if ismember('GMLDA',test_methods)
    results.GMLDA = GMLDA(settings);
end

if ismember('CCA3V',test_methods)
    results.CCA3V = CCA3V(settings);
end

if ismember('LCFS',test_methods)
    results.LCFS = LCFS(settings);
end

if ismember('deep_SM',test_methods)
    results.deep_SM = deep_SM(settings);
end

rmpath('./real_methods');
rmpath('./utils');

save('result_r.mat','results');
