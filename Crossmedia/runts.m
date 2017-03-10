addpath('./class_methods');
addpath('./utils');

settings.bmark_path = '../Benchmarks';
% bmarks: wiki, pascal_sentence, nus_wide
settings.bmarks = {'wiki','pascal_sentence','nus_wide'};
% methods: Class, deep_Class
settings.iterN = 5;
test_methods = {'Class','deep_Class'};
results = struct;

if ismember('Class',test_methods)
    results.Class = Class(settings);
end

if ismember('deep_Class',test_methods)
    results.deep_Class = deep_Class(settings);
end

rmpath('./class_methods');
rmpath('./utils');

save('result_ts.mat','results');