function [rec_train,rec_test,trainpred,testpred] = evalPerClassRecognitionRate(traind, trainl, testd, testl, fwdfun)

y = fwdfun(testd);
[~,my] = max(y,[],2);
[~,ml] = max(testl,[],2);
nClasses = length(unique(ml));
trainpred = my;

perClassError = 0;
for i=1:nClasses
    indx = find(ml==i);
    if isempty(indx)
        fprintf('test: class %d has no samples\n',i);
    else
        perClassError = perClassError + sum(my(indx) == ml(indx)) / length(indx);
    end
end
rec_test = perClassError/nClasses;
fprintf('test rate has been computed using %d classes (%.2f%%)\n',nClasses,rec_test*100)



y = fwdfun(traind);
[~,my] = max(y,[],2);
[~,ml] = max(trainl,[],2);
nClasses=length(unique(ml));
testpred = my;

perClassError = 0;
for i=1:nClasses
    indx = find(ml==i);
    if isempty(indx)
        fprintf('train: class %d has no samples\n',i);
    else
        perClassError = perClassError + sum(my(indx) == ml(indx)) / length(indx);
    end
end
rec_train = perClassError/nClasses;
fprintf('train rate has been computed using %d classes (%.2f%%)\n',nClasses,rec_train*100)

fprintf('Average per Class Recognition rate: test %.2f%%\t train %.2f%%\n',rec_test*100,rec_train*100);


