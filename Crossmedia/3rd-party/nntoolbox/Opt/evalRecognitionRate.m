function [rec_train,rec_test,trainpred,testpred] = evalRecognitionRate(traind, trainl, testd, testl, fwdfun)
T0 = cputime;

y = fwdfun(testd);
[~,my] = max(y,[],2);
[~,ml] = max(testl,[],2);
testpred = my;
rec_test = sum(my == ml) / length(ml);
fprintf('test recognition rate: %.2f%%, elapsed time %fs\n',rec_test*100,cputime-T0)

T1 = cputime;
y = fwdfun(traind);
[~,my] = max(y,[],2);
[~,ml] = max(trainl,[],2);
trainpred = my;
rec_train = sum(my == ml) / length(ml);
fprintf('train recognition rate: %.2f%%, elapsed time %fs\n',rec_train*100,cputime-T1)

fprintf('total elapsed time %f\n', cputime - T0)

