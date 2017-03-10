function [maps] = crossMediaSearch( seeds, GT, X1_test, X1_binary, X1_lowD)
%%
% 
% beta is the parameter, seeds are the query seeds,
% GT is the groundtruth
% X1_tr and X2_tr are the training images, which are n*m
% X1_test is the testing images, n1*m1
% X2_test is the testing images, n2*m2
% 
% maps is the result for a setting gamma, alpha and beta, maps is steps*24


%% Firstly we get different W and b
% And then we generate the hash code of the testing dataset
% Lastly, we use seeds to query these testing dataset
% 
% 
% 
%%
% steps=[5:5:60 80 100 150 200 400];
codeLen=8:16:144;
steps_num=length(codeLen);
% maps=zeros(steps_num,24);
maps=zeros(2,10);
% for len=3:10
    % Get Y, W, b
%     [feaDimX1, trainNum1] = size(X1_tr);
%     [feaDimX2, trainNum2] = size(X2_tr);
%     
%     eyemat_m1 = eye(feaDimX1);
%     eyemat_m2 = eye(feaDimX2);
%     W1 = (X1_tr*X1_tr'+beta*eyemat_m1)\X1_tr*Y1;
%     W2 = (X2_tr*X2_tr'+beta*eyemat_m2)\X2_tr*Y2;
% 
%     % Start to query
     [feaDimX, vid_num] = size(X1_test');
% %     oneline = ones(vid_num,1);
%     X1_lowD = X1_test*W1;   %low dimensity kf
%     clear W1 W2;
%     X1_med = median(X1_lowD);
%     X1_binary=(X1_lowD>repmat(X1_med,vid_num,1));
%     save binaryCodeMHash_all binaryCode;
%     First test the binary code
%     Then test the low dimensional data
    map = binarySearch(seeds, X1_binary, GT);
    maps(1, :) = map;
    for query_id=1:10
        seed_num = seeds(query_id);
        kf_query = X1_lowD(seed_num,:);
%       Start to search
        result = sum((X1_lowD-repmat(kf_query,vid_num,1)).*(X1_lowD-repmat(kf_query,vid_num,1)),2);
        result=0-result; % Here the result is the similarity
        map= evaluationMAP(GT(:,query_id), result);
        maps(2,query_id) = map;
    end







