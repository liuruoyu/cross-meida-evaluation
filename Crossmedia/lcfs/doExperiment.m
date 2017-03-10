%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% doExperiment: Learning Coupled Feature Spaces for Cross-modal Matching
% Kaiye Wang
% NLPR, CASIA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function doExperiment
    clear;
    clc;
    
    %% load data
    load('.\voc data\train');
    load('.\voc data\test');
    X_a = TrImage;
    X_b = TrText;
    Y = trY;
    
    Test_a = TeImage;
    Test_b = TeText;
    test_Y = teY; 
    
    %% train 
    lambda_1 = 0.1;
    lambda_2 = 0.001;
    ite = 5;
    
    [ W_a, W_b ] = LCFS_ite( X_a, X_b, Y, lambda_1, lambda_2, ite);
    
    %% test
    [ map1, map2 ] = Test_LCFS(Test_a, Test_b, test_Y, W_a, W_b);
   
    disp('End!');
end

