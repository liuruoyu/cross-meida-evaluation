% Semantics-Preserving Hashing for Cross-View Retrieval
clear;
addpath(genpath('markSchmidt/'));

%% Parameter Setting
globalBits = [16, 32, 64, 128]; 
datasets = {'wikiData', 'mirflickr25k'};                        
kernelSamps = [500, 500];                            % sampling size for kernel logistic regression

dtN = length(datasets);
recallLevelStep = 0.05;

fid = fopen('SePH.txt', 'w');
fprintf(fid, '\r\n\r\nStarted: %s', datestr(now,31));

% Model parameters
Model = {};
Model.alpha = 1e-2;

%% SePH
for di = 1 : dtN
    clearvars -except globalBits datasets dtN di recallLevelStep fid Model kernelSamps;
    load(['datasets/', datasets{di}, '.mat']);
        
    v = 2;
    viewsName = {'Image', 'Text'};
    
    RetrXs = cell(1, v);                            % Retrieval Set
    RetrXs{1} = I_tr;
    RetrXs{2} = T_tr;    
    queryXs = cell(1, v);                           % Query Set
    queryXs{1} = I_te;
    queryXs{2} = T_te;    
    clear I_tr T_tr I_te T_te;
    
    % Feature Pretreatment
    for i = 1 : v
        meanV = mean(RetrXs{i}, 1);
        RetrXs{i} = bsxfun(@minus, RetrXs{i}, meanV);
        queryXs{i} = bsxfun(@minus, queryXs{i}, meanV);
    end
    
    trainNum = length(sampleInds);                  % Training Set
    trainXs = cell(1, v);
    trainXs{1} = RetrXs{1}(sampleInds, :);
    trainXs{2} = RetrXs{2}(sampleInds, :);
    
    % Calculation of P for supervised learning (normalized cosine similarity)
    P = zeros(trainNum, trainNum);
    trainLabels = L_tr(sampleInds, :);
    if size(trainLabels, 2) > 1
        num1 = 1 ./ sqrt(sum(trainLabels .* trainLabels, 2));
        num1(isinf(num1) | isnan(num1)) = 1;
        trainLabels = diag(num1) * trainLabels;
        
        P = trainLabels * trainLabels';                             % cosine similarity for multi-label cases
    else    
        for ti = 1 : trainNum
            P(ti, :) = double(trainLabels == trainLabels(ti))';     % binary similarity (i.e. 0 and 1) for multi-class cases
        end
    end
    trainLabels = L_tr(sampleInds, :);                              % recover to the original labels 
    
    % Training & Testing
    bitN = length(globalBits);
    bits = globalBits;
    
    queryNum = size(L_te, 1);
    
    runtimes = 10;                                                  % 10 runs
    mAPs = zeros(bitN, v, runtimes, 2);
    trainMAPs = zeros(bitN, runtimes);
    
    for bi = 1 : bitN        
        bit = bits(bi);
        fprintf(fid, '\r%s Bit %d:\r', datasets{di}, bit);

        for ri = 1 : runtimes
            % Deriving Code Matrix of Training Set
            ydata = minKLD(Model.alpha/bit/trainNum, P, bit);
            trainH = sign(ydata);

            % Evaluating the Quality of Learnt Hash Codes for Training Set
            trEv = trainEval(trainLabels, trainH);
            fprintf('Runtime %d, Manifold Evaluation MAP [%.4f]\r', ri, trEv);
            fprintf(fid, 'Runtime %d, Manifold Evaluation MAP [%.4f]\r', ri, trEv);
            trainMAPs(bi, ri) = trEv;

            % RBF Kernel
            z = trainXs{1} * trainXs{1}';
            z = repmat(diag(z), 1, trainNum)  + repmat(diag(z)', trainNum, 1) - 2 * z;
            k1 = {};
            k1.type = 0;
            k1.param = mean(z(:));                                  %  $\sigma^2$ for RBF kernel in image view

            z = trainXs{2} * trainXs{2}';
            z = repmat(diag(z), 1, trainNum)  + repmat(diag(z)', trainNum, 1) - 2 * z;
            k2 = {};
            k2.type = 0;
            k2.param = mean(z(:));                                  %  $\sigma^2$ for RBF kernel in text view    

            % Kernel Logistic Regression (KLR)£¬Developed by Mark Schimidt     
            for si = 1 : 2
                kernelSampleNum = kernelSamps(di);
                if si == 1 && kernelSampleNum > trainNum
                    break;
                elseif si == 2 && kernelSampleNum > trainNum / 2
                    break;
                end

                sampleType = 'Random';
                if si == 1
                    % Random Sampling for Learning KLR
                    kernelSamples = sort(randperm(trainNum, kernelSampleNum));
                    kernelXs{1} = trainXs{1}(kernelSamples, :);
                    kernelXs{2} = trainXs{2}(kernelSamples, :);
                else
                    sampleType = 'Kmeans';
                    % Kmeans Sampling for Learning KLR
                    opts = statset('Display', 'off', 'MaxIter', 100);
                    [INX, C] = kmeans(trainXs{1}, kernelSampleNum, 'Start', 'sample', 'EmptyAction', 'singleton', 'Options', opts, 'OnlinePhase', 'off');
                    kernelXs{1} = C;

                    [INX, C] = kmeans(trainXs{2}, kernelSampleNum, 'Start', 'sample', 'EmptyAction', 'singleton', 'Options', opts, 'OnlinePhase', 'off');
                    kernelXs{2} = C;
                end

                % Kernel Matrices
                K01 = kernelMatrix(kernelXs{1}, kernelXs{1}, k1);
                K02 = kernelMatrix(kernelXs{2}, kernelXs{2}, k2);
                trainK1 = kernelMatrix(trainXs{1}, kernelXs{1}, k1);
                trainK2 = kernelMatrix(trainXs{2}, kernelXs{2}, k2);
                RetrK1 = kernelMatrix(RetrXs{1}, kernelXs{1}, k1);
                RetrK2 = kernelMatrix(RetrXs{2}, kernelXs{2}, k2);
                queryK1 = kernelMatrix(queryXs{1}, kernelXs{1}, k1);
                queryK2 = kernelMatrix(queryXs{2}, kernelXs{2}, k2);        

                % Hash Codes for Retrieval Set and Query Set
                B1 = zeros(size(L_tr, 1), bit);                             % Unique Hash Codes for Both Views of Retrieval Set
                B21 = zeros(queryNum, bit);                                 % Hash Codes for Image View of Query Set
                B22 = zeros(queryNum, bit);                                 % Hash Codes for Text View of Query Set

                options.Display = 'final';        
                C = 0.01;                                                   % Weight for Regularization. 1e-2 is Good Enough.
                
                % KLR for Each Bit
                for b = 1 : bit
                    tH = trainH(:, b);

                    % View 1 (Image View)
                    funObj = @(u)LogisticLoss(u, trainK1, tH);
                    w = minFunc(@penalizedKernelL2, zeros(size(K01, 1),1), options, K01, funObj, C);  
                    B21(:, b) = sign(queryK1 * w);
                    z11 = 1 ./ (1 + exp(-RetrK1 * w));                                     % P(pos | V_1)
                    z10 = 1 - z11;                                                         % P(neg | V_1)

                    % View 2 (Text View)
                    funObj = @(u)LogisticLoss(u, trainK2, tH);
                    w = minFunc(@penalizedKernelL2, zeros(size(K02, 1),1), options, K02, funObj, C); 
                    B22(:, b) = sign(queryK2 * w);
                    z21 = 1 ./ (1 + exp(-RetrK2 * w));                                     % P(pos | V_2)     
                    z20 = 1 - z21;                                                         % P(neg | V_2)

                    % Retrieval Set (Combining)
                    B1(:, b) = sign(z11 .* z21 - z10 .* z20);                               % sign(P(pos | V_1)P(pos | V_2) - P(neg | V_1)P(neg | V_2))
                end

                B1 = bitCompact(sign(B1) >= 0);
                B21 = bitCompact(sign(B21) >= 0);
                B22 = bitCompact(sign(B22) >= 0);

                % Evaluation
                vi = 1;
                hammingM = double(HammingDist(B21, B1))';
                [ mAPValue ] = perf_metric4Label( L_tr, L_te, hammingM );
                mAPs(bi, vi, ri, si) = mAPValue;
                fprintf('%s Bit %d Runtime %d Sampling Type [%s] Sampling Num [%d], %s query %s: MAP [%.6f]\r', ...,
                    datasets{di}, bit, ri, sampleType, kernelSampleNum, viewsName{1}, viewsName{2}, mAPValue);
                fprintf(fid, '%s Bit %d Runtime %d Sampling Type [%s] Sampling Num [%d], %s query %s: MAP [%.6f]\r', ...,
                    datasets{di}, bit, ri, sampleType, kernelSampleNum, viewsName{1}, viewsName{2}, mAPValue);

                vi = 2;
                hammingM = double(HammingDist(B22, B1))';                
                [ mAPValue ] = perf_metric4Label( L_tr, L_te, hammingM );
                mAPs(bi, vi, ri, si) = mAPValue;
                fprintf('%s Bit %d Runtime %d Sampling Type [%s] Sampling Num [%d], %s query %s: MAP [%.6f]\r', ...,
                    datasets{di}, bit, ri, sampleType, kernelSampleNum, viewsName{2}, viewsName{1}, mAPValue);
                fprintf(fid, '%s Bit %d Runtime %d Sampling Type [%s] Sampling Num [%d], %s query %s: MAP [%.6f]\r', ...,
                    datasets{di}, bit, ri, sampleType, kernelSampleNum, viewsName{2}, viewsName{1}, mAPValue);
            end
        end
    end
    
    % Save Results
    if ~exist('results', 'dir')
        mkdir('results');
    end
    save(['results/SePH_', datasets{di}, '.mat'], 'mAPs', 'bits', 'kernelSampleNum', 'Model', 'trainMAPs');    
end
fclose(fid);