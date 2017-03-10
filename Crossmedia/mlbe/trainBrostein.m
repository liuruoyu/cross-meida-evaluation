function [PROJX,PROJY,variance,history] = trainBrostein(Xp_train,Yp_train,Xn_train,Yn_train, K, M, alpha, loss)

addpath('./brostein_mmssh');
% Initialize
np = size(Xp_train,2);  % # of positive training pairs
nn = size(Xn_train,2);  % # of negative training pairs
dx = size(Xp_train,1);  % Data dimensionality
dy = size(Yp_train,1);  % Data dimensionality

use_testset   = false;
show_plot     = false;

wp = ones(np,1); wp = wp/sum(wp);
wn = ones(nn,1); wn = wn/sum(wn);

if nargin < 5,
    K = [10 50];              % Subspace size + number of directions to search
end
if nargin < 6,
    M = 64;             % Code size
end
if nargin < 7,
    alpha = .1;         % Reweighting speed
end
if nargin < 8,
    loss = 'quad';
end

history = [];
history.train_error = [];
history.test_error = [];
history.loss = [];

PROJX = zeros(M, dx+1);
PROJY = zeros(M, dy+1);
variance = zeros(M,1);

mprintf('', 'Training %d-bit code   dimension: %d,%d\n', M, dx,dy);

dp_train  = zeros(1, size(Xp_train,2));
dn_train  = zeros(1, size(Xn_train,2));

if use_testset,
    dp_test  = zeros(1, size(Xp_test,2));
    dn_test  = zeros(1, size(Xn_test,2));
end

sum_alpha = 0;
for m = 1:M,

    tic;
    % Find best projection
    [projx,projy, xp,yp,xn,yn, alpha, curloss] = adaboost_iter_cross(Xp_train,Yp_train,Xn_train,Yn_train,wp,wn,K(1),K(2), alpha, loss);

    % Add to projection matrices
    projx = projx(:)';
    projy = projy(:)';
    PROJX(m,:) = projx;
    PROJY(m,:) = projy;
    
%     % Compute variance
%     variance(m) = 0.5*(mean(xn.^2) + mean(yn.^2));
%     
%     % Training rates
%     str = mprintf('', '   Evaluating training error...'); 
%     xp = sign(xp);
%     yp = sign(yp);
%     xn = sign(xn);
%     yn = sign(yn);
%     sum_alpha = sum_alpha + alpha;
%     dp_train  = dp_train + alpha*double(xp ~= yp);
%     dn_train  = dn_train + alpha*double(xn ~= yn);
%     [eer_train,fpr1_train,fpr01_train, dee_train,dfr1_train,dfr01_train, dist_train,fp_train,fn_train] = ...
%         calculate_rates(dp_train/sum_alpha, dn_train/sum_alpha, []);
%     history.train_error(end+1,:) = [eer_train,fpr1_train,fpr01_train];
%     history.loss(end+1) = curloss;

%     % Test rates
%     if use_testset,
%         str = mprintf(str, '   Evaluating test error...'); 
%         xp_test = sign(projx(1:end-1)*Xp_test + projx(end));
%         yp_test = sign(projy(1:end-1)*Yp_test + projy(end));
%         xn_test = sign(projx(1:end-1)*Xn_test + projx(end));
%         yn_test = sign(projy(1:end-1)*Yn_test + projy(end));
%         dp_test = dp_test + alpha*double(xp_test ~= yp_test);
%         dn_test = dn_test + alpha*double(xn_test ~= yn_test);   
%         [eer_test, fpr1_test, fpr01_test,  dee_test, dfr1_test, dfr01_test,  dist_test, fp_test, fn_test]  = ...
%             calculate_rates(dp_test/sum_alpha, dn_test/sum_alpha, []);
%         history.test_error(end+1,:) = [eer_test, fpr1_test, fpr01_test];
%     end    

%     % Output rates
%     str = mprintf(str, '   %-3d loss %6.2f  variance %6.2f  elapsed time %s\n', m, curloss, variance(m), format_time(toc));
%     mprintf('', '         Train:  eer %6.2f%%   fp@fn=1%% %6.2f%%   fp@fn=0.1%% %6.2f%%\n', 100*eer_train, 100*fpr1_train, 100*fpr01_train);
%     if use_testset,
%         mprintf('', '         Test:   eer %6.2f%%   fp@fn=1%% %6.2f%%   fp@fn=0.1%% %6.2f%%\n', 100*eer_test, 100*fpr1_test, 100*fpr01_test);
%     end
    
%     % Show plot
%     if show_plot,
%         if use_testset,
%              plot(dist_train, fp_train, 'r',  dist_train, fn_train, 'b', ...
%                   dist_test,  fp_test,  ':r', dist_test,  fn_test,  ':b');
%              legend('FP (Train)', 'FN (Train)', 'FP (Test)', 'FN (Test)');
%         else
%              plot(dist_train, fp_train, 'r', dist_train, fn_train, 'b');
%              legend('FP (Train)', 'FN (Train)');
%         end
%         axis([0 1 0 1]);
%         xlabel('Normalized distance');
%         drawnow;
%     end

    % Reweight examples: 
    % - correctly classified ones are downweighted
    % - incorrectly classified ones are boosted
    %fprintf(1, '  Reweighting examples with alpha=%.4f\n', alpha); 
    dwp = exp(-alpha*(xp.*yp));
    dwn = exp(+alpha*(xn.*yn)); 
    wp  = wp(:) .* dwp(:);
    wn  = wn(:) .* dwn(:);
    wp  = wp / sum(wp);
    wn  = wn / sum(wn);
    
end
% End greedy adaboost iterations



