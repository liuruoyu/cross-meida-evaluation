function [ U1, U2, P1, P2, Y ] = solveCMFH( X1, X2, lambda, mu, gamma, bits )
%SOLVECMFH Summary of this function goes here
% Collective Matrix Factorization Hashing Algorithm
%   minimize_{U1, U2, P1, P2, Y}    lambda*||X1 - U1 * Y||^2 + 
%      (1 - lambda)||X2 - U2 * Y||^2 + 
%      mu * (||Y - P1 * X1||^2 + ||Y - P2 * X2||^2) +
%      gamma * (||U1||^2 + ||U2||^2 + ||P1||^2 + ||P2||^2 + ||Y||^2)
% Notation:
% X1: data matrix of View1, each column is a sample vector
% X2: data matrix of View2, each column is a sample vector
% lambda: trade off between different views
% mu: trade off between collective matrix factorization and linean
% projection
% gamma: parameter to control the model complexity
% 
% Reference:
% GG Ding, Yuchen Guo, Jile Zhou
% "Collective Matrix Factorization Hashing for Multimodal Data"
% IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
% (Manuscript)
%
% Version1.0 -- Oct/2013
% Written by Yuchen Guo (yuchen.w.guo@gmail.com)
%
%

%% random initialization
[row, col] = size(X1);
[rowt, colt] = size(X2);
Y = rand(bits, col);
U1 = rand(row, bits);
U2 = rand(row, bits);
P1 = rand(bits, row);
P2 = rand(bits, rowt);
threshold = 0.01;
lastF = 99999999;
iter = 1;

%% compute iteratively
while (true)
		% update U1 and U2
    U1 = X1 * Y' / (Y * Y' + gamma * eye(bits));
    U2 = X2 * Y' / (Y * Y' + gamma * eye(bits));
    
		% update Y    
    Y = (lambda * U1' * U1 + (1- lambda) * U2' * U2 + 2 * mu * eye(bits) + gamma * eye(bits)) \ (lambda * U1' * X1 + (1 - lambda) * U2' * X2 + mu * (P1 * X1 + P2 * X2));
    
    %update W1 and W2
    P1 = Y * X1' / (X1 * X1' + gamma * eye(row));
    P2 = Y * X2' / (X2 * X2' + gamma * eye(rowt));
    
    % compute objective function
    norm1 = lambda * norm(X1 - U1 * Y, 'fro');
    norm2 = (1 - lambda) * norm(X2 - U2 * Y, 'fro');
    norm3 = mu * norm(Y - P1 * X1, 'fro');
    norm4 = mu * norm(Y - P2 * X2, 'fro');
    norm5 = gamma * (norm(U1, 'fro') + norm(U2, 'fro') + norm(Y, 'fro') + norm(P1, 'fro') + norm(P2, 'fro'));
    currentF= norm1 + norm2 + norm3 + norm4 + norm5;
    fprintf('\nobj at iteration %d: %.4f\n reconstruction error for collective matrix factorization: %.4f,\n reconstruction error for linear projection: %.4f,\n regularization term: %.4f\n\n', iter, currentF, norm1 + norm2, norm3 + norm4, norm5);
    if (lastF - currentF) < threshold
        fprintf('algorithm converges...\n');
        fprintf('final obj: %.4f\n reconstruction error for collective matrix factorization: %.4f,\n reconstruction error for linear projection: %.4f,\n regularization term: %.4f\n\n', currentF,norm1 + norm2, norm3 + norm4, norm5);
        return;
    end
    iter = iter + 1;
    lastF = currentF;
end
return;
end

