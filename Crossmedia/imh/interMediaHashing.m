function [] = crossMedia(X1, X2, X1_test, X2_test, n2)
%% 
% X1 is the image dataset, (n1+n2)*m1
% X2 is the text dataset, (n2+n3)*m2
% 
%%
tic;

fid =fopen(['result\maps.txt'],'w');


X1 = X1';
X2 = X2';
%% Steps used to get L1 and L2
para.k=10;
para.sigma=1;
L1=Laplacian_GK(X1,para);
L2=Laplacian_GK(X2,para);
%%
% n2 = 500;
% views{1}=hsv_tr;views{2}=lbp_tr;

% X is the training data
% X = [views{1};views{2}];
% viewsTest is the testing data
% viewsTest = [hsvNor_big lbpNor_big]';
% 
% [trainF,trainNum ] = size(views{1});

% oneline = ones(trainNum,1);
% max_map=0;
% training
    for beta=-6:3:6
        for lambda=-6:3:6
            %% Steps to get the Y1, Y2 ( First get the W1, W2, and then get the binary codes for testing dataset)
            [X1_low, X1_binary, X2_low, X2_binary] = crossMediaTraining(X1, X2, n2, beta, lambda, L1, L2, X1_test, X2_test);
%                 tic;[v, eigval, Y1, Y2] = crossMediaHashing(X1, X2, n2, beta, lambda, L1, L2);
%                  filev=['result\Y1_low',num2str(n2),num2str(beta+6),num2str(lambda+6)];
%                  filee=['result\Y2_low',num2str(n2),num2str(beta+6),num2str(lambda+6)];
%                  save (filev,'Y1','-ascii');
%                  save (filee,'Y2','-ascii');
% 
%                 clear v eigval;
            %% Steps to seach after getting Y1 and Y2
        %         Start to query
                load data\wikiGT.mat;
                load data\wikiseeds.mat;
%                 Y1 = load(filev);
%                 Y2 = load(filee);
%                 
                 [maps] = crossMediaSearch( seeds, GT, X1_test, X1_binary, X1_low)
%                 fprintf(fid,'beta=%d lambda=%d\n',beta,lambda);
%                 len=size(maps,2);
%                 fprintf(fid, '%d ', mean(maps,2));
%                 fprintf(fid, '\n');
%                 for line=1:len
%                     fprintf(fid,'%d ',maps(1,line));
%                     fprintf(fid,' ');
%                 end
%                 fprintf(fid, '\n');
%                 for line=1:len
%                     fprintf(fid,'%d ',maps(2,line));
%                     fprintf(fid,' ');
%                 end
%                 fprintf(fid,'\n');                
                toc
        end
    end


% fclose(fid);
toc;