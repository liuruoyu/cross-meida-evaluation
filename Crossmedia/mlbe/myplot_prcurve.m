function myplot_prcurve(premat, recmat, retmat)

mr = zeros(1,size(recmat.XY,2));
mp = zeros(1,size(recmat.XY,2));

for i = 1:size(recmat.XY,2)
    idx = find(retmat.XY(:,i)>0);
    mr(i) = mean(recmat.XY(idx,i));
    mp(i) = mean(premat.XY(idx,i));
end

figure;plot(mr, mp, 'o-', 'LineWidth',2);set(gca, 'FontSize',20);
title('Image Query vs. Text Database');h = get(gca, 'title');set(h, 'FontSize', 20);
xlabel('Recall');h = get(gca, 'xlabel');set(h, 'FontSize', 20);
ylabel('Precision');h = get(gca, 'ylabel');set(h, 'FontSize', 20);

mr = zeros(1,size(recmat.YX,2));
mp = zeros(1,size(recmat.YX,2));
for i = 1:size(recmat.YX,2)
    idx = find(retmat.YX(:,i)>0);
    mr(i) = mean(recmat.YX(idx,i));
    mp(i) = mean(premat.YX(idx,i));
end
figure;plot(mr, mp, 'o-', 'LineWidth',2);set(gca, 'FontSize',20);
title('Text Query vs. Image Database');h = get(gca, 'title');set(h, 'FontSize', 20);
xlabel('Recall');h = get(gca, 'xlabel');set(h, 'FontSize', 20);
ylabel('Precision');h = get(gca, 'ylabel');set(h, 'FontSize', 20);

% figure;hold on;
% plot(mean(retmat.XY,1), mean(recmat.XY,1), 'bo-', 'LineWidth',2);
% % for i = 1:size(retmat.XY,1)
% %     idx = find(retmat.XY(i,:)>0);
% %     plot(retmat.XY(i,idx), recmat.XY(i,idx), 'o-', 'LineWidth',2);
% % end
% set(gca, 'FontSize',20);
% title('Image Query vs. Text Database');h = get(gca, 'title');set(h, 'FontSize', 20);
% xlabel('Recall');h = get(gca, 'xlabel');set(h, 'FontSize', 20);
% ylabel('No. points');h = get(gca, 'ylabel');set(h, 'FontSize', 20);
% 
% figure;hold on;
% plot(mean(retmat.YX,1), mean(recmat.YX,1), 'bo-', 'LineWidth',2);
% % for i = 1:size(retmat.YX,1)
% %     idx = find(retmat.YX(i,:)>0);
% %     plot(retmat.YX(i,idx), recmat.YX(i,idx), 'o-', 'LineWidth',2);
% % end
% set(gca, 'FontSize',20);
% title('Text Query vs. Image Database');h = get(gca, 'title');set(h, 'FontSize', 20);
% xlabel('Recall');h = get(gca, 'xlabel');set(h, 'FontSize', 20);
% ylabel('No. points');h = get(gca, 'ylabel');set(h, 'FontSize', 20);