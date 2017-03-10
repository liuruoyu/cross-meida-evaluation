function myplot_reccurve(premat, recmat, retmat)

% nlines = size(retmat.XY,1);
% 
% mr = zeros(1,size(recmat.XY,2));
% mp = zeros(1,size(recmat.XY,2));
figure;hold on;
plot(mean(retmat.XY,1), mean(recmat.XY,1), 'bo-', 'LineWidth',2);
% plot(mean(retmat1.XY,1), mean(recmat1.XY,1), 'ro-', 'LineWidth',2);
% for i = 1:nlines
%     ids = find(retmat.XY(i,:)>0);
%     plot(retmat.XY(i,ids), recmat.XY(i,ids), 'bo-', 'LineWidth',2);
%     ids = find(retmat1.XY(i,:)>0);
%     plot(retmat1.XY(i,ids), recmat1.XY(i,ids), 'ro-', 'LineWidth',2);
% end
set(gca, 'FontSize',20);
title('Image Query vs. Text Database');h = get(gca, 'title');set(h, 'FontSize', 20);
xlabel('No. Examples');h = get(gca, 'xlabel');set(h, 'FontSize', 20);
ylabel('Recall');h = get(gca, 'ylabel');set(h, 'FontSize', 20);


figure;hold on;
plot(mean(retmat.YX,1), mean(recmat.YX,1), 'bo-', 'LineWidth',2);
% plot(mean(retmat1.YX,1), mean(recmat1.YX,1), 'ro-', 'LineWidth',2);
% for i = 1:nlines
%     ids = find(retmat.YX(i,:)>0);
%     plot(retmat.YX(i,ids), recmat.YX(i,ids), 'bo-', 'LineWidth',2);
%     ids = find(retmat1.YX(i,:)>0);
%     plot(retmat1.YX(i,ids), recmat1.YX(i,ids), 'ro-', 'LineWidth',2);
% end
set(gca, 'FontSize',20);
title('Text Query vs. Image Database');h = get(gca, 'title');set(h, 'FontSize', 20);
xlabel('No. Examples');h = get(gca, 'xlabel');set(h, 'FontSize', 20);
ylabel('Recall');h = get(gca, 'ylabel');set(h, 'FontSize', 20);