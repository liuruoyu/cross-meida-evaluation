function myvalidation
    Dim = 8; repeats = 10;
    SimsMat = zeros(repeats,Dim+1);
%     W = randn(Dim,Dim); W = (W+W')/2; mw = max(W(:)); W = W/((Dim)*mw);
    load goodW.mat
    u = randsrc(Dim,1); 
    for i = 1:repeats
       
       SimsMat(i,:) = computesim(u, W);
%         if SimsMat(i,end)<SimsMat(i,1)
%             save 'goodW.mat' W
%         end
    end
    
    figure;multierrorbar(SimsMat,zeros(size(SimsMat)));
%     plot(1:Dim+1,sims,'s-','LineWidth',2,'MarkerSize',10);
end

function sims = computesim(u, W)
Dim = length(u);
sims = zeros(Dim+1,1);
sims(1) = u'*W*u;
for hamdist = 1:Dim
    ind = nchoosek(1:Dim,hamdist);
    sim = 0;
    for n = 1:size(ind,1)
        fv = ones(Dim,1);
        fv(ind(n,:)) = -1;
        sim = sim+u'*W*(u.*fv);
    end
    sims(hamdist+1) = sim/size(ind,1);
end
end