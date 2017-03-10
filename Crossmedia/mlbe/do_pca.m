function [Xtr,Xte] = do_pca(Xtr, Xte, nb)

% performing PCA on X
npca = min(nb, size(Xtr, 2));
opts.disp = 0;
[pc, l] = eigs(cov(Xtr), npca, 'LM', opts);
Xtr = Xtr*pc;
Xte = Xte*pc;
