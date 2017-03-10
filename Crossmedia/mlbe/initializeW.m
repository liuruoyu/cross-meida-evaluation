function [W] = initializeW(K)
    W = randn(K,K); W = (W +W')/2;
end