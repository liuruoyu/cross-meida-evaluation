function kcode = one_k_coding(data, nclasses)
% Creates the 1-k coding of the data for nclasses. 
%   e.g. label 3 will become [0 0 1 0 0 0 0 0 0 0] in case of nclasses = 10
% Returns a matrix of dimensions: [size(data,1), nclasses]

%   This file is part of netlabExtensions.
%   August 2011
%   Jonathan Masci <jonathan@idsia.ch>

values = 1:size(data,1);
kcode = full(sparse(values, data, 1, size(data,1), nclasses));
