function y = randMinMax(w, m, M)
% Return a uniform distributed tensor sized as w with values in the range
% [m,M]

y = m + (M - m) * rand(size(w), class(w));