function y = randsample(s, n, k, replace, w)
%RANDSAMPLE Random sample, with or without replacement.
%   Y = RANDSAMPLE(N,K) returns Y as a vector of K values sampled 
%   uniformly at random, without replacement, from the integers 1:N.
%
%   Y = RANDSAMPLE(POPULATION,K), where POPULATION is a vector of two or more
%   values, returns K values sampled uniformly at random, without replacement,
%   from the values in the vector POPULATION.
%
%   Y = RANDSAMPLE(...,REPLACE) returns a sample taken with replacement if
%   REPLACE is true, or without replacement if REPLACE is false (the default).
%
%   Y = RANDSAMPLE(...,true,W) returns a weighted sample, using positive
%   weights W, taken with replacement.  W is often a vector of probabilities.
%   This function does not support weighted sampling without replacement.
%
%   Y = RANDSAMPLE(S, ...) uses the random stream S for random number 
%   generation.  S is a random stream created using RandStream. 
%   Default is the MATLAB default random number stream.
%
%   Example:  Generate a random sequence of the characters ACGT, with
%   replacement, according to specified probabilities.
%
%      R = randsample('ACGT',48,true,[0.15 0.35 0.35 0.15])
%
%   See also RAND, RANDPERM, RANDSTREAM.

%   Copyright 1993-2009 The MathWorks, Inc.
%   $Revision: 1.1.4.7 $  $Date: 2009/10/10 20:11:17 $


% Process the stream argument, if present
if ~isnumeric(s) && isa(s,'RandStream')  % simple test first for speed
    defaultStream = false;
    nargs = nargin - 1;
    if nargs < 2
        error('stats:randsample:TooFewInputs','Not enough input arguments.');
    end
else
    defaultStream = true;
    nargs = nargin;
    if nargs < 2
        error('stats:randsample:TooFewInputs','Not enough input arguments.');
    end
    % shift right to drop s from the argument list
    if nargs == 4
        w = replace;
        replace = k;
    elseif nargs == 3
        replace = k;
    end

    k = n;
    n = s;
end

if numel(n) == 1
    population = [];
else
    population = n;
    n = numel(population);
    if length(population)~=n
        error('stats:randsample:BadPopulation','POPULATION must be a vector.');
    end
end

if nargs < 3
    replace = false;
end

if nargs < 4
    w = [];
elseif ~isempty(w)
    if length(w) ~= n
        if isempty(population)
            error('stats:randsample:InputSizeMismatch',...
                'W must have length equal to N.');
        else
            error('stats:randsample:InputSizeMismatch',...
                'W must have the same length as the population.');
        end
    else
        sumw = sum(w);
        if ~(sumw > 0) || ~all(w>=0) % catches NaNs
            error('stats:randsample:InvalidWeights',...
                  'W must contain non-negative values with at least one positive value.');
        end
        p = w(:)' / sumw;
    end
end

switch replace
    
    % Sample with replacement
    case {true, 'true', 1}
        if isempty(w)
            if defaultStream
                y = randi(n,k,1);
            else
                y = randi(s,n,k,1);
            end

        else
            edges = min([0 cumsum(p)],1); % protect against accumulated round-off
            edges(end) = 1; % get the upper edge exact
            if defaultStream
                [~, y] = histc(rand(k,1),edges);
            else
                [~, y] = histc(rand(s,k,1),edges);
            end
        end
        
    % Sample without replacement
    case {false, 'false', 0}
        if k > n
            if isempty(population)
                error('stats:randsample:SampleTooLarge',...
                    'K must be less than or equal to N for sampling without replacement.');
            else
                error('stats:randsample:SampleTooLarge',...
                    'K must be less than or equal to the population size.');
            end
        end
        
        if isempty(w)
            % If the sample is a sizeable fraction of the population,
            % just randomize the whole population (which involves a full
            % sort of n random values), and take the first k.
            if 4*k > n
                if defaultStream
                    rp = randperm(n);
                else
                    rp = randperm(s,n);
                end
                y = rp(1:k);
                
            % If the sample is a small fraction of the population, a full sort
            % is wasteful.  Repeatedly sample with replacement until there are
            % k unique values.
            else
                x = zeros(1,n); % flags
                sumx = 0;
                while sumx < floor(k) % prevent infinite loop when 0<k<1
                    if defaultStream
                        x(randi(n,1,k-sumx)) = 1; % sample w/replacement
                    else
                        x(randi(s,n,1,k-sumx)) = 1; % sample w/replacement
                    end
                    sumx = sum(x); % count how many unique elements so far
                end
                y = find(x > 0);
                if defaultStream
                    y = y(randperm(k));
                else
                    y = y(randperm(s,k));
                end
            end
        else
            error('stats:randsample:NoWeighting',...
                'Weighted sampling without replacement is not supported.');

        end
    otherwise
        error('stats:randsample:BadReplaceValue',...
            'REPLACE must be either true or false.');
end

if ~isempty(population)
    y = population(y);
else
    y = y(:);
end
