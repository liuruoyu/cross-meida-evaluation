% Authors: Herve Jegou and Matthijs Douze, firstname.lastname@inria.fr
% Copyright INRIA 2008-2010. License: GPL

% This function computes the AP for a query
function ap = score_ap_from_ranks1 (ranks, nres)

% number of images ranked by the system
nimgranks = length (ranks);  


if 1
ranks = ranks - 1;

% accumulate trapezoids in PR-plot
ap = 0;

recall_step = 1 / nres;

for i = 1:nimgranks
  rank = ranks(i);
  
  if rank == 0
    precision_0 = 1.0;
  else
    precision_0 = (i - 1) / rank;
  end
  
  precision_1 = i / (rank + 1);
  ap = ap + (precision_0 + precision_1) * recall_step / 2;
end
%----------------------------------------
else

  % number of images ranked by the system
nimgranks = length (ranks);  

% accumulate trapezoids in PR-plot
ap = 0;

recall_step = 1 / nres;

for i = 1:nimgranks
  rank = ranks(i);
  
  if rank == 1
    precision_0 = 1.0;
  else
    precision_0 = (i - 1) / (rank - 1);
  end
  
  precision_1 = i / rank;
  ap = ap + (precision_0 + precision_1) * recall_step / 2;
end

end