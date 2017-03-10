function [w] = updatew_scalar_bern(w0, U, V, S, O, params)

% learn w
step = 1e-4;
showtips = 0;
UM = U'*V;ids = O>0;
obj = computeobj_bern(w0, UM, S,ids, params);
if showtips == 1, fprintf('new objective: %.4f\n', obj);end
for i = 1:100
    SigMa = logsig(w0*UM);
    GM = S.*(1-SigMa).*UM-(1-S).*SigMa.*UM;
    
    gradientw = params.phi*w0 - sum(GM(ids));
    nw = w0 - step*gradientw;
    newobj = computeobj_bern(nw,UM,S,ids,params);
    if showtips == 1, fprintf('new objective: %.4f\n', newobj);end
    if obj - newobj > 1e-6
        obj = newobj;
        w0 = nw;
    elseif obj - newobj >= 0
        w0 = nw;
        break;
    else
        break;
    end    
end
w = w0;
end

function obj = computeobj_bern(w, UM, S, ids, params)

% UM = U'*V;
SigMa = logsig(w*UM);
GM = S.*log(SigMa-eps) + (1-S).*log(1-SigMa+eps);

% ids = O>0;
obj = 0.5*params.phi*w*w -  sum(GM(ids));
end