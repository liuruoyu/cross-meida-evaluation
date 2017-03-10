function [w] = updatew_scalar_gaussian(w0, U, S, params)
% this is new version based on clearer interpretation (1-16-2012)
% this function can be used for updating W^{x} or W^{y}

% learn w
step = 1e-10;
UM = U'*U;

obj = computeobj_gaussian(w0,UM,S,params);
% fprintf('new objective: %.4f\n', obj);
for i = 1:100
    
    SigMa = logsig(w0*UM);
    gradientw = params.phi*w0 - params.theta*sum(sum(triu((S-SigMa).*SigMa.*(1-SigMa).*UM)));

    nw = w0 - step*gradientw;
    newobj = computeobj_gaussian(nw,UM,S,params);
    
    if newobj>obj
        nw = w0;
        break;
    elseif abs(newobj-obj)<1e-6
        break;
    else
%         fprintf('new objective: %.6f\n', newobj);
        obj = newobj;
        w0 = nw;
    end    
end
w = nw;

end

function obj = computeobj_gaussian(w,UM,S,params)

% UM = U'*U;
SigMa = logsig(w*UM);
obj = 0.5*params.phi*w*w + 0.5*params.theta*sum(sum(triu((S-SigMa).^2)));
end