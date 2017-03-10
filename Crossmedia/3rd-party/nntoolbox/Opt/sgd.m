function [netout options] = sgd(net, options, x, t)
% The learning supports mini batches, momentum and weight decay.

% Jonathan Masci <jonathan.masci@gmail.com>


if isempty(options) 
    options.nepochs = 1;
    options.lrate = 0.001;
    options.lratedecay = 0.97;
    options.momentum = 0.9;
    options.minibatch = 1;
%     options.getX = @(x,idxs) {net.getX(x,idxs)};
%     options.getT = @(x,idxs) net.getT(x,idxs);
    options.getData = @(x,t,idxs) deal(x(idxs,:), t(idxs,:));
    options.getNData = @(x) net.getNData(x);
    options.grad = @(w,x,t) net.grad(w,x,t);
    
    
    options.validFun = @(net) 0;
    options.testAfterK = 5;
    options.testFun = @(net) 0;
    
    options.doAtEachK = 5; % execute a function every doAtEachK iterations
    options.doAtEachKfun = @(net, e) 0;
    
    options.shuffle = 1;
end

fprintf('Training for %i epochs with lrate %f, lratedecay %f and mu %f\n',options.nepochs,options.lrate,options.lratedecay,options.momentum);

% Extract initial weights from the network
w = net.getParams();
dwold = zeros(length(w),1,class(w));
netout = net;


ndata = options.getNData(x);
nBatches = ceil(ndata / options.minibatch);


for e=1:options.nepochs
    tic;

    cost = 0;
    
    if options.shuffle
        idxs = randperm(ndata);
    else
        idxs = 1:ndata;
    end
    
    % just to monitor the weight change per epoch
    we = w;
    fprintf('        ');
    for i=1:nBatches    
        fprintf('\b\b\b\b\b\b%05.02f%%',(i/nBatches)*100)
        
        from = (i - 1) * options.minibatch + 1;
        to = min(i * options.minibatch, ndata);

%         data = options.getX(x,idxs(from:to));
%         tt = options.getT(t,idxs(from:to));
        [data, tt] = options.getData(x,t,idxs(from:to));
            
        [cost_, g] = options.grad(w,data,tt);
        cost = cost + cost_;
        
        dw = options.momentum * dwold(:) - options.lrate * (g(:) / (to - from + 1));
        w = w(:) + dw(:);
        
        net.setParams(w);
        w = net.getParams(); % in case setParams has some constraints for sum=1 etc

        dwold = dw;
        
        clear data tt
    end
    fprintf('\n');
    
    elapsed_time = toc;
    
    options.lrate = options.lrate * options.lratedecay;
    
    fprintf('\n================================\nEpoch %i\nElapsed time %f, lrate = %f\t ||w||=%f\nCost = %f\n================================\n', e, elapsed_time, options.lrate, norm(w - we), cost/ndata);
%     if isnan(norm(w - we))
%         warning('sgd returned because ||w|| = NaN')
%         return
%     end
    
    drawnow;
    pause(.1);
    
    options.validFun(net);
    
    if mod(e,options.testAfterK) == 0
        options.testFun(net);
    end
    
    if mod(e,options.doAtEachK) == 0
        options.doAtEachKfun(net, e);
    end
end
