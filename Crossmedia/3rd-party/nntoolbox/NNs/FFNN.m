classdef FFNN < INN
    properties
        h
        onGpu
        lossFun
        batchSize = intmax;
        nwts
        layers
        acts
        
        getX = @(x,idxs) x(idxs,:)
        getT = @(x,idxs) x(idxs,:)
        getNData = @(x) size(x,1)
    end
    
    methods
        function obj = FFNN(lossFun)
            obj.lossFun = lossFun;
            obj.layers = {};
            obj.acts = {};
            obj.h = {};
            obj.nwts = 0;
        end
        
        function addLayer(obj, l, act)
            obj.layers{end+1} = l;
            obj.acts{end+1} = act;
            obj.nwts = obj.nwts + l.nParams();
        end
        
        function y = fwd(obj, x)
            obj.h{1} = obj.acts{1}.apply(obj.layers{1}.fwd(x));
            for i = 2:length(obj.layers)
                obj.h{i} = obj.acts{i}.apply(obj.layers{i}.fwd(obj.h{i-1}));
            end
            
            y = obj.h{end};
        end
        
        function [cost, g] = grad(obj, w, x, y)
            obj.setParams(w);
            
            nData = obj.getNData(x);
            nBatches = ceil(nData / obj.batchSize);
            
            to = min(obj.batchSize, nData);
            [cost, g] = obj.grad_(obj.getX(x, 1:to), obj.getT(y, 1:to));
            
            for i=2:nBatches
                from = (i - 1) * obj.batchSize + 1;
                to = min(i * obj.batchSize, nData);
                
                [cost_, g_] = obj.grad_(obj.getX(x, from:to), obj.getT(y, from:to));
                
                cost = cost + cost_;
                g = g + g_;
            end
            
            obj.h = {};
        end
        
        function n = nParams(obj)
            n = obj.nwts;
        end
        
        function w = getParams(obj)
            tmp = obj.layers{1}.getParams();
            w = zeros(obj.nParams(), 1, class(tmp));
            mark = numel(tmp);
            w(1:mark) = tmp;
            for i=2:length(obj.layers)
                tmp = obj.layers{i}.getParams();
                w(mark+1:mark+numel(tmp)) = tmp;
                mark = mark + numel(tmp);
            end
        end
        
        function setParams(obj, w)
            assert(length(w) == obj.nParams());
            
            mark = obj.layers{1}.nParams();
            tmp = w(1:mark);
            obj.layers{1}.setParams(tmp);
            
            for i=2:length(obj.layers)
                tmp = w(mark+1:mark+obj.layers{i}.nParams());
                obj.layers{i}.setParams(w(mark+1:mark+numel(tmp)));
                mark = mark + obj.layers{i}.nParams();
            end
        end
        
        function o = clone(obj)
            o = FFNN(obj.lossFun);
            for i=1:length(obj.layers)
                o.addLayer(obj.layers{i}.clone(), obj.acts{i});
            end
            o.setParams(obj.getParams());
        end
        
        function computeGradOnly(obj)
            error('not implemented for FFNN')
        end
        
        function computeGradOnlyCell(obj)
            error('not implemented for FFNN')
        end
        
        function [cost, g] = grad_(obj, x, y)
            gc = cell(1, length(obj.layers));
            
            obj.fwd(x);
            [cost, deltas] = obj.lossFun.apply(obj.h{end}, y);
            
            if obj.lossFun.derivOut()
                deltas = obj.acts{end}.deriv1(obj.h{end}, deltas);
            end
            
            for i = length(obj.layers):-1:2
                % Evaluate i-th layer's gradient.
                gc{i} = obj.layers{i}.grad(obj.h{i-1}, deltas);
                
                % Now do the backpropagation.
                deltas = obj.layers{i}.bkp(deltas);
                deltas = obj.acts{i}.deriv1(obj.h{i-1}, deltas);
            end
            
            % Evaluate first layer's gradient.
            gc{1} = obj.layers{1}.grad(x, deltas);
            
            % Put the gradient into one column vector
            g = [];
            for i=1:length(obj.layers)
                g = [g; gc{i}(:)];
            end
        end
    end
end
