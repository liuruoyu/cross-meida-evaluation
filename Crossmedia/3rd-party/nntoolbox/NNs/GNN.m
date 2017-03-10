classdef GNN < INN
    % Graph neural net.
    % This is the most flexible net in the toolbox.
    properties
        h
        d
        onGpu
        lossFun
        batchSize = intmax;
        nwts
        layers
        acts
        CT
        fwdCounter
        bkpCounter
        
        getX = @(x,idxs) x(idxs,:)
        getT = @(x,idxs) x(idxs,:)
        getNData = @(x) size(x,1)
        
        layersToTrain
    end
    
    methods
        function obj = GNN(lossFun)
            obj.lossFun = lossFun;
            obj.layers = {};
            obj.acts = {};
            obj.h = {};
            obj.nwts = 0;
            obj.fwdCounter = {};
            obj.bkpCounter = {};
            
            obj.layersToTrain = {};
        end
        
        function addLayer(obj, l, a, from, to)
            obj.layers{end+1} = l;
            obj.acts{end+1} = a;
            obj.nwts = obj.nwts + l.nParams();
            
            % Connections
            obj.CT{end+1}.from = from;
            obj.CT{end}.to = to;
            
            % Counters for activation and derivation application
            for i=1:length(to)
                if length(obj.fwdCounter) < to(i)
                    obj.fwdCounter{to(i)} = 1;
                else
                    obj.fwdCounter{to(i)} = obj.fwdCounter{to(i)} + 1;
                end
            end
            for i=1:length(from)
                if length(obj.bkpCounter) < from(i)
                    obj.bkpCounter{from(i)} = 1;
                else
                    obj.bkpCounter{from(i)} = obj.bkpCounter{from(i)} + 1;
                end
            end
            
            % By default the layer is to be trained
            obj.layersToTrain{end+1} = 1;
        end
        
        function y = dumpLayer(obj, x, idLayer)
            nData = obj.getNData(x);
            nBatches = ceil(nData / obj.batchSize);

            to = min(obj.batchSize, nData);
            obj.fwd(obj.getX(x,1:to));
            tmp = obj.h{idLayer};
            if ndims(tmp == 2)
                y = zeros(nData, size(tmp,2), class(tmp));
                y(1:to,:) = tmp;
            else
                y = zeros(size(tmp,1), size(tmp,2), size(tmp,3), nData, class(tmp));
                y(:,:,:,1:to) = tmp;
            end
            for i=2:nBatches
                from = (i - 1) * obj.batchSize + 1;
                to = min(i * obj.batchSize, nData);
                obj.fwd(obj.getX(x,from:to)); 
                tmp = obj.h{idLayer};
                if ndims(tmp == 2)
                    y(from:to,:) = tmp;
                else
                    y(:,:,:,from:to) = tmp;
                end
            end            
        end
        
        function y = fwdOnly(obj, x, nDims)
            % nDims: number of output dimensions
            
            nData = obj.getNData(x);
            nBatches = ceil(nData / obj.batchSize);

            to = min(obj.batchSize, nData);
            tmp = obj.fwd(obj.getX(x,1:to));
            if nDims == 2
                y = zeros(nData, size(tmp,2), class(tmp));
                y(1:to,:) = tmp;
            else
                y = zeros(size(tmp,1), size(tmp,2), size(tmp,3), nData, class(tmp));
                y(:,:,:,1:to) = tmp;
            end
            for i=2:nBatches
                from = (i - 1) * obj.batchSize + 1;
                to = min(i * obj.batchSize, nData);
                tmp = obj.fwd(obj.getX(x,from:to)); 
                if nDims == 2
                    y(from:to,:) = tmp;
                else
                    y(:,:,:,from:to) = tmp;
                end
            end            
        end
        
        function y = fwd(obj, x)
            % Cell input only for batch size 1
            
            obj.h = {};
            if iscell(x)
                for i=1:length(x)
                    obj.h{i} = x{i};
                end
                nin = length(x);
            else
                obj.h{1} = x;
                nin = 1;
            end
            
            counter = zeros(length(obj.layers) + nin, 1);
            
            for i = 1:length(obj.layers)
                tmp = obj.layers{i}.fwdCell(obj.h(obj.CT{i}.from));
                for j = 1:length(tmp)
                    if length(obj.h) < obj.CT{i}.to(j) || isempty(obj.h{obj.CT{i}.to(j)})
                        obj.h{obj.CT{i}.to(j)} = tmp{j};
                    else
                        obj.h{obj.CT{i}.to(j)} = obj.h{obj.CT{i}.to(j)} + tmp{j};
                    end
                    counter(obj.CT{i}.to(j)) = counter(obj.CT{i}.to(j)) + 1;
                    
                    % check whether to activate it
                    if counter(obj.CT{i}.to(j)) == obj.fwdCounter{obj.CT{i}.to(j)}
                        obj.h{obj.CT{i}.to(j)} = obj.acts{i}.apply(tmp{j});
                    end
                end
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
            obj.d = {};
        end
        
        function [cost, g] = gradCell(obj, w, x, y)
            obj.setParams(w);
            
            nData = obj.getNData(x);
            nBatches = ceil(nData / obj.batchSize);
            
            to = min(obj.batchSize, nData);
            [cost, g] = obj.gradCell_(obj.getX(x, 1:to), obj.getT(y, 1:to));
            
            for i=2:nBatches
                from = (i - 1) * obj.batchSize + 1;
                to = min(i * obj.batchSize, nData);
                
                [cost_, g_] = obj.gradCell_(obj.getX(x, from:to), obj.getT(y, from:to));
                
                cost = cost + cost_;
                g = cellfun(@plus, g, g_);
            end
            
            obj.h = {};
            obj.d = {};
        end
        
        function n = nParams(obj)
            n = obj.nwts;
        end
        
        function w = getParams(obj)
            tmp = obj.layers{1}.getParams();
            w = zeros(obj.nParams(), 1, class(gather(tmp)));
            mark = numel(tmp);
            w(1:mark) = gather(tmp);
            for i=2:length(obj.layers)
                tmp = obj.layers{i}.getParams();
                w(mark+1:mark+numel(tmp)) = gather(tmp);
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
            o = GNN(obj.lossFun);
            for i=1:length(obj.layers)
                o.addLayer(obj.layers{i}.clone(), obj.acts{i}, obj.CT{i}.from, obj.CT{i}.to);
            end
            o.setParams(obj.getParams());
            o.layersToTrain = obj.layersToTrain;
        end
        
        function g = computeGradOnly(obj)
            gc = obj.computeGradOnlyCell();
            
            % Put the gradient into one column vector
            g = [];
            for i=1:length(gc)
                tmp = [];
                for j=1:length(gc{i})
                    tmp = [tmp; gc{i}{j}(:)];
                end
                g = [g; tmp];
            end
            
            obj.d = {};
        end
        
        function g = computeGradOnlyCell(obj)
            g = cell(1, length(obj.layers));
            for i = 1:length(obj.layers)
                g{i} = obj.layers{i}.gradCell(obj.h(obj.CT{i}.from), obj.d(obj.CT{i}.to));
                if ~obj.layersToTrain{i}
                    for j=1:length(g{i})
                        g{i}{j} = g{i}{j} * 0;
                    end
                end
            end
            
            obj.d = {};
        end
        
        function bkp(obj)
            % Deltas for the output must be set
            
            counter = zeros(length(obj.layers) + 1, 1);
            
            for i = length(obj.layers):-1:2
                tmp = obj.layers{i}.bkpCell(obj.d(obj.CT{i}.to));
                for j = 1:length(tmp)
                    if length(obj.d) < obj.CT{i}.from(j) || isempty(obj.d{obj.CT{i}.from(j)})
                        obj.d{obj.CT{i}.from(j)} = tmp{j};
                    else
                        obj.d{obj.CT{i}.from(j)} = obj.d{obj.CT{i}.from(j)} + tmp{j};
                    end
                    counter(obj.CT{i}.from(j)) = counter(obj.CT{i}.from(j)) + 1;
                    
                    % check whether to derivate it
                    if counter(obj.CT{i}.from(j)) == obj.bkpCounter{obj.CT{i}.from(j)}
                        obj.d{obj.CT{i}.from(j)} = obj.acts{i-1}.deriv1(obj.h{obj.CT{i}.from(j)}, obj.d{obj.CT{i}.from(j)});
                    end
                end
            end
        end
        
        function [cost, g] = grad_(obj, x, y)
            obj.d = {};
            
            obj.fwd(x);
            
            [cost, obj.d{length(obj.h)}] = obj.lossFun.apply(obj.h{end}, y);
            
            if obj.lossFun.derivOut()
                obj.d{length(obj.h)} = obj.acts{end}.deriv1(obj.h{end}, obj.d{length(obj.h)});
            end
            
            obj.bkp();
            
            g = obj.computeGradOnly();
        end
        
        function [cost, g] = gradCell_(obj, x, y)
            obj.d = {};
            g = cell(1, length(obj.layers));
            
            obj.fwd(x);
            
            [cost, obj.d{length(obj.h)}] = obj.lossFun.apply(obj.h{end}, y);
            
            if obj.lossFun.derivOut()
                obj.d{length(obj.h)} = obj.acts{end}.deriv1(obj.h{end}, obj.d{length(obj.h)});
            end
            
            obj.bkp();
            
            for i = 1:length(obj.layers)
                g{i} = obj.layers{i}.gradCell(obj.h(obj.CT{i}.from), obj.d(obj.CT{i}.to));
            end
        end
    end
end
