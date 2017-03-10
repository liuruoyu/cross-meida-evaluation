classdef CMNN < handle
    %cross-modal similarity net
    
    properties
        net1
        net2
        
        margin
        batchSize
        nwts
        
        getX1
        getX2
        getT
        getNData
    end
    
    methods(Static)
        function checkInput(x)
            if iscell(x)
                return
            end
            if ~isstruct(x)
                error('input data is not struct');
            end
            
            if ~isfield(x, 'x1')
                error('field x1 missing');
            end
            
            if ~isfield(x, 'x2')
                error('field x2 missing');
            end
        end
    end
    
    methods
        function obj = CMNN(net1, net2, margin)
            obj.nwts = net1.nParams() + net2.nParams();
            
            obj.net1 = net1;
            obj.net2 = net2;
            
            obj.margin = margin;
            obj.batchSize = 1E6;
            
            obj.getX1 = @(x,idxs) x(idxs,:);
            obj.getX2 = @(x,idxs) x(idxs,:);
            obj.getT = @(x,idxs) x(idxs,:);
            obj.getNData = @(x) size(x,1);
        end
        
        function o = clone(obj)
            o = CMNN(obj.net1.clone(), obj.net2.clone(), obj.margin);
            o.batchSize = obj.batchSize;
        end
        
        function y = fwdM1(obj, x)
            y = obj.net1.fwd(x);
        end
        
        function y = fwdM2(obj, x)
            y = obj.net2.fwd(x);
        end
        
        function w = getParams(obj)
            w = [obj.net1.getParams(); obj.net2.getParams()];
        end
        
        function setParams(obj, w)
            obj.net1.setParams(w(1:obj.net1.nParams()));
            obj.net2.setParams(w(obj.net1.nParams()+1:end));
        end
        
        function n = nParams(obj)
            n = obj.nwts;
        end
        
        function [cost, g] = grad(obj, w, x, t)
            CMNN.checkInput(x);
            
            % set params
            obj.setParams(w);
            
            % loop over batches to compute the gradient
            ndata = obj.getNData(x.x1);
            nBatches = ceil(ndata/obj.batchSize);
            
            to = min(obj.batchSize, ndata);
            x_.x1 = obj.getX1(x.x1, 1:to);
            x_.x2 = obj.getX2(x.x2, 1:to);
            t_ = obj.getT(t, 1:to);
            [cost, g] = obj.grad_(x_, t_);
            for i=2:nBatches
                from = (i - 1) * obj.batchSize + 1;
                to = min(i * obj.batchSize, ndata);
                
                x_.x1 = obj.getX1(x.x1, from:to);
                x_.x2 = obj.getX2(x.x2, from:to);
                t_ = obj.getT(t, from:to);
                
                [cost_, g_] = obj.grad_(x_, t_);
                cost = cost + cost_;
                g = g + g_;
            end
        end
        
        function [cost, g] = gradCell(obj, w, x, t)
            SNN.checkInput(x);
            
            % set params
            obj.setParams(w);
            
            % loop over batches to compute the gradient
            ndata = obj.getNData(x.x1); %size(x.x1, 1);
            nBatches = ceil(ndata/obj.batchSize);
            
            to = min(obj.batchSize, ndata);
            x_.x1 = obj.getX1(x.x1, 1:to);
            x_.x2 = obj.getX2(x.x2, 1:to);
            t_ = obj.getT(t, 1:to);
            [cost, g] = obj.gradCell_(x_, t_);
            for i=2:nBatches
                from = (i - 1) * obj.batchSize + 1;
                to = min(i * obj.batchSize, ndata);
                
                x_.x1 = obj.getX1(x.x1, from:to);
                x_.x2 = obj.getX2(x.x2, from:to);
                t_ = obj.getT(t, from:to);
                
                [cost_, g_] = obj.gradCell_(x_, t_);
                cost = cost + cost_;
                for j = 1:length(g)
                    g{j} = cellfun(@plus, g{j}, g_{j}, 'UniformOutput', false);
                end
            end
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    methods(Access=protected)
        function [cost, g] = grad_(obj,x,t)
            %% fwd
            y1 = obj.net1.fwd(x.x1);
            y2 = obj.net2.fwd(x.x2);
            
            %% err
            delout = y1 - y2;
            delout_sq = delout.^2;
            
            % Dissimilar
            l2 = sqrt(sum(delout_sq, 2));
            m = max(0, obj.margin - l2);
            
            c = zeros(size(l2), class(t));
            c(l2 ~= 0,:) = m(l2 ~= 0,:) ./ l2(l2 ~= 0,:);
            c(l2 == 0,:) = obj.margin;
            
            c_rep = repmat(-c .* (1 - t), [1, size(delout,2)]);
            c_rep = c_rep .* delout;
            t_rep = repmat(t, [1, size(delout,2)]);
            delout = t_rep .* delout + c_rep;
            % it is just the opposite sign so it is not needed
            % delout2 = -t_rep .* delout - c_rep;
                        
            %% cost
            outerr = t .* sum(delout_sq, 2) + (1 - t) .* sum(m.^2, 2);
            cost = 0.5 * sum(outerr);
            
            
            %% grad
            obj.net1.d{length(obj.net1.h)} = delout;
            obj.net2.d{length(obj.net2.h)} = -delout;
            % deriv out
            obj.net1.d{length(obj.net1.h)} = obj.net1.acts{end}.deriv1(obj.net1.h{end}, obj.net1.d{length(obj.net1.h)});
            obj.net2.d{length(obj.net2.h)} = obj.net2.acts{end}.deriv1(obj.net2.h{end}, obj.net2.d{length(obj.net2.h)});

            obj.net1.bkp();
            obj.net2.bkp();
            
            g = [obj.net1.computeGradOnly(); obj.net2.computeGradOnly()];
        end
        
        function [cost, g] = gradCell_(obj,x,t)
            %% fwd
            y1 = obj.net1.fwd(x.x1);
            y2 = obj.net2.fwd(x.x2);
            
            %% err
            delout = y1 - y2;
            delout_sq = delout.^2;
            
            % Dissimilar
            l2 = sqrt(sum(delout_sq, 2));
            m = max(0, obj.margin - l2);
            
            c = zeros(size(l2), class(t));
            c(l2 ~= 0,:) = m(l2 ~= 0,:) ./ l2(l2 ~= 0,:);
            c(l2 == 0,:) = obj.margin;
            
            c_rep = repmat(-c .* (1 - t), [1, size(delout,2)]);
            c_rep = c_rep .* delout;
            t_rep = repmat(t, [1, size(delout,2)]);
            delout = t_rep .* delout + c_rep;
            % it is just the opposite sign so it is not needed
            % delout2 = -t_rep .* delout - c_rep;
                        
            %% cost
            outerr = t .* sum(delout_sq, 2) + (1 - t) .* sum(m.^2, 2);
            cost = 0.5 * sum(outerr);
            
            
            %% grad
            obj.net1.d{length(obj.net1.h)} = delout;
            obj.net2.d{length(obj.net2.h)} = -delout;
            % deriv out
            obj.net1.d{length(obj.net1.h)} = obj.net1.acts{end}.deriv1(obj.net1.h{end}, obj.net1.d{length(obj.net1.h)});
            obj.net2.d{length(obj.net2.h)} = obj.net2.acts{end}.deriv1(obj.net2.h{end}, obj.net2.d{length(obj.net2.h)});

            obj.net1.bkp();
            obj.net2.bkp();
            
            g1 = obj.net1.computeGradOnlyCell();
            g2 = obj.net2.computeGradOnlyCell();
            for i = 1:length(g1)
                g{i} = cellfun(@plus, g1{i}, g2{i}, 'UniformOutput', false);
            end
        end
    end
end
