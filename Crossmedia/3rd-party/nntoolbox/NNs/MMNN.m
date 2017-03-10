classdef MMNN < handle
    %multi-modal similarity net
    
    properties
        net1
        net2
        CMnet
        
        margin
        batchSize
        nwts
        
        alpha1
        alpha2
        
        getM1data
        getM2data
        getNData
    end
    
    methods(Static)
        function checkInput(x)
            if ~isstruct(x)
                error('input data is not struct');
            end
            
            if ~isfield(x, 'M1')
                error('field M1 missing');
            end
            SNN.checkInput(x.M1);
            
            if ~isfield(x, 'M2')
                error('field M2 missing');
            end
            SNN.checkInput(x.M2);
            
            if ~isfield(x, 'CM')
                error('field CM missing');
            end
            SNN.checkInput(x.CM);
        end
    end
    
    methods
        function obj = MMNN(net1, net2, margin, alpha1, alpha2)
            % net1: siamese net for modality 1
            % net2: siamese net for modality 2
            obj.nwts = net1.nParams() + net2.nParams();
            
            obj.net1 = net1;
            obj.net2 = net2;
            obj.CMnet = CMNN(net1.net, net2.net, margin);
            
            obj.margin = margin;
            obj.batchSize = 1E6;
            
            obj.alpha1 = alpha1;
            obj.alpha2 = alpha2;
        end
        
        function o = clone(obj)
            o = MMNN(obj.net1.clone(), obj.net2.clone(), obj.margin, obj.alpha1, obj.alpha2);
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
            MMNN.checkInput(x);
            
            %% set params
            obj.setParams(w);
            
            %% Get gradients and costs
            % Modality 1
            [cost1, g1] = obj.net1.grad(obj.net1.getParams, x.M1, x.M1.t);
            
            % Modality 2
            [cost2, g2] = obj.net2.grad(obj.net2.getParams, x.M2, x.M2.t);
            
            % Cross-modality
            [costCM, gCM] = obj.CMnet.grad(obj.CMnet.getParams, x.CM, x.CM.t);
            
            %% cost for MMNN
            cost = costCM + obj.alpha1 * cost1 + obj.alpha2 * cost2;
            
            %% grad
            g = gCM + [obj.alpha1 * g1; obj.alpha2 * g2];
        end
        
        function [cost, g] = gradCell(obj, w, x, t)
            MMNN.checkInput(x);
            
            %% set params
            obj.setParams(w);
            
            %% Get gradients and costs
            % Modality 1
            [cost1, g1] = obj.net1.gradCell(obj.net1.getParams, x.M1, x.M1.t);
            
            % Modality 2
            [cost2, g2] = obj.net2.gradCell(obj.net2.getParams, x.M2, x.M2.t);
            
            % Cross-modality
            [costCM, gCM] = obj.CMnet.gradCell(obj.CMnet.getParams, x.CM, x.CM.t);
            
            %% cost for MMNN
            cost = costCM + obj.alpha1 * cost1 + obj.alpha2 * cost2;
            
            %% grad
            g = gCM + [obj.alpha1 * g1; obj.alpha2 * g2];
            
            for i = 1:length(gCM)
                g{i} = cellfun(@plus, gCM{i}, [obj.alpha1 * g1{i}; obj.alpha2 * g2{i}], 'UniformOutput', false);
            end
        end
    end
end
