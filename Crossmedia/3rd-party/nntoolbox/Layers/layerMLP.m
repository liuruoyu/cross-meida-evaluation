classdef layerMLP < ILayer
    properties
        nin
        nout
        datatype
        w
        w_old
        b
        nwts
        unitLength
    end
    
    methods
        function obj = layerMLP(nin, nout, datatype, unitLength)
            obj.nin = nin;
            obj.nout = nout;
            if nargin < 3
                obj.datatype = 'double';
            else
                obj.datatype = datatype;
            end
            if nargin < 4
                unitLength = false;
            end
            obj.unitLength = unitLength;
            
            obj.w = feval(obj.datatype, randn(nin, nout)/sqrt(nin + 1));
            obj.b = feval(obj.datatype, randn(1, nout)/sqrt(nin + 1)) .* 0;
            
            obj.nwts = numel(obj.w) + numel(obj.b);
            
            obj.projectConstraints();
        end
        
        function gw = l2scaledg(obj, gw)
            normeps = 1e-5;
            epssumsq = sum(obj.w_old.^2, 1) + normeps;	
            alpha = 1;
            l2cols = sqrt(epssumsq)*alpha;

            gw = bsxfun(@rdivide, gw, l2cols) - ...
                bsxfun(@times, obj.w, sum(gw .* obj.w_old, 1) ./ epssumsq);
        end
        
        function y = fwd(obj, x)
%             y = x * obj.w + repmat(obj.b, [size(x,1), 1]);
            y = bsxfun(@plus, x * obj.w, obj.b);
        end

        function y = fwdCell(obj, x)
            y = {obj.fwd(x{1})};
        end

        function y = bkp(obj, x)
            y = x * obj.w';
        end
        
        function y = bkpCell(obj, x)
            y = {obj.bkp(x{1})};
        end
        
        function g = grad(obj, x, y)
            gw = x' * y;
            gb = sum(y, 1);
            
            if obj.unitLength
                gw = obj.l2scaledg(gw);
            end
            
            g = [gw(:); gb(:)];
        end
        
        function g = gradCell(obj, x, y)
            x = x{1};
            y = y{1};
            
            g = cell(2,1);
            g{1} = x' * y;
            if obj.unitLength
                g{1} = obj.l2scaledg(g{1});
            end
            g{2} = sum(y, 1);
        end
        
        function n = nParams(obj)
            n = obj.nwts;
        end
        
        function w = getParams(obj)
            w = [obj.w(:); obj.b(:)];
        end
        
        function w = getParamsCell(obj)
            w{1} = obj.w;
            w{2} = obj.b(:);
        end
        
        function setParams(obj, w)
            obj.w = reshape(w(1:numel(obj.w)), size(obj.w));
            obj.b = reshape(w(numel(obj.w)+1:end), size(obj.b));
            
            
            obj.projectConstraints()
        end
        
        function setParamsCell(obj, w)
            obj.w = w{1};
            obj.b = w{2};
            
            obj.projectConstraints()
        end
        
        function n = getNOut(obj)
            n = obj.nout;
        end
        
        function o = clone(obj)
            o = layerMLP(obj.nin, obj.nout, obj.datatype);
            o.setParams(obj.getParams());
        end
        
        function projectConstraints(obj)
            if obj.unitLength
                obj.w_old = obj.w;
                
                normeps = 1e-5;
                epssumsq = sum(obj.w.^2, 1) + normeps;
                alpha = 1; % make it a param?
                l2 = sqrt(epssumsq) * alpha;
                
                obj.w = bsxfun(@rdivide, obj.w, l2);
            end
        end
    end
end