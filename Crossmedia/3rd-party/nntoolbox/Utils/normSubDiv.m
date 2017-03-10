function y = normSubDiv(x, getX, getNData)

if nargin < 2
    getX = @(x,idxs) x(:,:,:,idxs);
end
if nargin < 3
    getNData = @(x) size(x,4);
end

nChannels = size(getX(x,1),3);

nn = GNN(lossMSE());
nn.addLayer(layerSubtractiveNormalization(nChannels),actLinear(),1,2)
nn.addLayer(layerDivisiveNormalization(nChannels),actLinear(),2,3)
nn.getX = getX;
nn.getNData = getNData;
nn.batchSize = 1;

y = nn.fwdOnly(x, 4);