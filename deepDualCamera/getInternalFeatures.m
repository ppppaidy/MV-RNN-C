function features = getInternalFeatures(path, sStr, indicies, params)
path = [0;path;0];
% get depth
[dummy, depth1] = max(path);
depth2 = length(path) - depth1 + 1;
% account for missing leafs
features = [length(path) depth1 depth2 length(sStr) (indicies(2) - indicies(1))];

features = (features - params.features.mean)./params.features.std;

return
