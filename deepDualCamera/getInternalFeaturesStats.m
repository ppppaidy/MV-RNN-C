function params = getInternalFeaturesStats(allSStr,allSTree,allIndicies,params)
% features [path length, path depth1, path depth2, sentence length, length between elements]
numFeatures = 5;

params.features.mean = 0;
params.features.std = ones(1,numFeatures);

allFeatures = zeros(length(allSTree),numFeatures);

thisTree = tree();
for s = 1:length(allSTree)
    thisTree.pp = allSTree{s}';
    fullPath = findPath(thisTree,allIndicies(s,:));
    allFeatures(s,:) = getInternalFeatures(fullPath, allSStr{s},allIndicies(s,:),params);
end
% Find the mean
params.features.mean = mean(allFeatures,1);
allFeatures = bsxfun(@minus,allFeatures, params.features.mean);
% Find the standard deviation
params.features.std  = std(allFeatures);

% the command below would then give allFeatures a mean of 0 and std 1
% allFeatures = bsxfun(@rdivide,allFeatures,params.features.std);
return



