function allSNN = getNN(Wv,allSNum,allIndicies,params)
% Function to get neighest neighbors of the elements for all sentences.
% Requires a lot of memory. 
addpath(genpath('./tools'))

if ~isfield(params,'tinyDataSet')
    params.tinyDataSet = 0;
end

if isempty(Wv)
    disp('Loading Wv..')
    load([params.paths.data 'pretrainedWeights.mat'],'Wv');
end



elemInds = [];
for s = 1:length(allSNum)
    elemInds = unique([elemInds allSNum{s}(allIndicies(s,:))]);
end

all2Batch = containers.Map(num2cell(elemInds),num2cell(1:length(elemInds)));

disp('Getting euclidean distance...');
if params.tinyDataSet
    M = slmetric_pw(Wv(:,elemInds), Wv(:,1:1000), 'eucdist');
else
    M = slmetric_pw(Wv(:,elemInds), Wv, 'eucdist');
end
disp('Done getting metrics');

lenM = size(M,1);
firstHalf = floor(lenM/2);
disp('Sorting neighbors...')
[dummy inds1] = sort(M(1:firstHalf,:),2);
disp('50%...')
[dummy inds2] = sort(M(firstHalf+1:end,:),2);
inds = [inds1;inds2];
disp('Done sorting neighbors')


allSNN = cell(1,length(allSNum));

for s = 1:length(allSNN)
    allSNN{s} = [inds(all2Batch(allSNum{s}(allIndicies(s,1))),2:11);
        inds(all2Batch(allSNum{s}(allIndicies(s,2))),2:11)]';
end

return







