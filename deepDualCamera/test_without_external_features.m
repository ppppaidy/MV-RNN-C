function F1 = test_without_external_features(Wv,Wo,W,WO,Wcat,params,type)
% If you'd like to load the SemEval task 8 data provided call:
%
% test_without_external_features(Wv,Wo,W,WO,Wcat,params,type)
% params - must have params.paths.data set to directory of SemEval data
% type - either 'test' or 'train' 
%
% You can also load specific files by calling 
%
% test_without_external_features(Wv,Wo,W,WO,Wcat,params)
% with params.paths.dataFile set to the file to load


if ~exist('type','var')
    type = [];
end

% load testing data
[allSNum, allSStr, allSTree, allSNN,allIndicies, ...
    categories, sentenceLabels] = loadData(params,type);


if params.tinyDataSet, sentences = 1:10; else sentences = 1:length(allSNum); end

predictLabel = zeros(1,length(sentences));

for s = sentences
    
    tree = forwardPropTree(allSNum{s},allSTree{s},allSStr{s}, allSNN{s},allIndicies(s,:), ...
        Wv,Wo,W,WO,Wcat,params);
    
    [p, predictLabel(s)] = max(tree.y(:));
end

disp(predictLabel);


F1 = getF1score(predictLabel,categories,sentenceLabels,'WOEF',params);
if ~isempty(F1)
    disp(['F1 score without external features is: ' num2str(F1)]);
end

return
