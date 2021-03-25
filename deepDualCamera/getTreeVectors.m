function [data sentenceLabels categories] = getTreeVectors(Wv,Wo,W,WO,Wcat,params,type)

[allSNum, allSStr, allSTree, allSNN,allIndicies, ...
    categories, sentenceLabels] = loadData(params,type);

[dummy fanIn] = randWcat(params);
data = zeros(fanIn, length(allSNum));

onlyGetVectors = true;

if params.tinyDataSet
    sentences = 1:10;
else
    sentences = 1:length(allSNum);
end

for s = sentences
    
    tree = forwardPropTree(allSNum{s},allSTree{s},allSStr{s},allSNN{s},allIndicies(s,:), ...
        Wv,Wo,W,WO,Wcat,params, onlyGetVectors);
    
    data(:,s) = [tree.pooledVecPath(:);tree.NN_vecs(:);tree.features(:)];
    if mod(s,1000) == 0
        disp([type ': Sentence Number ' num2str(s)]);
    end
end
return
