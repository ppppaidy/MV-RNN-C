function [allSNum_batch,allSNN_batch,Wv_batch,Wo_batch,allWordInds, params] =...
    getRelevantWords(allSNum, allSNN,batchInd, allIndicies,Wv,Wo,params)
% Becasuse not all words are used in our dictionary, only the relvant words
% in these sentences will be used to save memory. 


%% add to our batch the padding and nearest neighbors
allWordInds = 1;
for s = batchInd
    allWordInds = unique([allWordInds allSNum{s} reshape(allSNN{s}(1:params.NN,:),1,[])]);
end

allWordInds=allWordInds(2:end);% ignore -1
all2Batch = containers.Map(num2cell(allWordInds),num2cell(1:length(allWordInds)));
% get index of mid context padding, start sentence is always 1
Wv_batch = Wv(:,allWordInds);
Wo_batch = Wo(:,allWordInds);
disp('Updating allSNum to new batch numbers...');
allSNum_batch = allSNum(batchInd);
allSNN_batch = allSNN(batchInd);
%% update all indicies for the batch
for s = 1:length(allSNum_batch)
    for w = 1:length(allSNum_batch{s})
        if allSNum_batch{s}(w)>0
            allSNum_batch{s}(w) = all2Batch(allSNum_batch{s}(w));
        end
    end
    allSNN_batch{s}(params.NN+1:end,:) = [];
    for e = 1:2
        for nn = 1:params.NN
            allSNN_batch{s}(nn,e) = all2Batch(allSNN_batch{s}(nn,e));
        end
    end
end

return
