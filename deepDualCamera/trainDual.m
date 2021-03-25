function trainDual(params,options)

% change matlabs default start-up seed
stream = RandStream('mt19937ar','Seed',sum(100*clock));
RandStream.setDefaultStream(stream);

addpath(genpath('./tools'))

if ~exist('params','var')
    [params options] = initParams();
end

[allSNum, allSStr, allSTree, allSNN,allIndicies, categories,sentenceLabels] = loadData(params,'Train');

params = getInternalFeaturesStats(allSStr,allSTree,allIndicies,params);

% load pre_trained weights
load([params.paths.data 'pretrainedWeights.mat'],'Wv','W','WO','words');


% Init Dual Parameters
Wo = 0.01*randn(params.wordSize + 2*params.wordSize*params.rankWo,length(words));
Wo(1:params.wordSize,:) = ones(params.wordSize,size(Wo,2));
Wcat = randWcat(params);


% matlab pooling
success = false;
for i = 1:5
    try
        if ~ismac && isunix && matlabpool('size') == 0 && (~isfield(options,'DerivativeCheck') || (isfield(options,'DerivativeCheck') && strcmpi(options.DerivativeCheck,'off')))
            numCores = feature('numCores')
            if numCores==16
                numCores=8
            end
            matlabpool('open',numCores);
        end
        success=true;
        break;
    catch err
        display(['Error: ' err.message ' retrying...']);
    end
end
if ~success
    display('Retries unsuccesfull');
    rethrow(err);
end


% TRAIN
% take only the relevant words
% filter Wv and Wo by this mini-batch, so we can give a smaller vocab to training
if params.tinyDataSet, sentences = 1:10; else sentences = 1:length(allSNum); end

[allSNum_batch, allSNN_batch, Wv_batch, Wo_batch, allWordInds, params] = ...
    getRelevantWords(allSNum,allSNN,sentences,allIndicies,Wv,Wo,params);


% Optimize
[X decodeInfo] = param2stack(Wv_batch,Wo_batch,W,WO,Wcat);
X = minFunc(@costFct_preTrainDual,X,options,decodeInfo,params,allSNum_batch,allSStr(sentences),allSTree(sentences),allSNN_batch,sentenceLabels(sentences),...
    allIndicies(sentences,:));
[Wv_batch,Wo_batch,W,WO, Wcat] = stack2param(X, decodeInfo);


% map vocab back
Wv(:,allWordInds) = Wv_batch;
Wo(:,allWordInds) = Wo_batch;


% Testing
if params.test_without_external_features
    F1 = test_without_external_features(Wv,Wo,W,WO,Wcat,params,'test');
    saveWeights(Wv,Wo,W,WO,Wcat,['weights_WOEF_acc_' num2str(F1)],params);
end


if params.test_with_external_features
    [F1 Wcat_WEF]= test_with_external_features(Wv,Wo,W,WO,Wcat,params,'test');
    saveWeights(Wv,Wo,W,WO,Wcat_WEF,['weights_WEF_acc_' num2str(F1)],params);
end

if isunix && ~ismac
    matlabpool close
end
return


function saveWeights(Wv,Wo,W,WO,Wcat, fileName, params)
save([params.paths.outputFolder fileName '.mat'],'Wv','Wo','W','WO','Wcat','params');
return
