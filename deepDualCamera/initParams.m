function [params options] = initParams()

params.regC = 7e-3;
params.regC_Wcat = 5e-4;
params.regC_WcatFeat = 5e-7;
params.regC_Wv = 1e-5;
params.regC_Wo = 1e-6;

params.wordSize = 50;
params.rankWo = 2;
params.NN = 3;

params.numInContextWords = 3; % number inner context words
params.numOutContextWords = 3; % number outter context words

params.tinyDataSet = ismac;
params.categories = 19;


%% paths
params.paths.data = '../dataCamera/';
params.paths.outputFolder = '../output/emnlp/';
params.paths.results = [params.paths.outputFolder 'results/'];

if ~exist(params.paths.outputFolder,'dir')
    mkdir(params.paths.outputFolder);
end
if ~exist(params.paths.results,'dir')
    mkdir(params.paths.results);
end


%% used for optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% L-BFGS
options.Method = 'lbfgs';
if params.tinyDataSet
    options.MaxIter = 5;
else
    options.MaxIter = 26;
end

params.test_without_external_features = 1;
params.test_with_external_features = 1;
return