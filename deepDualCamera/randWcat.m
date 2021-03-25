function [Wcat fanIn] = randWcat(params,withExternalFeatures)

if ~exist('withExternalFeatures','var')
    params.hyp = 0; params.NER = 0; params.POS = 0;
end

% for top node
l = 1;

% for the elements
l = l + 2;

% for the averaged outer words
l = l + 2;

% for all the inner context words
l = l + params.numInContextWords*2;

% for averaged nearest neighbors
l = l + 2;

numFeats = length(params.features.std);

fanIn = (l * params.wordSize)+numFeats+params.hyp+params.NER+params.POS;

Wcat = 0.005*randn(params.categories ,fanIn);

