function [externalData params] = addExternalFeatures(data,params,type)

[allHypVecs allNERVecs allPOSVecs] = loadExternalFeatures(params,type);

% add in external features
params.hyp = numel(allHypVecs{1});
params.NER = numel(allNERVecs{1});
params.POS = numel(allPOSVecs{1});

externalData = zeros(size(data,1)+params.hyp+params.NER+params.POS,size(data,2));
for s = 1:size(data,2)
    externalData(:,s)=[data(:,s); allHypVecs{s}(:); allNERVecs{s}(:); allPOSVecs{s}(:)];
end
return
