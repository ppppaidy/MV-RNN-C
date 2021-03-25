function [F1 Wcat]= test_with_external_features(Wv,Wo,W,WO,Wcat,params,type)

if ~exist('type','var')
    type = [];
end

[test_data sentenceLabelsTest categories] = getTreeVectors(Wv,Wo,W,WO,Wcat,params,type);
test_data = addExternalFeatures(test_data,params,type);

% Train if Wcat is not the right size
if (size(Wcat,2) ~= size(test_data,1)) 
    
    disp('Training Wcat on external features');
    
    [train_data sentenceLabels] = getTreeVectors(Wv,Wo,W,WO,Wcat,params,'Train');
    [train_data params] = addExternalFeatures(train_data,params,'Train');
    
    
    % get a new Wcat with external features
    WcatOrig = Wcat;
    Wcat = randWcat(params,true);
    Wcat(:,1:size(WcatOrig,2)) = WcatOrig;
    [X decodeInfo] = param2stack(Wcat);
    
    % params & options
    params.regC_Wcat2 = 2e-3; 
    params.regC_WcatFeat = 5e-7; 
    params.regC_WcatHyp = 5e-4; 
    params.regC_WcatNER = 1e-5; 
    params.regC_WcatPOS = 1e-5; 
    

    options.MaxIter = 16;
    options.display = 'on';
    options.Method = 'lbfgs';
    
    % optimize
    X = minFunc( @softmaxCost, X, options, decodeInfo, train_data, params, sentenceLabels);
    
    Wcat = stack2param(X, decodeInfo);
end

num = exp(Wcat * test_data);
y = num./repmat(sum(num),size(num,1),1);


OtherThreshold = 0.3;
predictLabel = zeros(1,size(y,2));
for s = 1:size(y,2)
    [p, predictLabel(s)] = max([y(1:9,s); 0; y(11:end,s)]);
    
    if ~(p > OtherThreshold)
        predictLabel(s) = 10;
    end
end

F1 = getF1score(predictLabel,categories,sentenceLabelsTest,'WEF',params);
if ~isempty(F1)
    display(['F1 score with external features: ' num2str(F1)]);
end
return;
