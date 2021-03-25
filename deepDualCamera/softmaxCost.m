function [cost,grad] = softmaxCost(X, decodeInfo, train_data, params, sentenceLabels,iterTesting)

Wcat = stack2param(X,decodeInfo);

numLabels = size(Wcat,1);

calcs = exp(Wcat * train_data);

m = length(sentenceLabels);
grad = zeros(size(Wcat));
cost = 0;
for i = 1:m
    cost = cost + log(calcs(sentenceLabels(i),i)/sum(calcs(:,i)));
    
    I = eye(numLabels);
    truth = I(:,sentenceLabels(i));
    error = truth - calcs(:,i)/sum(calcs(:,i));
    grad = grad + error * train_data(:,i)';
end

numFeats = length(params.features.std);

hyp = params.hyp;
ner = params.NER;
pos = params.POS;


cost = -(1/m)*cost + params.regC_Wcat2/2 * sum(sum(Wcat(:,1:end-numFeats-hyp-ner-pos).^2)) + ...
    params.regC_WcatFeat/2 * sum(sum(Wcat(:,end-numFeats-hyp-ner-pos+1:end-hyp-ner-pos).^2)) + ...
    params.regC_WcatHyp/2 * sum(sum(Wcat(:,end-hyp-ner-pos+1:end-ner-pos).^2)) + ...
    params.regC_WcatNER/2 * sum(sum(Wcat(:,end-ner-pos+1:end-pos).^2)) + ...
    params.regC_WcatPOS/2 * sum(sum(Wcat(:,end-pos+1:end).^2));

grad = -(1/m)*grad + [params.regC_Wcat2 * Wcat(:,1:end-numFeats-hyp-ner-pos) ...
    params.regC_WcatFeat * Wcat(:,end-numFeats-hyp-ner-pos+1:end-hyp-ner-pos)  ...
    params.regC_WcatHyp * Wcat(:,end-hyp-ner-pos+1:end-ner-pos) ...
    params.regC_WcatNER * Wcat(:,end-ner-pos+1:end-pos) ...
    params.regC_WcatPOS * Wcat(:,end-pos+1:end)];

[grad dummy] = param2stack(grad);
return