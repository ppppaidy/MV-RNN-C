function [cost,grad] = costFct_preTrainDual(X,decodeInfo,params,allSNum,allSStr,allSTree,allSNN,sentenceLabels,allIndicies,iterTesting)

[Wv,Wo,W,WO,Wcat] = stack2param(X, decodeInfo);
numSent = length(allSNum);
numFeats = length(params.features.std);
wsz = params.wordSize;


Wv_df = zeros(size(Wv));
Wo_df = zeros(size(Wo));
W_df = zeros(size(W));
WO_df = zeros(size(WO));
Wcat_df = zeros(size(Wcat));
totalCost = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute the cost and gradient for each sentence
parfor t = 1:numSent 
    if length(allSNum{t})==1
        continue;
    end
    [df_s_Wv,df_s_Wo,df_s_W,df_s_WO, df_s_Wcat,cost] = costAndGradOneSent_preTrainDualRNN(allSNum{t},allSTree{t},allSStr{t},...
        allSNN{t},sentenceLabels(t),allIndicies(t,:),Wv,Wo,W,WO,Wcat,params);
    totalCost = totalCost + cost;
    Wcat_df = Wcat_df + df_s_Wcat;
    W_df = W_df + df_s_W;
    WO_df = WO_df + df_s_WO;
    Wv_df = Wv_df + df_s_Wv;
    Wo_df = Wo_df + df_s_Wo;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Divide by number of sentences
cost = (1/numSent)*totalCost;
Wcat_df = 1/numSent * Wcat_df;
W_df = 1/numSent * W_df;
Wv_df   = 1/numSent * Wv_df;
WO_df   = 1/numSent * WO_df;
Wo_df   = 1/numSent * Wo_df;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Add regularization to gradient
Wcat_df = Wcat_df + [params.regC_Wcat * Wcat(:,1:end-numFeats) params.regC_WcatFeat*Wcat(:,end-numFeats+1:end)];
Wv_df = Wv_df + params.regC_Wv * Wv;
W_df  = W_df + params.regC  * [W(:,1:end-1) zeros(size(W,1),1)];
Wo_df = Wo_df + params.regC_Wo * [(Wo(1:wsz,:)-ones(wsz,size(Wo,2)));Wo(wsz+1:end,:)];
WO_df = WO_df + params.regC * WO(:,1:end);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Add regularization to cost

cost = cost + params.regC_Wcat/2 * sum(sum(Wcat(:,1:end-numFeats).^2)) + ...
    params.regC_WcatFeat/2 * sum(sum(Wcat(:,end-numFeats+1:end).^2)) + ...
    params.regC_Wv/2 * sum(sum(Wv(:).^2)) + params.regC/2 * sum(sum(W(:,1:end-1).^2)) + ...
    params.regC_Wo/2 * sum(sum([(Wo(1:wsz,:)-ones(wsz,size(Wo,2)));Wo(wsz+1:end,:)].^2)) + ...
    params.regC/2 * sum(WO(:).^2);


[grad, dummy] = param2stack(Wv_df,Wo_df,W_df,WO_df,Wcat_df);
grad = full(grad);
return
