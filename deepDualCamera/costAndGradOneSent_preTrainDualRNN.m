function [df_s_Wv,df_s_Wo,df_s_W,df_s_WO,df_s_Wcat,cost] = costAndGradOneSent_preTrainDualRNN(sNum,sTree,sStr,sNN,label,indicies,Wv,Wo,W,WO,Wcat,params)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forward Prop: Greedy Tree Parse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tree = forwardPropTree(sNum,sTree,sStr,sNN,indicies,Wv,Wo,W,WO,Wcat,params);

cost = -log(tree.y(label));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Backprop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[df_s_Wcat,tree.nodeVecDeltas,tree.NN_deltas,paddingDelta] = backpropPool(tree, label, Wcat, params);

deltaDown_vec = zeros(params.wordSize,1);
deltaDown_op = zeros(params.wordSize,params.wordSize);

topNode = tree.getTopNode();
[df_s_Wv,df_s_Wo,df_s_W,df_s_WO] = backpropAll(tree,W,WO,Wo,params,deltaDown_vec,deltaDown_op,topNode,size(Wv,2),indicies,sNN);

%Backprop into Padding
df_s_Wv(:,1) = df_s_Wv(:,1) + paddingDelta;