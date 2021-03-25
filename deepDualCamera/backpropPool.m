function [df_s_Wcat, nodeVecDeltas, NN_deltas, paddingDelta] = backpropPool(tree, label,Wcat, params)

I = eye(length(tree.y));
groundTruth = I(:,label);

numFeatures = length(params.features.std);

catInput = [tree.pooledVecPath(:);tree.NN_vecs(:);tree.features(:)];

topDelta = tree.y - groundTruth;
df_s_Wcat = topDelta*catInput';

nodeVecDeltas = zeros(size(tree.nodeAct_a));
paddingDelta = zeros(size(tree.nodeAct_a,1),1);

% Get the deltas for each node to pass down
nodeDeltas = tree.poolMatrix * Wcat(:,1:end-numFeatures)' * topDelta;
nodeDeltas = reshape(nodeDeltas,params.wordSize,[]);

% Pass out the deltas to each node
for n = 1:length(tree.nodePath)
    node = tree.nodePath(n);
   if node == 0
      paddingDelta = paddingDelta + nodeDeltas(:,n);
   else
       if tree.isLeafVec(node)
           nodeVecDeltas(:,node) = nodeVecDeltas(:,node) + nodeDeltas(:,n);
       else
           nodeVecDeltas(:,node) = nodeVecDeltas(:,node) + nodeDeltas(:,n) .* (1 - tree.nodeAct_a(:,node).^2);
       end
   end
end

% Backprop into the nearest neighbors
NN_deltas = nodeDeltas(:,n+1:end);