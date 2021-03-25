function thisTree = forwardPropTree(sNum,sTree,sStr,sNN,indicies,Wv,Wo,W,WO,Wcat,params, onlyGetVectors)

if ~exist('onlyGetVectors','var')
    onlyGetVectors = false;
end

wsz = params.wordSize;
r = params.rankWo;

words = find(sNum>0);
numTotalNodes = length(sNum);

allV = Wv(:,sNum(words));
allO = Wo(:,sNum(words));

thisTree = tree();
% set tree structure of tree
thisTree.pp = sTree';
% set which nodes are leaf nodes
thisTree.isLeafVec = zeros(numTotalNodes,1);
thisTree.isLeafVec(words) = 1;

thisTree.nodeNames = 1:length(sTree);
thisTree.nodeLabels = sNum;

% the inputs to the parent
thisTree.ParIn_z=zeros(params.wordSize,numTotalNodes); % empty for leaf nodes
thisTree.ParIn_a=zeros(params.wordSize,numTotalNodes);
% the new operators
thisTree.nodeOp_A=zeros(params.wordSize^2,numTotalNodes);
% the scores for each decision
thisTree.score = zeros(numTotalNodes,1);
% the children of each node (for speed)
thisTree.kids = zeros(numTotalNodes,2);

% initialize the vectors and operators of the words (leaf nodes)
thisTree.nodeAct_a(:,words) = allV;

for thisWordNum = 1:length(words)
    thisTree.nodeOp_A(:,thisWordNum) = reshape(diag(allO(1:wsz,thisWordNum))+reshape(allO(wsz+1:wsz*(1+r),thisWordNum),wsz,r)*...
        reshape(allO(wsz*(1+r)+1:end,thisWordNum),wsz,r)',wsz^2,1);
end

toMerge = words;

while length(toMerge)>1
    % find unpaired bottom leaf pairs (initially words) that share parent
    i=0;
    foundGoodPair=0;
    while ~foundGoodPair
        i=i+1;
        if sTree(toMerge(i))==sTree(toMerge(i+1))
            foundGoodPair=1;
        end
    end
    newParent = sTree(toMerge(i));
    kid1 = toMerge(i);
    kid2 = toMerge(i+1);
    thisTree.kids(newParent,:) = [kid1 kid2];
    % set new parent to be possible merge candidate
    toMerge(i) = newParent;
    % delete other kid
    toMerge(i+1) = [];
    
    a = thisTree.nodeAct_a(:,kid1);
    A = reshape(thisTree.nodeOp_A(:,kid1),params.wordSize,params.wordSize);
    b = thisTree.nodeAct_a(:,kid2);
    B = reshape(thisTree.nodeOp_A(:,kid2),params.wordSize,params.wordSize);
    
    l_a = B*a;
    r_a = A*b;
    
    thisTree.nodeAct_a(:,newParent) = tanh(W * [l_a;r_a;1]);
    
    P_A = reshape( WO * [A;B] ,params.wordSize^2,1);
    
    % save all this for backprop:
    thisTree.ParIn_a(:,kid1) = l_a;
    thisTree.ParIn_a(:,kid2) = r_a;
    thisTree.nodeOp_A(:,newParent) = P_A;
end
% vecPath - contains the vectors along the node path
% nodePath - contains the nodes along the node path
thisTree.nodePath = findPath(thisTree,indicies);

% Get the internal features
thisTree.features = getInternalFeatures(thisTree.nodePath, sStr, indicies, params);

% Add only the top node
thisTree.nodePath = max(thisTree.nodePath);
thisTree.pooledVecPath = thisTree.nodeAct_a(:,thisTree.nodePath);

% Add the elements
thisTree.pooledVecPath = [thisTree.pooledVecPath thisTree.nodeAct_a(:,indicies(1)) thisTree.nodeAct_a(:,indicies(2))];
thisTree.nodePath = [thisTree.nodePath; indicies(1); indicies(2)];
thisTree.poolMatrix = eye(3*wsz);

% add outer context words
thisTree = addOuterContext(thisTree, indicies, length(sStr), params);

% add inner context words
thisTree = addInnerContext(thisTree,indicies, Wv, length(sStr), params);

% add in nearest neighbors
thisTree = addNN(thisTree, Wv, sNN, params);

catInput = [thisTree.pooledVecPath(:);thisTree.NN_vecs(:);thisTree.features(:)];

if ~onlyGetVectors
    num = exp(Wcat*catInput);
    thisTree.y = num/sum(num);
end
return


function thisTree = addOuterContext(thisTree, indicies, sentLen, params)
n = params.numOutContextWords;
wsz = params.wordSize;
wszI = eye(wsz);

% element 1
o1 = max(indicies(1)-n,1):indicies(1)-1;
o1L = length(o1);
if isempty(o1)
	o1 = indicies(1);
end
thisTree.poolMatrix = [thisTree.poolMatrix zeros(size(thisTree.poolMatrix,1),wsz); ...
    zeros(o1L*wsz,size(thisTree.poolMatrix,2)) repmat((1/o1L)*wszI,o1L,1)];
% element 2
o2 = indicies(2)+1:min(indicies(2)+n,sentLen);
o2L = length(o2);

thisTree.poolMatrix = [thisTree.poolMatrix zeros(size(thisTree.poolMatrix,1),wsz); ...
    zeros(o2L*wsz,size(thisTree.poolMatrix,2)) repmat((1/o2L)*wszI,o2L,1)];

thisTree.nodePath = [thisTree.nodePath; o1'; o2';];
thisTree.pooledVecPath = [thisTree.pooledVecPath mean(thisTree.nodeAct_a(:,o1),2) mean(thisTree.nodeAct_a(:,o2),2)];
return


function thisTree = addInnerContext(thisTree, indicies, Wv, sentLen, params)
wsz = params.wordSize;
n = params.numInContextWords;

% element1 
o1 = indicies(1)+1:min(indicies(1)+n,sentLen);

if length(o1) < n
    o1 = [o1 repmat(sentLen,1,n-length(o1))];
end
thisTree.poolMatrix = [thisTree.poolMatrix zeros(size(thisTree.poolMatrix,1),wsz*n); ...
    zeros(wsz*n,size(thisTree.poolMatrix,2)) eye(wsz*n)];

% element 2
o2 = max(indicies(2)-n,1):indicies(2)-1;
vecs = thisTree.nodeAct_a(:,o2);
o2L = length(o2);
if o2L < n % use PADDING for this case
    o2 = [repmat(0,1,n-o2L) o2];
    vecs = [repmat(Wv(:,1),1,n-o2L) vecs];
end
thisTree.poolMatrix = [thisTree.poolMatrix zeros(size(thisTree.poolMatrix,1),wsz*n); ...
    zeros(wsz*n,size(thisTree.poolMatrix,2)) eye(wsz*n)];

thisTree.nodePath = [thisTree.nodePath; o1'; o2'];
thisTree.pooledVecPath = [thisTree.pooledVecPath thisTree.nodeAct_a(:,o1) vecs];
return


function thisTree = addNN(thisTree, Wv, sNN, params)
wsz = params.wordSize;
wszI = eye(wsz);
thisTree.NN_vecs = [mean(Wv(:,sNN(1:params.NN,1)),2) mean(Wv(:,sNN(1:params.NN,2)),2)];
thisTree.poolMatrix = [thisTree.poolMatrix zeros(size(thisTree.poolMatrix,1),wsz); ...
    zeros(wsz,size(thisTree.poolMatrix,2)) (1/params.NN)*wszI];

thisTree.poolMatrix = [thisTree.poolMatrix zeros(size(thisTree.poolMatrix,1),wsz); ...
    zeros(wsz,size(thisTree.poolMatrix,2)) (1/params.NN)*wszI];
return

