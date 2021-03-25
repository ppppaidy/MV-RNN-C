function [df_Wv,df_Wo,df_W,df_WO] = backpropAll(thisTree,W,WO,Wo,params,deltaUp_vec,deltaUp_op,thisNode,numWordsInBatch,indicies,NN)

wsz = params.wordSize;
r = params.rankWo;

numWords = thisTree.numLeafs;

df_Wv = spalloc(wsz,numWordsInBatch,(numWords+params.NN*2+2)*wsz); % plus 2 for possible padding

df_Wo = spalloc(wsz+2*r*wsz,numWordsInBatch,(numWords+params.NN*2)*(wsz+2*r*wsz));

%%%%%%%%%%%%%%
% df_W
% add here: delta's from your pooled matrix instead of deltaDownAddScore
deltaVecDownFull = deltaUp_vec + thisTree.nodeVecDeltas(:,thisNode);

kids = thisTree.getKids(thisNode);
kidsParInLR{1} = thisTree.ParIn_a(:,kids(1));
kidsParInLR{2} = thisTree.ParIn_a(:,kids(2));

kidsAct = [kidsParInLR{1} ;kidsParInLR{2} ; 1];
df_W =  deltaVecDownFull*kidsAct';

%%%%%%%%%%%%%%
% df_W
kidsOps{1} = thisTree.nodeOp_A(:,kids(1));
kidsOps{2} = thisTree.nodeOp_A(:,kids(2));


deltaOpDownFull = deltaUp_op;
df_WO = deltaOpDownFull * [reshape(kidsOps{1},params.wordSize,params.wordSize) ; reshape(kidsOps{2},params.wordSize,params.wordSize)]';

WOxDeltaUp = WO' * deltaOpDownFull;

Wdelta_bothKids = W' * deltaVecDownFull;
Wdelta_bothKids = Wdelta_bothKids(1:2*params.wordSize);
Wdelta_bothKids= reshape(Wdelta_bothKids,params.wordSize,2);

otherKid(1) = 2;
otherKid(2) = 1;

kidsActLR{1} = thisTree.nodeAct_a(:,kids(1));
kidsActLR{2} = thisTree.nodeAct_a(:,kids(2));

% collect deltas from each children (they cross influence each other via operators)
for c = 1:2
    delta_intoMatrixVec = Wdelta_bothKids(:,c);
    deltaDown_op{otherKid(c)} = delta_intoMatrixVec*kidsActLR{c}';
    otherChildOp = reshape(kidsOps{otherKid(c)},params.wordSize,params.wordSize);
    if thisTree.isLeafVec(kids(c))
        deltaDown_vec{c} = otherChildOp' * delta_intoMatrixVec;
    else
        deltaDown_vec{c} = otherChildOp' * delta_intoMatrixVec .* (1 - kidsActLR{c}.^2);
    end
end

for c = 1:2
    if thisTree.isLeaf(kids(c))
        thisWordNum = thisTree.nodeLabels(kids(c));
        df_Wv(:,thisWordNum) = df_Wv(:,thisWordNum)+deltaDown_vec{c} + thisTree.nodeVecDeltas(:,kids(c));
        
        NN_ind = 0;
        if kids(c) == indicies(1)
            NN_ind = 1;
        elseif kids(c) == indicies(2)
            NN_ind = 2;
        end
        if NN_ind
            el_NN = NN(:,NN_ind);
            for i = 1:params.NN
                df_Wv(:,el_NN(i)) = df_Wv(:,el_NN(i)) + thisTree.NN_deltas(:,NN_ind);
            end
        end
        
        deltaDown_op{c} = deltaDown_op{c} + WOxDeltaUp((c-1)*params.wordSize+1:c*params.wordSize,:);
        
        numP = params.wordSize*params.rankWo;
        dAlpha = diag(deltaDown_op{c});
    
        wordWo_v = Wo(wsz*(1+r)+1:end,thisWordNum);
        WoASD_v = reshape(wordWo_v,wsz,r);
        dWo_u = deltaDown_op{c}*WoASD_v;
        final_dWo_u = reshape(dWo_u, numP, 1);
            
        wordWo_u = Wo(wsz+1:wsz*(1+r),thisWordNum);
        WoASD_u = reshape(wordWo_u,wsz,r);
        dWo_v = deltaDown_op{c}'*WoASD_u;
        final_dWo_v = reshape(dWo_v,numP,1);
        df_Wo(:,thisWordNum) = df_Wo(:,thisWordNum) + [dAlpha;final_dWo_u; final_dWo_v];
    else
        deltaDown_op{c} = deltaDown_op{c} + WOxDeltaUp((c-1)*params.wordSize+1:c*params.wordSize,:);
        [df_Wv_new,df_Wo_new,df_W_new,df_WO_new] = backpropAll(thisTree,W,WO,Wo,params,deltaDown_vec{c},deltaDown_op{c},kids(c),numWordsInBatch,indicies,NN);
        df_Wv = df_Wv + df_Wv_new;
        df_Wo = df_Wo + df_Wo_new;
        df_W= df_W+ df_W_new;
        df_WO = df_WO + df_WO_new;
    end
end
