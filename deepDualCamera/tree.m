classdef tree
    
    properties
        % parent pointers
        pp = [];
        nodeNames;
        nodeFeatures;
        nodeOperators;

        nodeAct_z=[];
        nodeAct_a=[];
        % the inputs to the parent 
        ParIn_z=[]; % empty for leaf nodes
        ParIn_a=[];

        nodeOp_Z=[];
        nodeOp_A=[];
        
        nodeContextL = [];
        nodeContextR = [];
        nodeContextNumL = [];
        nodeContextNumR = [];
        % the parent pointers do not save which is the left and right child of each node, hence:
        % numNodes x 2 matrix of kids, [0 0] for leaf nodes
        kids = [];
        % matrix (maybe sparse) with L x S, L = number of unique labels, S= number of segments
        % ground truth:
        nodeLabels=[];
        % categories: computed activations (not softmaxed)
        catAct = [];
        catOut = [];
        % computed category
        nodeCat = [];
        pos = {};
        % if we have the ground truth, this vector tells us which leaf labels were correctly classified
        nodeCatsRight=0;
        isLeafVec = [];
        
        score=0;
        
        %below here was added by Brody
        nodePath = [];
        poolMatrix = [];
        y = [];
        
        nodeVecDeltas = [];
        
        pooledVecPath = [];
        features = [];
        NN_vecs = [];
        NN_deltas = [];

    end
    
    
    methods
        function num = numLeafs(obj)
            num = (length(obj.pp)+1)/2;
        end
        
        function id = getTopNode(obj)
            id = find(obj.pp==0);
        end
        
        function kids = getKids(obj,node)
            %kids = find(obj.pp==node);
            kids = obj.kids(node,:);
        end
        
        function p = getParent(obj,node)
            %kids = find(obj.pp==node);
            if node>0
                p = obj.pp(node);
            else
                p=-1;
            end
        end
        
        %TODO: maybe compute leaf-node-ness once and then just check for it
        function l = isLeaf(obj,node)
            l = ~any(obj.pp==node);
        end
        
        
        
        function plotTree(obj,se)
            %TREEPLOT Plot picture of tree.
            %   TREEPLOT(p) plots a picture of a tree given a row vector of
            %   parent pointers, with p(i) == 0 for a root and labels on each node.
            %
            %   Example:
            %      myTreeplot([2 4 2 0 6 4 6],{'i' 'like' 'labels' 'on' 'pretty' 'trees' '.'})
            %   returns a binary tree with labels.
            %
            %   Copyright 1984-2004 The MathWorks, Inc.
            %   $Revision: 5.12.4.2 $  $Date: 2004/06/25 18:52:28 $
            %   Modified by Richard @ Socher . org to display text labels
            
            p = obj.pp';
            %!! This 1:length(p) does not work if order is encoded in the order of the parent vector
            %[x,y,h]=treelayout(p,1:length(p));
            [x,y,h]=treelayout(p);
            f = find(p~=0);
            pp = p(f);
            X = [x(f); x(pp); NaN(size(f))];
            Y = [y(f); y(pp); NaN(size(f))];
            X = X(:);
            Y = Y(:);
            
            n = length(p);
            if n < 500,
                plot (x, y, 'wo', X, Y, 'b-');
            else
                plot (X, Y, 'r-');
            end;
            xlabel(['height = ' int2str(h)]);
            axis([0 1 0 1]);
            
            if ~isempty(obj.nodeNames)
                for l=1:min(length(obj.nodeNames),length(x))
                    if isnumeric(obj.nodeNames(l))
                        text(x(l),y(l),num2str(obj.nodeNames(l)),'Interpreter','none',...
                            'HorizontalAlignment','center','FontSize',8,'BackgroundColor',[1 1 .6])
                    else
                        if isLeaf(obj,l)
                        text(x(l),y(l),obj.nodeNames{l},'Interpreter','none',...
                            'HorizontalAlignment','center','FontSize',8,'BackgroundColor',[1 1 .6])
                        end
                    end
                    %                     if ~isempty(obj.nodeLabels)
                    %                         if iscell(obj.nodeNames)
                    %                             text(x(l),y(l),[obj.nodeLabels{l} '(' obj.nodeLabels{l} ')'],'Interpreter','none',...
                    %                                 'HorizontalAlignment','center','FontSize',8,'BackgroundColor',[1 1 .6])
                    %                         else
                    %                             % for numbers
                    %                             if isnumeric(obj.nodeLabels(l))
                    % %                                 if isinteger(obj.nodeLabels(l))
                    %                                     allL = obj.nodeLabels(:,l);
                    %                                     allL = find(allL);
                    %                                     if isempty(allL)
                    %                                         text(x(l),y(l),[num2str(obj.nodeNames(l))],'Interpreter','none',...
                    %                                         'HorizontalAlignment','center','FontSize',8,'BackgroundColor',[1 1 .6])
                    %                                     else
                    %                                         text(x(l),y(l),[num2str(obj.nodeNames(l)) ' (' mat2str(allL) ')'],'Interpreter','none',...
                    %                                             'HorizontalAlignment','center','FontSize',8,'BackgroundColor',[1 1 .6])
                    %                                     end
                    %
                    % %                                 else
                    % %                                     text(x(l),y(l),[obj.nodeLabels(l) ' ' num2str(obj.nodeLabels(l),'%.1f') ],'Interpreter','none',...
                    % %                                         'HorizontalAlignment','center','FontSize',8,'BackgroundColor',[1 1 .6])
                    % %                                 end
                    %                                 % change to font size 6 for nicer tree prints
                    %                             else
                    %                                 text(x(l),y(l),[obj.nodeNames{l}],'Interpreter','none',...
                    %                                     'HorizontalAlignment','center','FontSize',8,'BackgroundColor',[1 1 .6])
                    %                             end
                    %                         end
                    %                     end
                end
            end
            
            
        end
        
        
        function treeString= toPTBString(obj)
            %function treeString = treeToString(tTree,tStr,tPOS)
            
            currentParent = find(obj.pp==0);
            if isempty(currentParent)
                
                if nargin ==4
                    % has tPOS
                    error('not implemented')
                    %treeString = ['(' tPOS{currentParent} ' ' getSubTree(kids(1),tTree,tStr,tPOS) ' ' getSubTree(kids(2),tTree,tStr,tPOS) ')'];
                else
                    treeString = ['(NO ' obj.nodeNames{1} ')'];
                end                
                
            else
            if isempty(obj.pos)
                treeString = getSubTree(currentParent,obj.pp,obj.nodeNames);
            else
                treeString = getSubTree(currentParent,tTree,tStr,tPOS);                
            end
            end
            
        function subString = getSubTree(currentParent,tTree,tStr)
            %kids = find(tTree==currentParent);
            if currentParent>0
                kids = obj.kids(currentParent,:);
                if all(kids==0)
                    kids =[];
                end
            else
                kids =[];
            end
            if length(kids)==2
                if nargin ==4
                    % has tPOS
                    error('not implemented')
                    %subString = ['(' tPOS{currentParent} ' ' getSubTree(kids(1),tTree,tStr,tPOS) ' ' getSubTree(kids(2),tTree,tStr,tPOS) ')'];
                else
                    subString = ['(NO ' getSubTree(kids(1),tTree,tStr)  ' ' getSubTree(kids(2),tTree,tStr) ')'];
                end
            elseif isempty(kids)
                if sum(strcmp(tStr{currentParent},{'.',',','"',':','``','`','''',''''''}))
                    subString = ['(' tStr{currentParent} ' '  tStr{currentParent} ')'];
                elseif sum(strcmp(tStr{currentParent},{'--','-','...',';'}))
                    subString = ['(: '  tStr{currentParent} ')'];
                elseif sum(strcmp(tStr{currentParent},{'?','!'}))
                    subString = ['(. '  tStr{currentParent} ')'];
                else
                    if nargin==4
                        subString = ['(' tStr{currentParent} ' ' tStr{currentParent} ')']; %changed tPOS to tStr by ppppaidy
                    else
                        subString = ['(NO ' tStr{currentParent} ')'];
                    end
                end
            else
                error('wtf')
            end
        end
            
            
        end
        
        
    end
end
