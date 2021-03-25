function [allSHypVecs,allSNERVecs,allSPOSVecs] = processTags(allSStr, allIndicies, params)

addpath(genpath('./tools'))

[allSHyp,allSNER,allSPOS] = processTagFile(allSStr, params);

allSHypVecs = getHyp(allSHyp,allIndicies,params);

allSNERVecs = getNER(allSNER,allIndicies,params);

allSPOSVecs = getPOS(allSPOS,allIndicies,params);

return


function [allSHyp,allSNER,allSPOS] = processTagFile(allSStr, params)
% params just needs params.paths.tempData set to get tag file
% allSStr is needed to align the sentences

if ~exist('fileLines','var')
    fileLines = readTextFile([params.paths.tempData '/rawSentences.txt.tags']);
end

sentenceCount=1;
li=1;

%%% all sentences
allSStrAlt = {};
allSStrLem = {};
allSPOS = {};
allSNER = {};
allSHyp = {};


while li<=length(fileLines)
    sentStr={};
    sentLem={};
    sentPOS={};
    sentNER={};
    sentHyp={};
    
    if mod(li,10e3) == 0
        disp(['line: ' num2str(li)]);
    end
    
    [dummy, dummy, dummy, dummy, dummy, dummy, splitLine] = regexp(fileLines{li}, ' ');
    splitLine = splitLine(1:end-5);
    assert(mod(length(splitLine),5)==0)
    numWords = length(splitLine)/5;
    for i = 0:(numWords-1)
        sentStr{end+1} = splitLine{5*i+ 1};
        sentLem{end+1} = splitLine{5*i+ 3};
        sentPOS{end+1} = splitLine{5*i+ 2};
        sentNER{end+1} = splitLine{5*i+ 5};
        sentHyp{end+1} = splitLine{5*i+ 4};
    end
    
    
    
    allSStrAlt{end+1} = sentStr;
    allSStrLem{end+1} = sentLem;
    allSPOS{end+1} = sentPOS;
    allSNER{end+1} = sentNER;
    allSHyp{end+1} = sentHyp;
    
    sentenceCount=sentenceCount+1;
    
    li=li+1;
end

% 2717
% sentenceCount
% save(['../data/SemEval2010_task8_all_data/allHypNER_notAligned_' tt '.mat'],'allSStrAlt','allSStrLem','allSPOS','allSNER','allSHyp')

% load(['../data/SemEval2010_task8_all_data/outputConverter' tt '.mat'],'allSStr')


diffCount =0;
oriLarger = 0;
for i=1:length(allSStr)
    if((length(allSStr{i})-1)~=length(allSStrAlt{i}))
        
        sentStrNew = {};
        sentLemOri = {};
        sentPOSOri = {};
        sentNEROri = {};
        sentHypOri = {};
        
        % different tokenization
        % we want to be more like this guy:
        sentOri = allSStr{i};
        if strcmp(allSStr{i}(end),'.')
            sentOri = sentOri(1:end-1);
        end
        % so we change all these guys:
        sentStr = lower(allSStrAlt{i});
        for s = 1:length(sentStr)
            sentStr{s} = regexprep(sentStr{s},'[0-9]','2');
        end
        sentLem = allSStrLem{i};
        sentPOS = allSPOS{i};
        sentNER = allSNER{i};
        sentHyp = allSHyp{i};
        
        for l = length(sentOri):-1:1
            word = sentOri{l};
            [a b] = find(strcmp(word,sentStr));
            if length(b)==1
                closestInd=b;
                %simple perfect match
            elseif length(b)>1
                % multiple matches
                [dummy,ind]=min(abs(b-l));
                closestInd = b(ind);
            elseif strcmp(word,'UNK')
                %                 if length(sentStr)<l-3
                %                     cand = min(length(sentStr),l+3):(min(1,l-3));
                %                 else
                cand = (max(1,l-3):min(length(sentStr),l+3));
                %                 end
                assert(~isempty(cand))
                closestInd = 0;
                for q= cand
                    % those UNKs of have a dash
                    a = strfind(sentStr{q},'-');
                    if ~isempty(a)
                        closestInd = q;
                    end
                end
                % still not found, let's just take the longest word in this are
                if ~closestInd
                    maxL = 0;
                    for q= cand
                        % those UNKs of have a dash
                        a = length(sentStr{q});
                        if a>maxL
                            maxL=a;
                            closestInd = q;
                        end
                    end
                end
                assert(closestInd>0)
            else
                %find a match where this is a substring, e.g. % or 22
                cand = (max(1,l-5):min(length(sentStr),l+5));
                closestInd = 0;
                for q= cand
                    a = strfind(sentStr{q},word);
                    if ~isempty(a)
                        closestInd = q;
                    end
                end
                
                if ~closestInd
                    % let's just take that exact index
                    closestInd=min(l,length(sentStr));
                end
            end
            sentStrNew{l} = sentStr{closestInd};
            sentLemOri{l} = sentLem{closestInd};
            sentPOSOri{l} = sentPOS{closestInd};
            sentNEROri{l} = sentNER{closestInd};
            sentHypOri{l} = sentHyp{closestInd};
        end
%         sentOri
%         sentStrNew
        diffCount = diffCount +1;
        
        allSStrAlt{i} = sentStrNew;
        allSStrLem{i} = sentLemOri;
        allSPOS{i} = sentPOSOri;
        allSNER{i} = sentNEROri;
        allSHyp{i} = sentHypOri;
        
    end
    
end
return


function allHypVecs = getHyp(allSHyp, allIndicies, params)

load([params.paths.data 'externalFeaturesDictionaries.mat'],'hyp2Ind');

if ~exist('hyp2Ind','var')
    error('Need this dictionary, run the line below on training data to generate it');
    hyp2Ind = getHypDict(allSHyp);
end

allHypVecs = getHypVecIndicies(allSHyp,allIndicies,hyp2Ind);

return

function hyp2Ind = getHypDict(allSHyp)

% better faster way, but not compatible with old code
% counter = containers.Map();
% for s = 1:length(allSHyp)
%     if mod(s,500) == 0
%         display(num2str(s))
%     end
%     for j = 1:length(allSHyp{s})
%         hyp = allSHyp{s}{j};
%         if counter.isKey(hyp)
%             counter(hyp) = counter(hyp) + 1;
%         else
%             counter(hyp) = 1;
%         end
%     end
% end
% 
% hyps = counter.keys();
% hyp2Ind = containers.Map(hyps(1,:),num2cell(1:size(hyps,2)));

hyps = {};
for s = 1:length(allSHyp)
    for ind = 1:length(allSHyp{s})
        if isempty(hyps)
            hyps(2,1) = {1};
            hyps{1,1} = allSHyp{s}{ind};
        else
            
            foundInd = find(ismember(hyps(1,:), allSHyp{s}(ind))==1);
            if ~isempty(foundInd)
                hyps{2,foundInd} = hyps{2, foundInd} + 1;
            else
                e = size(hyps,2);
                hyps(2,e+1) = {1};
                hyps{1,e+1} = allSHyp{s}{ind};
            end
        end
    end
end
hyps = hyps(1,:);
hyp2Ind = containers.Map(hyps(1,:),num2cell(1:size(hyps,2)));

return

function allHypVecs = getHypVecIndicies(allSHyp,allIndicies,hyp2Ind)

numFeats = hyp2Ind.Count;
I = eye(numFeats);
z = zeros(numFeats,1);

for s = 1:length(allIndicies)
    e1 = allIndicies(s,1);
    e2 = allIndicies(s,2);
    bag = z;
    midNodes = e1+1:e2-1;
    % for middle context words
    for mid = midNodes
        for level = 1:size(allSHyp{s},1)
            h = allSHyp{s}{mid};
            if hyp2Ind.isKey(h)
                bag = bag | I(:,hyp2Ind(h));
            end
        end
    end
    
    e1Hyps = z;
    e2Hyps = z;
    for level = 1:size(allSHyp{s})
        h1 = allSHyp{s}{level,e1};
        h2 = allSHyp{s}{level,e2};
        if hyp2Ind.isKey(h1)
            e1Hyps = e1Hyps | I(:,hyp2Ind(h1));
        end
        if hyp2Ind.isKey(h2)
            e2Hyps = e2Hyps | I(:,hyp2Ind(h2));
        end
    end
    
    
    allHypVecs{s} = [e1Hyps e2Hyps bag];
end
return



function allNERVecs = getNER(allSNER,allIndicies,params)

load([params.paths.data 'externalFeaturesDictionaries.mat'],'ner2Ind');

if ~exist('ner2Ind','var')
    error('Need this dictionary, run the line below on training data to generate it');
    ner2Ind = getNERDictionary(allSNER);
end

allNERVecs = getNERVecIndicies(allSNER,allIndicies,ner2Ind);

return

function allNERVecs = getNERVecIndicies(allSNER,allIndicies,ner2Ind)

% process tags
allSNER2 = cell(size(allSNER));
for s = 1:length(allSNER)
    for w = 1:length(allSNER{s})
        word = allSNER{s}{w};
        ind = strfind(word,':');
        if ~isempty(ind)
            if length(ind) == 1
                allSNER2{s}{w} = word(ind+1:end);
            else
                allSNER2{s}{w} = word(ind(1)+1:ind(2)-1);
            end
        else
            allSNER2{s}{w} = word;
        end
    end
end

numFeats = ner2Ind.Count;


allNERVecs = cell(size(allSNER2));
z = zeros(numFeats,1);
I = eye(numFeats);

for s = 1:length(allIndicies)
    e1 = allIndicies(s,1);
    e2 = allIndicies(s,2);
    h1 =  allSNER2{s}{e1};
    if ner2Ind.isKey(h1)
        allNERVecs{s} = I(:,ner2Ind(h1));
    else
        allNERVecs{s} = z;
    end
    
    h2 = allSNER2{s}{e2};
    if ner2Ind.isKey(h2)
        allNERVecs{s} = [allNERVecs{s} I(:,ner2Ind(h2))];
    else
        allNERVecs{s} = [allNERVecs{s} z];
    end
    
    bag = z;
    
    for mid = e1+1:e2-1;
        htemp = allSNER2{s}{mid};
        if ner2Ind.isKey(htemp)
            bag = bag | I(:,ner2Ind(htemp));
        end
    end
    
    allNERVecs{s} = [allNERVecs{s} bag];
end


for s = 1:length(allNERVecs)
    allNERVecs{s} = allNERVecs{s}(2:end,:);
end
return

function ner2Ind = getNERDictionary(allSNER)


ners = {};
counts = [];
for s = 1:length(allSNER)
    for ind = 1:length(allSNER{s})
        if isempty(ners)
            ners(2,1) = {1};
            ners{1,1} = allSNER{s}{ind};
        else
            
            foundInd = find(ismember(ners(1,:), allSNER{s}(ind))==1);
            if ~isempty(foundInd)
                ners{2,foundInd} = ners{2, foundInd} + 1;
            else
                e = size(ners,2);
                ners(2,e+1) = {1};
                ners{1,e+1} = allSNER{s}{ind};
            end
        end
    end
end





allNERVecs = cell(size(allSNER));


% get rid of first and second semi-colon
ners2 = cell(1,length(ners));
for n = 1:length(ners)
    ind = strfind(ners{1,n},':');
    if ~isempty(ind)
        if length(ind) == 1
            ners2{n} = ners{1,n}(ind+1:end);
        else
            assert(length(ind) == 2)
            ners2{n} = ners{1,n}(ind(1)+1:ind(2)-1);
        end
    else
        ners2{n} = ners{1,n};
    end
end

ners2 = unique(ners2);

ner2Ind = containers.Map(ners2(1,:),num2cell(1:size(ners2,2)));
return



function allPOSVecs = getPOS(allSPOS,allIndicies,params)

load([params.paths.data 'externalFeaturesDictionaries.mat'],'pos2Ind');

if ~exist('pos2Ind','var')
    error('Need this dictionary, run the line below on training data to generate it');
    pos2Ind = getPOSDictionary(allSPOS,allIndicies);
end

allPOSVecs = getPOSVecIndicies(allSPOS,allIndicies,pos2Ind);

return

function pos2Ind = getPOSDictionary(allSPOS,allIndicies)


allSPOS2 = cell(size(allSPOS));

for s = 1:length(allSPOS)
    for w = 1:length(allSPOS{s})
        word = allSPOS{s}{w};
        
        allSPOS2{s}{w} = word(1);
    end
end

allSPOS = allSPOS2;


allSPOSStr = cell(1,length(allSPOS));
pos = {};
pos2 = {};
avgLength = 0;
for s = 1:length(allSPOS)
    str = '';
    e1 = allIndicies(s,1);
    e2 = allIndicies(s,2);
    mid = e1+1:e2-1;
    avgLength = avgLength + length(mid);
    for w = mid;
        if w ~= mid(1)
            pos2 = unique([pos2 [allSPOS{s}{w-1} '_' allSPOS{s}{w}]]);
        end
        str = [str '_' allSPOS{s}{w}];
    end
    
    str = str(2:end);
    allSPOSStr{s} = str;
    pos = unique([pos str]);
end


pos2Ind = containers.Map(pos2(1,:),num2cell(1:size(pos2,2)));
ind2Pos = containers.Map(num2cell(1:size(pos2,2)),pos2(1,:));


allCounts = zeros(length(pos2));
for s = 1:length(allSPOS)
    
    str = '';
    e1 = allIndicies(s,1);
    e2 = allIndicies(s,2);
    mid = e1+1:e2-1;
    for w = mid;
        if w ~=mid(1)
            str = [allSPOS{s}{w-1} '_' allSPOS{s}{w}];
            ind = pos2Ind(str);
            allCounts(ind) = allCounts(ind) + 1;
        end
    end
end


featsToUse = {};
for f = 1:length(allCounts)
    if allCounts(f) > 10
        featsToUse = [featsToUse ind2Pos(f)];
    end
end
    
pos2Ind = containers.Map(featsToUse(1,:),num2cell(1:size(featsToUse,2)));
% ind2Feat = containers.Map(num2cell(1:size(featsToUse,2)),featsToUse(1,:));


return

function allPOSVecs = getPOSVecIndicies(allSPOS,allIndicies,pos2Ind)

numFeats = pos2Ind.Count;
I = eye(numFeats);
z = zeros(numFeats,1);


allSPOS2 = cell(size(allSPOS));

for s = 1:length(allSPOS)
    for w = 1:length(allSPOS{s})
        word = allSPOS{s}{w};
        
        allSPOS2{s}{w} = word(1);
    end
end

allSPOS = allSPOS2;


allPOSVecs = cell(1,length(allIndicies));

for s = 1:length(allIndicies)
    e1 = allIndicies(s,1);
    e2 = allIndicies(s,2);
    first = z;
    bag = z;
    % for middle context words
    for mid = e1+2:e2-1;
        feat = [allSPOS{s}{mid-1} '_' allSPOS{s}{mid}];
        if pos2Ind.isKey(feat)
            if mid == (e1+2)
                first = first | I(:,pos2Ind(feat));
            else
                bag = bag | I(:,pos2Ind(feat));
            end
        end
    end
    
    allPOSVecs{s} = [first bag];
    
end

return