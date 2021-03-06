%%
% Input file

% Parse input with Stanford Parser
% From Stanford Parser README:
% On a Unix system you should be able to parse the English test file with the
% following command:
%    ./lexparser.sh input.txt > parsed.txt

%things edited:
% changed runReformat tree and used allSOStr for my indices, it's not what
% it used to be now.

if ~exist('inputFile','var')
    inputFile = 'parsed.txt';
end
if ~exist('outputDataFile','var')
    dataDir = '../dataCamera';
    % vocabFile = 'vocab.txt';     % one word per line, first word is UNKNOWN token
    outputDataFile = [dataDir '/allTrainData.mat'];
end





%Create hash table for the word list
% fid = fopen(vocabFile, 'r');
% fileLines = textscan(fid, '%s', 'delimiter', '\n', 'bufsize', 100000);
% fclose(fid);
% words=fileLines{1};
load('wsjBin_CW_train_prePro_maxL200_extCWVocab_delSpecChar0', 'words')


global wordMap;
wordMap = containers.Map(words,num2cell([1:length(words)]'));


allSNum = {};
allSStr = {};
allSOStr = {};
allSPOS = {};
allSTree = {};



fid = fopen(inputFile, 'r');
fileLines = textscan(fid, '%s', 'delimiter', '\n');%, 'bufsize', 100000);
fclose(fid);
fileLines=fileLines{1};


sNum=[];
c = [];
cc = [];
for i=1:length(fileLines)
    if mod(i,1000) == 0
        disp(['Line Number: ' num2str(i)]);
    end
    if (isempty(fileLines{i}))
        continue
    end
    
    if strcmp(fileLines{i}, 'SENTENCE_SKIPPED_OR_UNPARSABLE')
        allSNum{end+1} = [];
        allSStr{end+1} = [];
        allSOStr{end+1} = [];
        allSPOS{end+1} = [];
        allSTree{end+1} = [];
        c = [c length(allSNum)]
        continue
    end
    
    if strcmp(fileLines{i}, 'Sentence skipped: no PCFG fallback.')
        cc = [cc length(allSNum)]
        continue
    end
    
    line = regexp(fileLines{i},' ','split');
    if isempty(line)
        continue
    end
    %if strcmp(line{1},'((SINV') || (length(line) >= 2 && strcmp(line{1},'(') && (strcmp(line{2},'(S') || strcmp(line{2},'(FRAG')))
    % too many freakin special cases! see:  grep -hE "^\(" * | sort | uniq
    if isempty(sNum)
        sNum = [-1]; % -1 for internal nodes
        sStr = {''};
        sOStr = {''};
        
        posTag = regexp(fileLines{i}, '([A-Z]+)', 'match');
        sPOS = {posTag{1}};
        %             disp(['Starting new phrase (POS: ' sPOS{1} '). Full line is: ' fileLines{i}])
        
        if strcmp(fileLines{i}(1),'(') && strcmp(fileLines{i}(2),'(')
            line = ['(' line{1}(2:end) line(2:end)];
        end
        
        
        sTree= [0];
        lastParents = [1];
        currentParent = 1;
        if length(line)>2
            line = line(3:end);
        else
            continue;
        end
        
    end
    
    lineLength = length(line);
    s=1;
    if isstr(line)
        line={line};
    end
    
    %disp(line)
    
    while s<=lineLength
        startsBranch = strcmp(line{s}(1),'(');
        %             nextIsWord = s<lineLength && strcmp(line{s+1}(end),')');
        nextIsWord = s<lineLength && (strcmp(line{s+1}(end),')') || (~strcmp(line{s+1}(1),'(') && s<lineLength-1));
        % internal nodes
        if startsBranch && ~nextIsWord
            sTree=[sTree currentParent];
            sStr{end+1}='';
            sOStr{end+1}='';
            sPOS{end+1}=line{s}(2:end);
            sNum = [sNum -100];
            currentParent=length(sNum);
            lastParents = [lastParents currentParent];
            s=s+1;
            continue;
        end
        
        if startsBranch && nextIsWord
            numWords = 1;
            mm = regexp(line{s+numWords},'(');
            m = regexp(line{s+numWords},')');
            while length(m) <= length(mm)%isempty(m)
                word = line{s+numWords};
                % if so set indicie to size(sStr,2)+1
                thisNum = WordLookup(word);
                sStr{end+1} = [words{thisNum}];
                sOStr{end+1} = [word];
                sTree=[sTree currentParent];
                sPOS{end+1} = line{s}(2:end);
                sNum = [sNum thisNum];
                
                numWords = numWords+1;
                assert(s+numWords <= lineLength);
                m = regexp(line{s+numWords},')');
                mm = regexp(line{s+numWords},'(');
            end
            
            if ~isempty(mm)
                word = line{s+numWords}(mm+1:m-1);
            else
                word = line{s+numWords}(1:m-1);
            end
            %                 word = regexprep(word, '[0-9]', '2'); % replace all digits with 2
            
            word = lower(word);
            word = regexprep(word, '[0-9]', '2'); % replace all digits with 2
            %disp(word)
            
            
            thisNum = WordLookup(word);  %looks up word here
            
            
            sStr{end+1} = [words{thisNum}];
            sOStr{end+1} = [word];
            sTree=[sTree currentParent];
            sPOS{end+1} = line{s}(2:end);
            sNum = [sNum thisNum];
            s=s+numWords+1;
            lastParents=lastParents(1:(end-(length(m)-length(mm))+1));
            if isempty(lastParents)
                assert(length(sNum)==length(sStr));
                assert(length(sNum)==length(sPOS));
                assert(length(sNum)==length(sTree));
                %{
                disp('sNum');
                disp(sNum);
                disp('sStr');
                disp(sStr);
                disp('sOStr');
                disp(sOStr);
                disp('sPOS');
                disp(sPOS);
                disp('sTree');
                disp(sTree);
                %}
                allSNum{end+1} = sNum;
                allSStr{end+1} = sStr;
                allSOStr{end+1} = sOStr;
                allSPOS{end+1} = sPOS;
                allSTree{end+1} = sTree;
                
                s=s+1;
                
                sNum = [];
                sStr = {};
                sOStr = {};
                sPOS = {};
                sTree= [];
                
                continue
            end
            currentParent = lastParents(end);
            continue
        end
    end
end


% Below here I change allSOStr so that i may properly get indicies
% for the the paths. It is never actually used beyond here.

runReformatTree

%{
for i=1:length(allSNum)
	
                disp(i)
                disp('sNum');
                disp(allSNum{i});
                disp('sStr');
                disp(allSStr{i});
                disp('sOStr');
                disp(allSOStr{i});
                disp('sPOS');
                disp(allSPOS{i});
                disp('sKids');
                disp(allSKids{i})
                disp('sTree');
                disp(allSTree{i});
end
%}

load([dataDir '/toBeConverted.mat'])
numInd = double(numInd);
allIndicies = zeros(length(allSOStr),2);

for sentence = 1:length(allSOStr)
    posInd1 = [];
    posInd2 = [];
    for word = 1:length(allSOStr{sentence})
        
        if strcmpi(allSOStr{sentence}{word},elemInd{sentence,1})
            posInd1 = [posInd1 word];
        end
        if strcmpi(allSOStr{sentence}{word},elemInd{sentence,2})
            posInd2 = [posInd2 word];
        end
        
    end
    [dummy, ind] = min(abs(numInd(sentence,1)-posInd1));
    allIndicies(sentence,1) = posInd1(ind);
    [dummy, ind] = min(abs(numInd(sentence,2)-posInd2));
    allIndicies(sentence,2) = posInd2(ind);
    
end


save(outputDataFile,'allSNum','allSStr','allSTree','allSKids','allIndicies',...
    'sentenceLabels','categories','words','outputDataFile');

