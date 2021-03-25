function [allSNum, allSStr, allSTree, allSNN,allIndicies, ...
    categories, sentenceLabels] = loadData(params,type)

if isempty(type)
    load(params.paths.dataFile);
    
    if ~exist('allSNN','var')
        disp('Getting Nearest Neighbors...');
        allSNN = getNN([],allSNum,allIndicies,params);
        
        save(params.paths.dataFile,'allSNN','-append');        
    end
    
    if ~exist('sentenceLabels','var')
        sentenceLabels = [];
    end
    
else
    if strcmpi(type,'train')
        type = 'Train';
    elseif strcmpi(type,'test')
        type = 'Test';
    else
        error(['unrecognized option: ' type]);
    end
    load([params.paths.data 'allData' type]);
end

return
