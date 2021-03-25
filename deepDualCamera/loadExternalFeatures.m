function [allHypVecs allNERVecs allPOSVecs] = loadExternalFeatures(params,type)

if isempty(type) % not from SemEval8 data
    warning off all
    load(params.paths.dataFile,'external');
    warning on all
            
    if ~exist('external','var')
        disp('Getting external features...')
        load(params.paths.dataFile,'allSStr','allIndicies');
        
       [external.allHypVecs,external.allNERVecs,external.allPOSVecs] = ...
           processTags(allSStr, allIndicies, params);
       
       save(params.paths.dataFile,'external','-append');
       disp('Done');
    end
else 
    
    if strcmpi(type,'train')
        type = 'Train';
    elseif strcmpi(type,'test')
        type = 'Test';
    else
        error(['unrecognized option: ' type]);
    end
    load([params.paths.data 'allData' type],'external');
end

allHypVecs = external.allHypVecs;
allNERVecs = external.allNERVecs;
allPOSVecs = external.allPOSVecs;

return
