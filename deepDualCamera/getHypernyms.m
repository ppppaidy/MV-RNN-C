function [allHypVecs hyps] = getHypernyms(allSHyp,allIndicies)

% if strcmpi(type,'train'), type='Train';
% elseif strcmpi(type,'test'), type='Test';
% end

% load('../data/brody/allHypNER_aligned_train.mat','allSHyp');
load([params.paths.data 'externalFeaturesDictionaries.mat'],'hyps');

if ~exist('hyps','var')
    error('Need this dictionary, run the two lines below on training data to generate it');
    counter = getHypDict(allSHyp);
    hyps = counter.keys();
end

allHypVecs = getHypVecIndicies(allSHyp,allIndicies,hyps);

return

function counter = getHypDict(allSHyp)


counter = containers.Map();
for s = 1:length(allSHyp)
    if mod(s,500) == 0
        display(num2str(s))
    end
    for j = 1:length(allSHyp{s})
        hyp = allSHyp{s}{j};
        if counter.isKey(hyp)
            counter(hyp) = counter(hyp) + 1;
        else
            counter(hyp) = 1;
        end
    end
end

return

function allHypVecs = getHypVecIndicies(allSHyp,allIndicies,hyps)

hyp2Ind = containers.Map(hyps(1,:),num2cell(1:size(hyps,2)));
I = eye(size(hyps,2));
z = zeros(size(hyps,2),1);

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