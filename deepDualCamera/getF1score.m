function F1 = getF1score(predictLabel,categories,sentenceLabels,name,params)


proposedFileStr = [params.paths.results 'proposedAnswers' name '.txt']; %proposed answer
proposedFile = fopen(proposedFileStr,'w');
if proposedFile == -1
    error(['Unable to open file: ' proposedFile]);
end

if isempty(sentenceLabels)
    disp('No labels, will just output predictions')
    for j = 1:length(predictLabel)
        fprintf(proposedFile, [num2str(j) '\t' categories{predictLabel(j)} '\n']);
    end
    fclose(proposedFile);
    F1 = [];
    return;
end

answerFileStr = [params.paths.results 'answerKey' name '.txt'];
answerFile = fopen(answerFileStr,'w');
outFileStr = [params.paths.results 'results' name '.txt']; % results

for j = 1:length(predictLabel)
    fprintf(proposedFile, [num2str(j) '\t' categories{predictLabel(j)} '\n']);
    fprintf(answerFile, [num2str(j) '\t' categories{sentenceLabels(j)} '\n']);
end
fclose(proposedFile);
fclose(answerFile);


if isunix
    cmdString = ['./semeval2010_task8_scorer-v1.2.pl ' proposedFileStr ' ' answerFileStr  ' > ' outFileStr];
    system(cmdString);
    evalBString = textread(outFileStr,'%s','delimiter', '\n');
    I = strmatch('<<< The official score',evalBString);
    EqPos = strfind(evalBString{I},'=');
    perPos = strfind(evalBString{I},'%');
    F1 = str2double(evalBString{I}(EqPos+ 2:perPos - 1));
end


return