#!/bin/bash
# Runs the programs nessessary to convert a raw setence into a form suitable for testing the deepDual model

#./classifyRelations.sh  <inputfile> <outputfile> 
# outputfile will be the matlab format for the data. If you'd like to see the results and class labels, go to the params.paths.results directory set withing initParams.m



pathToConvertStanfordParserTrees=`pwd`/convertStanfordParserTrees
pathToStanfordParser=`pwd`/convertStanfordParserTrees/stanford-parser-2011-09-14
#pathToSST=/user/socherr/scr/projects/toolbox/nlp/wordnetSenseTagger-sst-light-0.4 #`pwd`/sst-light-0.4
pathToSST=`pwd`/wordnetSenseTagger-sst-light-0.4
pathToDeepDual=`pwd`/../deepDualCamera
pathToWeightsWOEF=`pwd`/weights_WOEF.mat
pathToWeightsWEF=`pwd`/weights_WEF.mat
pathToMyRNNC=`pwd`/../myRNNC
###################
dataDir=`pwd`/tempData
resultsDir=`pwd`/results
outputFile=`pwd`/"$2"


if [ ! $# -ge 2 ]; then
    echo 'Usage: ./convertSentences.sh <inputfile> <outputfile>'
    echo
    exit
fi

if [ ! -e "$dataDir" ]; then
    echo "Making dir: $dataDir"
    mkdir "$dataDir"
fi

if [ ! -e "$resultsDir" ]; then
    echo "Making dir: $resultsDir"
    mkdir "$resultsDir"
fi


# get raw sentences as well as well sentence labels
python processSentences.py "$1" "$dataDir"

if [ "$?" -ne 0 ]; then echo "python script failed"; exit 1; fi

# Run stanford parser for tree structures
echo
echo "Running Stanford Parser..."
java -mx600m -cp "$pathToStanfordParser/stanford-parser.jar:" edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat "penn" -sentences newline $pathToStanfordParser/grammar/englishPCFG.ser.gz  "$dataDir/rawSentences.txt" > "$dataDir/parsed.txt" &

if [ "$?" -ne 0 ]; then echo "stanford parser failed"; exit 1; fi


# Run sst for external features
echo "Running sst-light"
cd "$pathToSST"
./sst multitag-line "$dataDir/rawSentences.txt" 0 0 DATA/GAZ/gazlistall_minussemcor MODELS/WSJPOSc_base_20 DATA/WSJPOSc.TAGSET MODELS/SEM07_base_12 DATA/WNSS_07.TAGSET MODELS/WSJc_base_20 DATA/WSJ.TAGSET > "$dataDir/rawSentences.txt.tags" &
cd -

wait

cd "$pathToMyRNNC"
if [ ! -e "tempdata" ]; then
    echo "Making dir: tempdata"
    mkdir tempdata
fi
if [ ! -e "results" ]; then
    echo "Making dir: results"
    mkdir results
fi
gcc -c test_without_external_features.c
gcc -c loaddata.c
gcc -c forwardPropTree.c
gcc -c matrix.c
gcc -c findPath.c
gcc -c getInternalFeatures.c
gcc -c getF1score.c
gcc -o test_without_external_features test_without_external_features.o loaddata.o forwardPropTree.o matrix.o findPath.o getInternalFeatures.o getF1score.o -lm
cd -

# run the convertStanfordParserTrees for the final output
echo 
echo "Running convertStanfordParserTrees.m & testing script..."

matlab  -nodisplay <<EOF
cd $pathToConvertStanfordParserTrees;
dataDir='$dataDir';
inputFile='$dataDir/parsed.txt';
outputDataFile='$outputFile';
convertStanfordParserTrees;

% Now move onto testing the data
clear;
cd $pathToDeepDual;
load('$pathToWeightsWOEF','Wv','Wo','W','WO','Wcat','params');
params.paths.dataFile='$outputFile';
params.paths.results='$pathToMyRNNC/results/';
disp('Change .mat to what C can read')
disp('This may take some time')
outputCsv(['$pathToMyRNNC/tempdata/WvWOEF.txt'],Wv,5);
outputCsv(['$pathToMyRNNC/tempdata/WoWOEF.txt'],Wo,5);
outputCsv(['$pathToMyRNNC/tempdata/WWOEF.txt'],W,5);
outputCsv(['$pathToMyRNNC/tempdata/WOWOEF.txt'],WO,5);
outputCsv(['$pathToMyRNNC/tempdata/WcatWOEF.txt'],Wcat,5);
disp('Done!')
fid = fopen('$pathToMyRNNC/tempdata/paramsWOEF.txt','w');
fprintf(fid,'%f\n%f\n%f\n',params.regC,params.regC_Wcat,params.regC_WcatFeat);
fprintf(fid,'%f\n%f\n%d\n',params.regC_Wv,params.regC_Wo,params.wordSize);
fprintf(fid,'%d\n%d\n%d\n',params.rankWo,params.NN,params.numInContextWords);
fprintf(fid,'%d\n%d\n%d\n',params.numOutContextWords,params.tinyDataSet,params.categories);
fprintf(fid,'%d\n%d\n',params.test_without_external_features,params.test_with_external_features);
fprintf(fid,'%s\n%s\n',params.paths.data,params.paths.outputFolder);
fprintf(fid,'%s\n%s\n',params.paths.results,params.paths.dataFile);
for i=1:5
	fprintf(fid,'%f ',params.features.mean(i));
end
fprintf(fid,'\n');
for i=1:5
	fprintf(fid,'%f ',params.features.std(i));
end
fprintf(fid,'\n');
fclose(fid);
cd $pathToDeepDual;
[allSNum, allSStr, allSTree, allSNN,allIndicies, ...
    categories, sentenceLabels] = loadData(params,[]);
outputCsv(['$pathToMyRNNC/tempdata/allSNumWOEF.txt'],allSNum,0);
outputCsv(['$pathToMyRNNC/tempdata/allSStrWOEF.txt'],allSStr,1);
outputCsv(['$pathToMyRNNC/tempdata/allSTreeWOEF.txt'],allSTree,0);
outputCsv(['$pathToMyRNNC/tempdata/allSNNWOEF.txt'],allSNN,2);
outputCsv(['$pathToMyRNNC/tempdata/allIndiciesWOEF.txt'],allIndicies,3);
outputCsv(['$pathToMyRNNC/tempdata/categoriesWOEF.txt'],categories,4);
outputCsv(['$pathToMyRNNC/tempdata/sentenceLabelsWOEF.txt'],sentenceLabels,3);
cd $pathToMyRNNC;
cmdString = ['./test_without_external_features'];
system(cmdString);
%test_without_external_features(Wv,Wo,W,WO,Wcat,params);

% now test with external features
load('$pathToWeightsWEF','Wv','Wo','W','WO','Wcat','params');
params.paths.tempData='$dataDir'; % need to get raw sentences from sst tagger
params.paths.dataFile='$outputFile';
params.paths.results='$resultsDir/';
%test_with_external_features(Wv,Wo,W,WO,Wcat,params);
EOF

