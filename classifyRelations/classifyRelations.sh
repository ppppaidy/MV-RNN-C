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
#pathToMyRNNC=`pwd`/../myRNNC
###################
dataDir=`pwd`/tempData
resultsDir=`pwd`/results/
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
params.paths.results='$resultsDir';
test_without_external_features(Wv,Wo,W,WO,Wcat,params);

% now test with external features
load('$pathToWeightsWEF','Wv','Wo','W','WO','Wcat','params');
params.paths.tempData='$dataDir'; % need to get raw sentences from sst tagger
params.paths.dataFile='$outputFile';
params.paths.results='$resultsDir';
%test_with_external_features(Wv,Wo,W,WO,Wcat,params);
EOF

