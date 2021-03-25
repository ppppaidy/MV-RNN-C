import scipy.io
import numpy as np
import os
import sys

def formatLine(line):
    # This will take in a raw input sentence and return [[element strings], [indicies of elements], [sentence with elements removed]]
    words = line.split(' ')
    
    e1detagged = []
    e2detagged = []
    rebuiltLine = ''
    count = 1
    for word in words:
        # if tagged at all
        if word.find('<e') != -1 or word.find('</e') != -1:
            # e1 or e2
            if word[2] == '1' or word[word.find('>')-1] == '1':
                # remove tags from word for 
                e1detagged = getWord(words,word)
                e1detagged.append(count)
                # replace and tac back on . at end if needed
                word = replaceWord(word)
            else:
                e2detagged = getWord(words,word)
                e2detagged.append(count)
                word = replaceWord(word)
        rebuiltLine += ' ' + word
        firstChar = word[0]
        lastChar = word[len(word)-1]
        if firstChar == '(' or lastChar == ')' or lastChar == ',':
            count += 1
        count += 1
    rebuiltLine = rebuiltLine[1:len(rebuiltLine)]
    rebuiltLine += '\n'
    return [[e1detagged[0], e2detagged[0]],[e1detagged[1],e2detagged[1]],[e1detagged[2],e2detagged[2]],rebuiltLine]


def getWord(words, word):
    if endTwoWords(word):
        return [replaceWord(word, False), 1]
    else:
        return [replaceWord(word, False), 0]
    

def replaceWord(word, shouldEndSentence = True):
    wordList = word.split('</')
    endSentence = ''
    if len(wordList) == 2 and len(wordList[len(wordList)-1]) != 3:
        end = wordList[len(wordList)-1]
        endSentence += end[end.find('>')+1:len(end)]
    wordList = wordList[0].split('>')
    newWord = wordList[len(wordList)-1]
    if shouldEndSentence:
        newWord += endSentence
    return newWord
    

# if this has a two words ex. <e2>fast cars</e2>
def endTwoWords(word):
    return word.find('<e') == -1

def createCatDic(catNames):
    catDic = {}
    label = 0;
    for cat in catNames:
        catDic[catNames[label]]=label+1
        label += 1
    return catDic


def saveData(elemStrings,elemInd,sentenceLabels,catNames,dataDir):
    elemStringsCell = np.zeros((len(elemStrings),2), dtype=np.object)
    i=0
    for pair in elemStrings:
        elemStringsCell[i] = pair
        i += 1
    catCell = np.zeros((1,len(catNames)), dtype=np.object)
    i = 0
    for name in catNames:
        catCell[0,i] = name
        i += 1
    data = {}
    data['elemInd'] = elemStringsCell
    data['numInd'] = elemInd
    data['categories'] = catCell
    data['sentenceLabels'] = sentenceLabels
    scipy.io.savemat(dataDir + '/toBeConverted.mat',data,oned_as='column')



if __name__ == '__main__':

    #input sentences
    sentences = open(sys.argv[1], 'r')
    dataDir = sys.argv[2]
    
    # temporary output for the stanford parser    
    rawSentencesStr = dataDir + '/rawSentences.txt'    
    rawSentences = open(rawSentencesStr, "w")

    elemStrings = [] # 
    elemInd = [] # the index of the elements
    numberWritten = 0
    sentenceNum = 1
    
    catNames = ['Cause-Effect(e1,e2)','Component-Whole(e1,e2)','Content-Container(e1,e2)','Entity-Destination(e1,e2)','Entity-Origin(e1,e2)',
                'Instrument-Agency(e1,e2)','Member-Collection(e1,e2)','Message-Topic(e1,e2)','Product-Producer(e1,e2)','Other',
                'Cause-Effect(e2,e1)','Component-Whole(e2,e1)','Content-Container(e2,e1)','Entity-Destination(e2,e1)','Entity-Origin(e2,e1)',
                'Instrument-Agency(e2,e1)','Member-Collection(e2,e1)','Message-Topic(e2,e1)','Product-Producer(e2,e1)']
    catDic = createCatDic(catNames)

    sentenceLabel = []
    line = sentences.readline()
    while True: 
        while (line != '' and not line[0].isdigit()):
            line = sentences.readline()
        if line == '':
            break
        line = line[line.find('"')+1:len(line)-3]
        
        
        formatOutput = formatLine(line)
        line = formatOutput[3]
        elemStrings.append(formatOutput[0])
        elemInd.append(formatOutput[2])
        rawSentences.writelines(line)

        line = sentences.readline()
        if (line != '' and not line[0].isdigit()):
            category = line
            # use dictionary to get sentence label
            sentenceLabel.append(catDic.get(category.strip()))
            line = sentences.readline()
        
        numberWritten += 1
        sentenceNum += 1


    

    print "A total of " + str(numberWritten) + " sentences were found." 
    sentences.close()
    rawSentences.close()
    #runStanfordParser(stanfordParserDir, rawSentencesStr)
    saveData(elemStrings,elemInd,sentenceLabel,catNames,dataDir)
    

    problemWords = ['programme','neighbour','clamour','organisation','harbour','labour','behaviour']
    replaceMents = ['program', 'neighbor','clamor','organization','harbor','labor','behavior']                    
    
