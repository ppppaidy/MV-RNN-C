%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                                 %
%                                       Code                                                      %
% Semantic Compositionality through Recursive Matrix-Vector Spaces    	                          %
% Richard Socher, Brody Huval, Christopher D. Manning and Andrew Y. Ng							  %
% Conference on Empirical Methods in Natural Language Processing (EMNLP 2012, Oral) 		      %
% See http://www.socher.org for more information or to ask questions                              %
%                                                                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

-------------------------------------------
For classifying relations
-------------------------------------------
If you would like to get the relations of new data using our trained model, run the classifyRelations.sh script inside the classifyRelations folder:

classifyRelations.sh  <input text file>  <output mat file>

The data should be in the same format as SemEval Task 8 shown below. 

Example:

1	"The most common <e1>audits</e1> were about <e2>waste</e2> and recycling."
2	"The <e1>company</e1> fabricates plastic <e2>chairs</e2>."
3	"The school <e1>master</e1> teaches the lesson with a <e2>stick</e2>."
4	"The suspect dumped the dead <e1>body</e1> into a local <e2>reservoir</e2>."


Example with escape characters:
'8011\t"Ten million quake <e1>survivors</e1> moved into makeshift <e2>houses</e2>."\r\n'

This will produce a proposed answer file as shown below in the results folder in the working directory, as well as the data in a mat file specified as a command line argument.

Message-Topic(e1,e2)
Product-Producer(e2,e1)
Instrument-Agency(e2,e1)
Entity-Destination(e1,e2)


-------------------------------------------
For testing accuracy
-------------------------------------------
To test accuracy of the model with provided labels on new data, put the text in the following format, similar to the SemEval Task 8 text data. Then specify the paths within classifyRelations.sh and run this script as before:

classifyRelations.sh  <input text file>  <output mat file>


Example:

1	"The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>."
Component-Whole(e2,e1)

2	"The <e1>child</e1> was carefully wrapped and bound into the <e2>cradle</e2> by means of a cord."
Other

3	"The <e1>author</e1> of a keygen uses a <e2>disassembler</e2> to look at the raw assembly code."
Instrument-Agency(e2,e1)

4	"A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
Other


-------------------------------------------
Training and testing the full model
------------------------------------------- 

For training in testing of the full model with data from SemEval Task 8, run in matlab inside the deepDualCamera folder:

trainDual

For testing with previously trained parameters, run:

test_with_external_features
or
test_without_external_features

If you would like to run the code on a small machine for studying it, set params.tinyDataSet = 1 within initParams.


-------------------------------------------
External packages
-------------------------------------------

To classify relations or test accuracy with new data in the SemEvam Task 8 text format, we use the following:
- Stanford Parser (http://nlp.stanford.edu/software/lex-parser.shtml)
- convertStanfordParserTrees.m (provided)
- sst-light-0.4 (http://sourceforge.net/projects/supersensetag/)
- python with numpy and scipy
The first three are included. You need to make sure you have python installed properly. 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                   %
%                Bibtex                             %
%                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  If you use the code, please cite:
  
@inproceedings{SocherEtAl2012:MVRNN,
author = {Richard Socher and Brody Huval and Christopher D. Manning and Andrew Y. Ng},
title = {{Semantic Compositionality Through Recursive Matrix-Vector Spaces}},
booktitle = {Proceedings of the 2012 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
year = 2012
}  
