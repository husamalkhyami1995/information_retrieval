import glob
from nltk.tag import StanfordNERTagger
from os import path


def extract_names(file_name, param, fo):
    file = open(file_name, 'r')
    print(file_name)
    for line in file:
        s = param.tag(line.split())
        for name in s:
            if name[1] == 'LOCATION':
                FO.write(name[0]+" -> LOCATION"+ '\n')
            if name[1] == 'PERSON':
                FO.write(name[0] +"->  PERSON"+ '\n')
    file.close()


st = StanfordNERTagger('stanford-ner-4.0.0/classifiers/english.all.3class.distsim.crf.ser.gz',
                       'stanford-ner-4.0.0/stanford-ner.jar', encoding='utf8')

list_of_files = glob.glob('IR Homework/corpus/*.txt')

i = 1
fileName = 'IR Homework/corpus/' + str(i) + ".txt"
while path.exists(fileName):
    fileName = 'IR Homework/corpus/' + str(i) + ".txt"
    i += 1
    FO = open(fileName.replace('txt', 'names'), 'w+')
    extract_names(fileName, st, FO)
    FO.close()
