import glob
from nltk.tokenize import word_tokenize

corpus_files = glob.glob('IR Homework/corpus/*.txt')

with open('irregular verbs.txt') as f:
    verbs = f.read().splitlines()

for file_name in corpus_files:
    FI = open(file_name, 'r')
    FO = open(file_name.replace('txt', 'verbs'), 'w+')
    for line in FI:
        words = word_tokenize(line)
        for w in words:
            if w.lower() in verbs:
                FO.write(w + "\n")
    FO.close()
    FI.close()
