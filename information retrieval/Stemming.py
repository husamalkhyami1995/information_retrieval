import glob
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

corpus_files = glob.glob('IR Homework/corpus/*.txt')  # create the list of file
p_stemmer = SnowballStemmer(language='english')

# Read stop words file
with open('IR Homework/stop words.txt') as f:
    stopWords = f.read().splitlines()

while "" in stopWords:
    stopWords.remove("")

for file_name in corpus_files:
    FI = open(file_name, 'r')
    FO = open(file_name.replace('txt', 'stemmed'), 'w+')
    for line in FI:
        words = word_tokenize(line)
        for w in words:
            if w.isalnum():
                if w not in stopWords:
                    FO.write(f'word {w} has stemming {p_stemmer.stem(w)} \n')
    FO.close()
    FI.close()
