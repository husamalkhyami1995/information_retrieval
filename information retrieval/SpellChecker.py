from nltk.tokenize import word_tokenize

from autocorrect import Speller

spell = Speller(lang='en')

FI = open('IR Homework/Queries.txt', 'r')
for line in FI:
    words = word_tokenize(line)
    for w in words:
        w1 = str.lower(w)
        w2 = spell(w1)
        check = (w1 == w2)
        print(w1 + ' >>> ' + w2 + ' | ' + str(check))
FI.close()
