import ntpath

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer
from collections import Counter
from num2words import num2words
from nltk.tag import StanfordNERTagger
from fuzzy_match import match

from autocorrect import Speller
import nltk
import os
java_path = "C:\\Program Files\\Java\\jdk1.8.0_291\\bin\\java.exe"
os.environ['JAVA_HOME'] = java_path

import numpy as np

import re
import math
import glob
import time

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data):

    with open('C:\\Users\\husam\\Desktop\\IR with satnford lib\\IR Homework\\stop words.txt') as f:
        stopWords = f.read().splitlines()

    while ("" in stopWords): 
        stopWords.remove("") 
    # stop_words = stopwords.words('english')
    
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if str.upper(w) not in stopWords:
            new_text = new_text + " " + w
    return new_text

def auto_correct(data):

    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        new_text = new_text + " " + spell(str.lower(w))
    return new_text
    
def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def lemmatizing(data):
    
    lemmatizer = WordNetLemmatizer() 
    # stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + lemmatizer.lemmatize(w)

    return new_text

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text

def extract_dates(data):
    dates = []
    matches = re.findall('((\d{2,4}[\/ -]|(, )|(\(TIME, ))*(JANUARY|JAN|FEBRUARY|FEB|MARCH|MAR|APRIL|APR|MAY|MAY|JUNE|JUN|JULY|JUL|AUGUST|AUG|SEPTEMBER|SEP|OCTOBER|OCT|NOVEMBER|NOV|DECEMBER|DEC)(((, )|[.\/ -])*\d{1,4})+(\))*)', data)
    for match in matches:
        dates.append(match[0])
    return dates


st = StanfordNERTagger('C:\\Users\\husam\\Desktop\\IR with satnford lib\\stanford-ner-4.0.0\\classifiers\\english.all.3class.distsim.crf.ser.gz',
                       'C:\\Users\\husam\\Desktop\\IR with satnford lib\\stanford-ner-4.0.0\\stanford-ner.jar', encoding='utf8')


def extract_names(data):
    names = []
    s = st.tag(data.split())
    for name in s:
        if name[1] == 'PERSON':
            names.append(name[0])
    return names


# remove duplicated elements
def reducelist(data):
    return list(dict.fromkeys(data))

def compare_name(name,names_list,lim):
    return match.extract(str.upper(name), names_list, limit=lim, scorer='ratio')

def replace_name(from_name,to_name,query):
    new = ""
    # print(to_name[0])
    for w in word_tokenize(query):
        if w == from_name:
            new = new + " " + to_name[0]
        else:
            new = new + " " + w
    return new

def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c

def matching_score(k, query):

    corrected_query = auto_correct(query)
    query_names = extract_names(corrected_query)
    replaced_query = corrected_query
    for name in query_names:
        with open('C:\\Users\\husam\\Desktop\\IR with satnford lib\\names.txt') as f:
            a_names = reducelist(remove_punctuation(f.readlines()))
        new_name = compare_name(name,a_names,1)[0]
        replaced_query = replace_name(name,new_name,query)
    preprocessed_query = preprocess(replaced_query)

    tokens = word_tokenize(str(preprocessed_query))

    print("Matching Score")
    
    # print("\nCorrected Query:", corrected_query)
    # print("\nNew Query:", replaced_query)
    
    # print("")
    # print(tokens)
    
    query_weights = {}

    for key in tf_idf:
        
        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]
    
    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)
    
    # print("")
    
    l = []
    
    for i in query_weights[:k]:
        l.append(i[0])
    
    # print(l)
    return l

def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    return cos_sim

def gen_vector(tokens):

    Q = np.zeros((len(total_vocab)))
    
    counter = Counter(tokens)
    words_count = len(tokens)

    query_weights = {}
    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = math.log10((N+1)/(df+1))

        try:
            ind = total_vocab.index(token)
            Q[ind] = tf*idf
        except:
            pass
    return Q


def cosine_similarity(k, query):
    # print("Cosine Similarity")

    corrected_query = auto_correct(query)
    query_names = extract_names(corrected_query)
    replaced_query = corrected_query
    for name in query_names:
        with open('names.txt') as f:
            a_names = reducelist(remove_punctuation(f.readlines()))
        new_name = compare_name(name, a_names, 1)[0]
        replaced_query = replace_name(name, new_name, query)
    preprocessed_query = preprocess(replaced_query)

    tokens = word_tokenize(str(preprocessed_query))

    # print("\nQuery:", query)

    # print("")
    # print(tokens)

    d_cosines = []

    query_vector = gen_vector(tokens)

    for d in (D):
        d_cosines.append(cosine_sim(query_vector, d))

    out = np.array(d_cosines).argsort()[-k:][::-1]
    Out = np.delete(out, 0)

    # print("")

    # print(Out)
    return Out

def preprocess(data):
    data = convert_numbers(data)
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = lemmatizing(data)
    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
    return data

def calculatePrecisionAndRecallForResults(mainResult,result):

    count = 0

    for r in result:
        for R in mainResult:
            if r == R:
                count = count+1

    values = {'precision':0, 'recall':0, 'F':0}

    values['precision'] = precision = count / len(result)
    values['recall'] = recall = count / len(mainResult)

    if precision > 0 and recall > 0:
        values['F'] = 1 / ( 0.5 * ( (1/precision) + (1/recall) ) )

    return values

spell = Speller(lang='en')

list_of_files = glob.glob('C:\\Users\\husam\\Desktop\\IR with satnford lib\\IR Homework\\corpus_processed\\*.txt')  # create the list of file


processed_text = []
c = []
file_names = []

print("start indexing ....")
startIndexingTime = time.time()

for file_name in list_of_files:
    FI = open(file_name, 'r')
    f_n = os.path.splitext(os.path.basename(file_name))[0]
    text = FI.read().strip()
    FI.close()
    c = word_tokenize(str(preprocess(text)))
    # print(f_n+' : '+str(len(c)))
    file_names.append(f_n)
    processed_text.append(c)

DF = {}

N = len(processed_text)
for i in range(N):
    tokens = processed_text[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}
for i in DF:
    DF[i] = len(DF[i])

total_vocab_size = len(DF)
# print(total_vocab_size)

total_vocab = [x for x in DF]

j = 0

tf_idf = {}

for i in range(N):

    tokens = processed_text[i]
    doc = int(file_names[j])

    counter = Counter(tokens)
    words_count = len(tokens)

    for token in np.unique(tokens):
        tf = counter[token] / words_count
        df = doc_freq(token)
        idf = np.log10((N + 1) / (df + 1))

        tf_idf[doc, token] = tf * idf

    j += 1

D = np.zeros((N, total_vocab_size))

for i in tf_idf:
    try:
        ind = total_vocab.index(i[1])
        D[i[0]][ind] = tf_idf[i]
    except:
        pass

# indexing without extracting names Because I've Extracted them before and saved 'em in external file
IndexingTime = time.time() - startIndexingTime
print("finish indexing ....")

print(len(tf_idf))
Results = {}
R = open('C:\\Users\\husam\\Desktop\\IR with satnford lib\\IR Homework\\relevance.txt','r')
for f in R:
    r = list(map(int, f.split()))
    if len(r) > 0:
        i = r[0]
        r.pop(0)
        Results[i] = r
        

Query = ""
Query_num = 1
Qu = open('C:\\Users\\husam\\Desktop\\IR with satnford lib\\IR Homework\\Queries.txt','r')
FO = open('results.txt', 'w+')
FO.write("Indexing Time = "+str(IndexingTime)+" Seconds \n\n")
FO.write("P.S : indexing without extracting names Because I've Extracted them before and saved 'em in external file \n\n\n")
for line in Qu:
    if line.startswith('*FIND') or line.startswith('*STOP'):
        if len(Query) > 0:

            FO.write("----------------------------------------------------------------------------------------\n\n")

            FO.write("Query "+str(Query_num)+" :\n")
            print("process Query : "+str(Query_num))

            main_result = Results[Query_num]
            FO.write("  Main results : "+str(main_result)+"\n")

            numOfResults = 10
            if len(main_result) > 10:
                numOfResults = len(main_result)

            start_time = time.time()
            cosResult = cosine_similarity(numOfResults+1,Query) # we add 1 here because the first compare will be with the 0 index in (D), So the results always have the 0 value, So I had to get num +1 then remove the 0 value from results
            FO.write("  cosine_similarity results : "+str(cosResult))
            cosSimilarityTime = time.time() - start_time
            c = calculatePrecisionAndRecallForResults(main_result,cosResult)
            FO.write("\n  Execution time = "+str(cosSimilarityTime)+" Seconds \n    "+str(c)+"\n\n")

            start_time = time.time()
            matResult = matching_score(numOfResults,Query)
            FO.write("  matching_score results : "+str(matResult))
            matchingScoreTime = time.time() - start_time
            c = calculatePrecisionAndRecallForResults(main_result,matResult)
            FO.write("\n  Execution time = "+str(matchingScoreTime)+" Seconds \n    "+str(c)+"\n\n")

            Query = ""
            Query_num += 1
        pass
    else:
        Query += " "+line
FO.close()
Qu.close()
