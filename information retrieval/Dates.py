import glob
import re

corpus_files = glob.glob('IR Homework/corpus/*.txt')

for file_name in corpus_files:
    FI = open(file_name, 'r')
    FO = open(file_name.replace('txt', 'dates'), 'w+')
    for line in FI:

        dates = re.compile(r'((\d{2,4}[\/ -]|(, )|(\(TIME, ))*('
                           r'JANUARY|JAN|FEBRUARY|FEB|MARCH|MAR|APRIL|APR|MAY|JUNE|JUN|JULY|JUL|AUGUST|AUG'
                           r'|SEPTEMBER|SEP|OCTOBER|OCT|NOVEMBER|NOV|DECEMBER|DEC)(((, )|[.\/ -])*\d{1,4})+(\))*)|((['
                           r'1-9]|1[012])[-/.]([1-9]|[12][0-9]|3[01])[-/.](19|20)\d\d)|((1[012]|0[1-9])(3['
                           r'01]|2\d|1\d|0[1-9])(19|20)\d\d)|((1[012]|0[1-9])[-/.](3[01]|2\d|1\d|0[1-9])[-/.]('
                           r'19|20)\d\d)')
        result = dates.findall(line)
        if result is not None:
            print(result)

    FO.close()
    FI.close()
