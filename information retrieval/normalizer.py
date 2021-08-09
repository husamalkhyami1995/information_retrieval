import re

dictionary = {
    "u.s": "united states",
    "u.s.a": "united states",
    "u.n": "united nations",
    "c.i.a": "central intelligence agency",
    "r.p.f": "rally of the french people",
    "s.a.o": "secret army organisation",
    "i.e": "example",
    "m.p.s": "member of parliament of the united kingdom",
    "m.p": "member of the house of lords",
}


def expand(text: str):
    for key in dictionary.keys():
        text = text.replace(key, dictionary[key])
    return text


def reduce(text: str):
    abbreviations = re.findall(r"\w\.\w[.\w]*", text)
    for exp in abbreviations:
        text = text.replace(exp, exp.replace(".", ""))
    dash_separated = re.findall(r"\w-\w[-\w]*", text)
    for exp in dash_separated:
        text = text.replace(exp, exp.replace("-", ""))
    return text


def normalize(text):
    text = expand(text)
    text = reduce(text)
    return text

print(normalize("r.p.f"))