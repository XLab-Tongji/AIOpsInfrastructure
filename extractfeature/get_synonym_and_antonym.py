from nltk.corpus import wordnet as wn


def get_single_word_synonym(word):
    synonym = set()
    for synset in wn.synsets(word):
        temp = synset.lemma_names()
        for i in range(len(temp)):
            if temp[i] != word:
                synonym.add(temp[i])
    return synonym


def get_single_word_antonym(word):
    antonym = set()
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            if l.antonyms() and l.antonyms()[0].name() != word:
                antonym.add(l.antonyms()[0].name())
    return antonym


def to_string(pair):
    if pair[0] < pair[1]:
        return pair[0] + ',' + pair[1]
    return pair[1] + ',' + pair[0]


def get_synonym_and_antonym(ls):
    synonym = {}
    antonym = {}
    synonym_pair = set()
    antonym_pair = set()
    for i in range(len(ls)):
        a = ls[i].lower()
        synonym[ls[i]] = get_single_word_synonym(a)
        antonym[ls[i]] = get_single_word_antonym(a)
    for i in range(len(ls)):
        for word in ls:
            temp_word = word.lower()
            if temp_word in synonym[ls[i]]:
                synonym_pair.add(to_string([ls[i], word]))
            if temp_word in antonym[ls[i]]:
                antonym_pair.add(to_string([ls[i], word]))
    return synonym, antonym, synonym_pair, antonym_pair
