#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
liwc.py - using LIWC dictionary for sentiment analysis
    the script assumes a dictionary file at: liwc_data/LIWC2007_English131104.dic
@author: Yu-Ru Lin
@contact: yuruliny@gmail.com
@date: Jul 22, 2014

'''

import sys, os, csv
import nltk
import os

srcpath = os.path.dirname(os.path.abspath(__file__))

import time
import string
import pprint
from nltk.corpus import stopwords  # stopwords to detect language
from nltk import wordpunct_tokenize  # function to split up our words


#def utf8(str): return unicode(str, 'latin1').encode('utf-8').decode('utf8', 'ignore')


def get_language_likelihood(input_text):
    """Return a dictionary of languages and their likelihood of being the 
    natural language of the input text
    """

    input_text = input_text.lower()
    input_words = wordpunct_tokenize(input_text)

    language_likelihood = {}
    total_matches = 0
    for language in stopwords._fileids:
        language_likelihood[language] = len(set(input_words) &
                                            set(stopwords.words(language)))

    return language_likelihood


def get_language(input_text):
    """Return the most likely language of the given text
    """

    likelihoods = get_language_likelihood(input_text)
    return sorted(likelihoods, key=likelihoods.get, reverse=True)[0]


# read in liwc data file
def read_liwc(filename):
    liwc_data = open(filename, "r")

    mode = 0
    cat = {}
    dic = {}

    for line in liwc_data:
        line = line.strip("\r\n")
        if line.startswith('#'): continue  ## replace the line of 'like' with the next line
        if line == "%":
            mode += 1
            continue

        elif mode == 1:
            chunks = line.split("\t")
            cat[chunks[0]] = chunks[1]

        elif mode == 2:
            chunks = line.split("\t")
            word = chunks.pop(0)
            dic[word] = chunks

    return (cat, dic)  # cat = list of categories, dic = list of all words with categories


def get_cat2lex(cat, dic):
    lex2cat, cat2lex = {}, {}
    for lexicon in dic:
        for ci in dic[lexicon]:
            c = cat[ci]
            cat2lex.setdefault(c, set())
            cat2lex[c].add(lexicon)
            lex2cat.setdefault(lexicon, set())
            lex2cat[lexicon].add(c)
    return cat2lex, lex2cat


# read in dictionary and partition it into set of positive and negative word
def get_wordsets(dic):
    posemo = {}
    negemo = {}
    for word in dic:
        for cat in dic[word]:
            if cat in ['126']:
                posemo[word] = dic[word]
                continue
        for cat in dic[word]:
            if cat in ['19', '127', '128', '129', '130']:
                negemo[word] = dic[word]
                continue
    return (posemo, negemo)


'''
# determine if a tweet word matches an LIWC term (including prefix)
def matches(liwc_word, tweet_word):
    if liwc_word[-1] == "*":
        return tweet_word.startswith(liwc_word[:-1])
    else:
        return tweet_word == liwc_word
'''
def matches(liwc_word, tweet_word):
    if liwc_word[-1] == "*":
        liwc_word = liwc_word.replace('*', '')
    if liwc_word == tweet_word:
        return True
    else:
        return False

# general purpose function to determine if the string contains any of the
# substrings contained in set
def string_contains_any(string, set):
    for item in set:
        if item in string: return True
    return False


def detect_emoticons(tweet):
    pos_emoticons = [':-)', ':)', '(-:', '(:', 'B-)', ';-)', ';)']
    neg_emoticons = [':-(', ':(', ')-:', '):']

    emoticons_flag = 0
    if string_contains_any(tweet, pos_emoticons): emoticons_flag += 1
    if string_contains_any(tweet, neg_emoticons): emoticons_flag -= 1

    return emoticons_flag


# returns the positivity/negativity score for the given tweet
def classify(tweet):
    emo = detect_emoticons(tweet)
    if emo != 0: return emo

    # if no emoticons:

    tweet = (tweet.lower()).encode('utf-8')
    words = tweet.split(" ")

    word_count = 0  # len(words)
    pos_count = 0.0
    neg_count = 0.0

    # classify each of the words
    for word in words:
        if len(word) == 0 or word[0] == '@': continue  # if the word is prefixed with @, ignore it
        #word = word.translate(string.maketrans("", ""), string.punctuation)  # strip punctuation

        # check if the words match posemo/negemo
        for pos in posemo:
            if matches(pos, word):
                pos_count += 1
        for neg in negemo:
            if matches(neg, word):
                neg_count += 1
        word_count += 1

    pos_score = pos_count / word_count
    neg_score = neg_count / word_count

    if pos_score > neg_score: return 1
    if pos_score < neg_score: return -1
    return 0


def get_text2cat(cat, dic, text):
    cat2cnt = {}
    words = text.lower()
    words = words.translate(str.maketrans('', '', string.punctuation))
    words = words.replace('<', '')
    words = words.replace('>', '')
    #print(words)
    #words = (text.lower()).encode('utf-8')
    #words = words.translate(string.maketrans("", ""), string.punctuation)  # strip punctuation
    words = words.split()
    word_count = len(words)
    cat2cnt.setdefault('wc', word_count)
    dic_count = 0
    p1, p2, p3 = 0, 0, 0  # 1st,2nd,3rd Personal pronouns counts
    past, present, future = 0, 0, 0  # verb tense

    for word in words:
        for lexicon in dic:
            if matches(lexicon, word):
                dic_count += 1
                for cid in dic[lexicon]:
                    c = cat[cid]
                    if c in ['i', 'we']: p1 += 1
                    if c in ['you']: p2 += 1
                    if c in ['shehe', 'they']: p3 += 1
                    if c in ['past']: past += 1
                    if c in ['present']: present += 1
                    if c in ['future']: future += 1
                    cat2cnt.setdefault(c, 0)
                    cat2cnt[c] += 1
    cat2cnt.setdefault('dic_wc', dic_count)
    cat2cnt.setdefault('p1', p1)
    cat2cnt.setdefault('p2', p2)
    cat2cnt.setdefault('p3', p3)
    cat2cnt.setdefault('past', past)
    cat2cnt.setdefault('present', present)
    cat2cnt.setdefault('future', future)
    return cat2cnt


def get_text2word(cat, dic, text, select_cats=['negemo', 'posemo']):
    word2cnt, word2cat = {}, {}
    words = text.lower()
    words = words.translate(str.maketrans('', '', string.punctuation))
    words = words.replace('<', '')
    words = words.replace('>', '')
    print(words)
    #words = (text.lower()).encode('utf-8')
    #words = words.translate(string.maketrans("", ""), string.punctuation)  # strip punctuation
    words = words.split()

    for word in words:
        for lexicon in dic:
            if matches(lexicon, word):
                selected = False
                for cid in dic[lexicon]:
                    c = cat[cid]
                    if c in ['i', 'we']: c = 'p1'
                    if c in ['you']: c = 'p2'
                    if c in ['shehe', 'they']: c = 'p3'
                    if c in select_cats:
                        selected = True;
                        break
                if not selected: continue
                word2cnt.setdefault(lexicon, 0)
                word2cnt[lexicon] += 1
                for cid in dic[lexicon]:
                    c = cat[cid]
                    if c in ['i', 'we']: c = 'p1'
                    if c in ['you']: c = 'p2'
                    if c in ['shehe', 'they']: c = 'p3'
                    word2cat.setdefault(lexicon, [])
                    word2cat[lexicon].append(c)

    return word2cnt, word2cat


def output_posneg(cat, dic):
    cat2lex, lex2cat = get_cat2lex(cat, dic)
    # pprint.pprint(cat2lex['posemo'])
    print(cat2lex['posemo'])
    ofilename = 'liwc_terms_all.csv'
    header_row = ['category', 'lexicon']
    rows = [header_row]
#    for c in ['posemo', 'negemo']:
    for c in ['posemo', 'anger', 'anx', 'sad']:

        for w in cat2lex[c]:
            row = [c, w]
            rows.append(row)
    ofile = open(ofilename, "w")
    writer = csv.writer(ofile, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows(rows)
    ofile.close()
    print('save to', ofilename, len(rows))


def output_all_lexicon(cat, dic):
    cat2lex, lex2cat = get_cat2lex(cat, dic)
    ofilename = 'liwc_all_terms.csv'
    header_row = ['category', 'lexicon']
    rows = [header_row]
    for c, terms in cat2lex.iteritems():
#    for c in ['posemo', 'anger', 'anx', 'sad', ]:
        for w in terms:
            row = [c, w]
            rows.append(row)
    ofile = open(ofilename, "w")
    writer = csv.writer(ofile, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows(rows)
    ofile.close()
    print('save to', ofilename, len(rows))

# --- CALLED ON IMPORT ---

def getLIWCFeatures(cat, dic, text, selected_cats=None):
    result = {}
    if selected_cats == None:
        selected_cats = list(cat.values())
        selected_cats.extend(['p1', 'p2', 'p3', 'wc', 'dic_wc'])
        
    text_result = get_text2cat(cat, dic, text)
    #text_result2 = get_text2word(cat, dic, text)
    
    #print(text_result)
    #print(text_result2)
    
    for cat in selected_cats:
        if cat in text_result:
            result[cat] = text_result[cat]
        else:
            result[cat] = 0
    
    return result


if __name__ == '__main__':
    # posemo, negemo = get_wordsets(dic)

    # language = get_language('@EarvinBrown Wie geht es dir!')
    language = 'english'
    print(language)
    if language == "english":
        cat, dic = read_liwc('%s/liwc_data/LIWC2007_English131104.dic'%srcpath)
    if language == "french":
        cat, dic = read_liwc('%s/liwc_data/FrenchLIWCDictionary.dic'%srcpath)
    if language == "dutch":
        cat, dic = read_liwc('%s/liwc_data/LIWC2007_Dutch.dic'%srcpath)
    if language == "german":
        cat, dic = read_liwc('%s/liwc_data/LIWC2001_German.dic'%srcpath)

    print(get_text2cat(cat, dic, 'I am not worried, indeed!'))
    print(get_text2word(cat, dic, 'I am not worried, indeed!'))
    # print(get_text2word(cat, dic, 'I am not worried, indeed!'))
    # print(get_text2word(cat, dic, 'worried worry'))

    # output_posneg(cat,dic)
#    output_all_lexicon(cat,dic)

