#!/usr/bin/env python
# coding: utf-8

import re
import os
from itertools import chain

data_path = os.path.join(os.getcwd(), "data\en_words\words.txt")

word_regexes = {
    'en': r'[A-Za-z]+'
}

alphabets = {
    'en': 'abcdefghijklmnopqrstuvwxyz',
}


def load_data(file):
    f = open(file, 'r')
    words = {}
    for (i, word) in enumerate(f):
        word = word.strip('\n')
        pairs = word.split(" ")
        words[pairs[0]] = pairs[1]
    f.close()
    return words


class Correcter:
    def __init__(self, threshold=0, lang='en'):
        self.threshold = threshold
        self.nlp_data = load_data(data_path)
        self.lang = lang

        if threshold > 0:
            print(f'Original number of words: {len(self.nlp_data)}')
            self.nlp_data = {k: v for k, v in self.nlp_data.items() 
                            if v > threshold}
            print(f'After applying threshold: {len(self.nlp_data)}')

    def existing(self, words):
        """{'the', 'teh'} => {'the'}"""
        return set(word for word in words
                   if word in self.nlp_data)

    def autocorrect_word(self, word):
        """most likely correction for everything up to a double typo"""
        w = Word(word, self.lang)
        candidates = (self.existing([word]) or 
                      self.existing(list(w.typos())) or 
                      self.existing(list(w.double_typos())) or 
                      [word])
        
        return min(candidates, key=lambda k: self.nlp_data[k])

    def autocorrect_sentence(self, sentence):
        return re.sub(word_regexes[self.lang],
                      lambda match: self.autocorrect_word(match.group(0)),
                      sentence)
                      
    def __call__(self, sentence):
        return(self.autocorrect_sentence(sentence))


class Word(object):
    """container for word-based methods"""

    def __init__(self, word, lang='en'):
        """
        Generate slices to assist with typo
        definitions.
        'the' => (('', 'the'), ('t', 'he'),
                  ('th', 'e'), ('the', ''))
        """
        slice_range = range(len(word) + 1)
        self.slices = tuple((word[:i], word[i:])
                            for i in slice_range)
        self.word = word
        self.alphabet = alphabets[lang]

    def _deletes(self):
        """th"""
        return (self.concat(a, b[1:])
                for a, b in self.slices[:-1])

    def _transposes(self):
        """teh"""
        return (self.concat(a, reversed(b[:2]), b[2:])
                for a, b in self.slices[:-2])

    def _replaces(self):
        """tge"""
        return (self.concat(a, c, b[1:])
                for a, b in self.slices[:-1]
                for c in self.alphabet)

    def _inserts(self):
        """thwe"""
        return (self.concat(a, c, b)
                for a, b in self.slices
                for c in self.alphabet)
    
    def concat(self, *args):
        """reversed('th'), 'e' => 'hte'"""
        try:
            return ''.join(args)
        except TypeError:
            return ''.join(chain.from_iterable(args))


    def typos(self):
        """letter combinations one typo away from word"""
        yield from self._deletes()
        yield from self._transposes()
        yield from self._replaces()
        yield from self._inserts()

    def double_typos(self):
        """letter combinations two typos away from word"""
        return (e2 for e1 in self.typos()
                for e2 in Word(e1).typos())


correct = Correcter(lang='en')
correct("th")


# In[ ]:




