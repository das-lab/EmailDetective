# -*- coding: utf-8 -*-
import pickle

import gensim
from keras.preprocessing.text import Tokenizer
from gensim.models.word2vec import Word2Vec
import numpy as np
from config import TOP55
from mongodb import *

COUNT = 4000


def generateLine():
    for name in TOP55:
        contents = get_person_by_name3(name, content=1)
        for content in contents:
            c = content['content']
            yield c


class MyChars(object):

    def __iter__(self):
        for name in TOP55:
            contents = get_person_by_name3(name, content=1)
            for content in contents:
                c = content['content']
                j = []
                for char in c:
                    if char != ' ':
                        j.append(char)
                yield j


def char2vec():
    chars = MyChars()
    model = gensim.models.Word2Vec(chars, size=256, min_count=1, iter=5, window=5)
    model.save('w2v_50_256.m')


def train_token():
    token = Tokenizer(num_words=None, filters='', lower=False, char_level=True)
    for line in generateLine():
        token.fit_on_texts(line)
    pickle.dump(token, open('50char-token.pkl', 'wb'))


def load_char_emb():
    print('load-char-emb')
    tokenizer = pickle.load(open('50char-token.pkl', 'rb'))
    word_index = tokenizer.word_index
    vocab_size = len(word_index)
    model = Word2Vec.load('w2v_50_256.m')
    word_vectors = model.wv
    embeddings_index = dict()
    for word, vocab_obj in model.wv.vocab.items():
        if int(vocab_obj.index) < 100:
            embeddings_index[word] = word_vectors[word]
    del model, word_vectors
    num_words = min(100, vocab_size)
    not_found = 0
    embedding_matrix = np.zeros((num_words + 1, 256))  
    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            not_found += 1
    print('load-char-emb ok')
    return embedding_matrix
