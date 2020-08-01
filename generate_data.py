# coding=utf-8

import argparse
import pprint
import gensim
import numpy
import pickle
from glove import Glove
from glove import Corpus
from data import read_text_by_line


def vocab_build(data, vocab_path, min_count):
    """
    :param data:
    :param vocab_path:
    :param min_count:
    :return:
    """
    word2id = {}
    for sent_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)
    print("%s word save to pickle" % len(word2id))
    return word2id


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

sentense = []
data = read_text_by_line('data_path/original/train1.txt')
vocab = vocab_build(data, 'data_food/word2id.pkl', 3)
count = 1
for line in data:
    str = []
    for char in line:
        if char.isdigit():
            str.append('<NUM>')
        if '\u4e22' <= char <= '\u9fa5':
            str.append(char)
    count += 1
    if count % 10000 == 0:
        print("%s lines has been load! current str: %s" % (count, ''.join(str)))
    sentense.append(str)

# sentense = [['你','是','谁'],['我','是','中国人']]
corpus_model = Corpus()
corpus_model.fit(sentense, window=10)
#corpus_model.save('corpus.model')

print('Dict size: %s' % len(corpus_model.dictionary))
print('Collocations: %s' % corpus_model.matrix.nnz)
# print(corpus_model.matrix.todense().tolist())

glove = Glove(no_components=300, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=10,
          no_threads=1, verbose=True)
glove.add_dictionary(corpus_model.dictionary)
print(glove.dictionary)
# print(type(glove.word_vectors[0]))
# print(glove.word_vectors[glove.dictionary['当']], len(glove.word_vectors[glove.dictionary['当']]))
# print(glove.most_similar('北', number=10))

embedding = [[0.0]*300 for i in range(len(vocab))]
valid_emb_cnt = 0
for word, id in vocab.items():
    if word in glove.dictionary:
        valid_emb_cnt += 1
        emb_index = glove.dictionary[word]
        embedding[id] = glove.word_vectors[emb_index]
print("%s word has new embedding" % valid_emb_cnt)
np_embedding = numpy.asarray(embedding)
print(np_embedding.shape)
# with open('data_food/pretrain_embedding.npy', 'wb') as f:
numpy.save("data_food/pretrain_embedding.npy", np_embedding)