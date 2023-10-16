# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 21:04:50 2021

@author: c
"""
import numpy as np
import pandas as pd
import string
import os
import shutil
import pickle as pickle
from sklearn.utils import shuffle
from os.path import expanduser
from nltk.corpus import wordnet
from transformers import BertTokenizer, RobertaTokenizer



#得到单词的所有词义集合
def get_synsets(input_lemma):
    #wordnet = nltk.corpus.wordnet
    synsets = []
    for syn in wordnet.synsets(input_lemma):
        for lemma in syn.lemmas():
            synsets.append(lemma.name())

    return synsets


#得到两个句子之间的相似度矩阵
def get_similarity(text_a, text_b, k):
    #wordnet = nltk.corpus.wordnet
    left = text_a[k]
    left = left.replace('\'', '\' ')
    right = text_b[k]
    right = right.replace('\'', '\' ')

    left_lsent = left.lower().split()
    right_lsent = right.lower().split()
    print('num:', k)
    sim = []
    for i in range(len(left_lsent)):
        word = left_lsent[i]
        tmp = []
        for j in range(len(right_lsent)):
            targ = right_lsent[j]
            left_syn = get_synsets(word)
            right_syn = get_synsets(targ)#得到两个单词的所有lemma集合
            left = wordnet.synsets(word)
            right = wordnet.synsets(targ)#得到两个单词的sense集合
            if left != [] and right != []:
                if targ in left_syn or word in right_syn:#如果两个单词词义相同
                    tmp.append(1.0)
                else:                               #词义不同，计算相似度
                    flag = False
                    for m in left_syn:
                        for n in right_syn:
                            if m==n :
                                tmp.append(1.0)
                                flag = True
                                break
                        if flag == True:
                            break
                    if flag:
                        continue
                    count1, count2= 0, 0
                    ScoreList1, ScoreList2 = 0, 0
                    for word1 in left:
                        for word2 in right:
                            try:
                                score1 = word1.wup_similarity(word2)
                            except:
                                score1 = 0.0
                            try:
                                score2 = word2.wup_similarity(word1)
                            except:
                                score2 = 0.0
                            #score1 = word1.stop_similarity(word2)
                            if score1 is not None:
                                ScoreList1 += score1
                                count1 += 1
                            if score2 is not None:
                                ScoreList2 += score2
                                count2 += 1

                    if count1 + count2 != 0:
                        similarity = (ScoreList1 + ScoreList2)/(count1 + count2)
                        tmp.append(similarity)
                    else:
                        if word == targ:
                            tmp.append(1)
                        else:
                            tmp.append(0)
            else:
                if word == targ:#如果单词在wordnet中不存在，相同则为1，不同则为0
                    tmp.append(1)
                else:
                    if word in string.punctuation or targ in string.punctuation:
                        tmp.append(1)
                    else:
                        tmp.append(0)
        sim.append(tmp)#一个词与另一个句子所有词的相似度

    return sim

def output(todir, test):

    test_a, test_b, test_similarity = test['text_a'], test['text_b'], test['similarity']
    if 'ro' in todir:
        tokenizer = RobertaTokenizer.from_pretrained('./roberta_model')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    test['sim'] = [''] * len(test_a)
    ids = []
    # f = open('./datasets/quora/tmp/dev_um_similarity.txt', 'w', encoding='utf-8')
    for idx in range(len(test_a)):
        print(idx)
        text_a = test_a[idx]
        text_b = test_b[idx]
        if type(text_a) == float or type(text_b) == float or text_a == 'N/A' or \
                text_a == 'N\A' or text_b == 'N/A' or text_b == 'N\A':
            print(text_a, text_b)
            test['sim'][idx] = ''
            ids.append(idx)
            continue
         # pair of sequences: ``<s> A </s></s> B </s>``
        text_a = text_a.replace('\'', '\' ')
        text_b = text_b.replace('\'', '\' ')
        text_a = text_a.lower().split()
        # test['text_a'][idx] = text_a
        # text_a = text_a.split()
        text_b = text_b.lower().split()
        # test['text_b'][idx] = text_b
        # text_b = text_b.split()
        if 'ro' in todir:
            text = ['<s>'] + text_a + ['</s>', '</s>'] + text_b + ['</s>']

            tmp_sim = np.zeros((len(text), len(text)))
            similarity = np.array(eval(test_similarity[idx]))
            
            tmp_sim[1:len(text_a) + 1, len(text_a) + 3:-1] = similarity
            tmp_sim[len(text_a) + 3:-1, 1:len(text_a) + 1] = similarity.T
        else:
            text = ['[CLS]'] + text_a + ['[SEP]'] + text_b + ['[SEP]']

            tmp_sim = np.zeros((len(text), len(text)))
            similarity = np.array(eval(test_similarity[idx]))
            
            tmp_sim[1:len(text_a) + 1, len(text_a) + 2:-1] = similarity
            tmp_sim[len(text_a) + 2:-1, 1:len(text_a) + 1] = similarity.T

        tag_a = []
        for i in range(len(text)):
            word = text[i]
            token = tokenizer.tokenize(word)
            tag = [i] * len(token)
            tag_a = tag_a + tag

        tag = np.zeros((len(tag_a), len(tag_a)))
        for i in range(len(tag_a)):
            for j in range(len(tag_a)):
                tag[i][j] = tmp_sim[tag_a[i]][tag_a[j]]


        tag = tag.tolist()
        test['sim'][idx] = tag

    test.drop(ids, inplace=True)
    data = test.reset_index(drop=True)
    del data['similarity']
    text = data.rename(columns={'sim':'similarity'})

    text.to_csv(todir, sep='\t', index=None)
    print('end')


dataset = 'STS-B' #dataset_name
for task in ['_large', '_sim', '_ro_large', '_ro_sim']:
    for name in ['train', 'dev', 'test']:
        todir = f'./datasets/{dataset}{task}'
        if not os.path.exists(todir):
            os.mkdir(todir) 
        todir = todir + f'/{name}.tsv'
        if 'large' in task:
            shutil.copyfile('./datasets/{}/{}.tsv'.format(dataset, name), todir)
        else:
            test_data = pd.read_csv('./datasets/{}/{}.tsv'.format(dataset, name), sep='\t')
            test_a, test_b, label, similarity = test_data['text_a'], test_data['text_b'], test_data['labels'], []
            print('total num:', len(label))
            for i in range(len(label)):
                sim = get_similarity(test_a, test_b, i)
                similarity.append(sim)
            # data = pd.DataFrame(columns=['text_a', 'text_b', 'labels', 'similarity'])
            # data["text_a"] = text_a
            # data["text_b"] = text_b
            # data["labels"] = label
            # data["similarity"] = similarity
            # data.to_csv('./datasets/{}/{}_similarity.tsv'.format(dataset, name), sep='\t', encoding='utf-8', index = False)

            test_similarity = similarity
            if 'ro' in todir:
                tokenizer = RobertaTokenizer.from_pretrained('./roberta_model')
            else:
                tokenizer = BertTokenizer.from_pretrained('./bert_model')

            test_data['similarity'] = [''] * len(test_a)
            ids = []
            # f = open('./datasets/quora/tmp/dev_um_similarity.txt', 'w', encoding='utf-8')
            for idx in range(len(test_a)):
                print(idx)
                text_a = test_a[idx]
                text_b = test_b[idx]
                if type(text_a) == float or type(text_b) == float or text_a == 'N/A' or \
                        text_a == 'N\A' or text_b == 'N/A' or text_b == 'N\A':
                    print(text_a, text_b)
                    test_data['similarity'][idx] = ''
                    ids.append(idx)
                    continue
                # pair of sequences: ``<s> A </s></s> B </s>``
                text_a = text_a.replace('\'', '\' ')
                text_b = text_b.replace('\'', '\' ')
                text_a = text_a.lower().split()
                # test['text_a'][idx] = text_a
                # text_a = text_a.split()
                text_b = text_b.lower().split()
                # test['text_b'][idx] = text_b
                # text_b = text_b.split()
                if 'ro' in todir:
                    text = ['<s>'] + text_a + ['</s>', '</s>'] + text_b + ['</s>']

                    tmp_sim = np.zeros((len(text), len(text)))
                    similarity = np.array(test_similarity[idx])
                    
                    tmp_sim[1:len(text_a) + 1, len(text_a) + 3:-1] = similarity
                    tmp_sim[len(text_a) + 3:-1, 1:len(text_a) + 1] = similarity.T
                else:
                    text = ['[CLS]'] + text_a + ['[SEP]'] + text_b + ['[SEP]']

                    tmp_sim = np.zeros((len(text), len(text)))
                    similarity = np.array(test_similarity[idx])
                    
                    tmp_sim[1:len(text_a) + 1, len(text_a) + 2:-1] = similarity
                    tmp_sim[len(text_a) + 2:-1, 1:len(text_a) + 1] = similarity.T

                tag_a = []
                for i in range(len(text)):
                    word = text[i]
                    token = tokenizer.tokenize(word)
                    tag = [i] * len(token)
                    tag_a = tag_a + tag

                tag = np.zeros((len(tag_a), len(tag_a)))
                for i in range(len(tag_a)):
                    for j in range(len(tag_a)):
                        tag[i][j] = tmp_sim[tag_a[i]][tag_a[j]]


                tag = tag.tolist()
                test_data['similarity'][idx] = tag

            test_data.drop(ids, inplace=True)
            data = test_data.reset_index(drop=True)

            data.to_csv(todir, sep='\t', index=None)
            print('end')
































