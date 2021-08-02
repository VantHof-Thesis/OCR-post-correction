#!/usr/bin/env python
# coding: utf-8

# In[1]:


# BERT

# do not forget to set the parameters topn_detection, topn_correction, and method

import pandas as pd
import numpy as np
import re
from transformers import BertTokenizer, BertForMaskedLM
import torch
from transformers import AdamW
from tqdm import tqdm
import gensim
from gensim.models import Word2Vec 
from gensim.models import KeyedVectors
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import pickle
import ast
import statistics as s
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
import nltk
from transformers import pipeline
import copy
from nltk.corpus import stopwords
import os
import subprocess
from time import sleep
from bs4 import BeautifulSoup
from bs4.dammit import EncodingDetector


# In[2]:


import torch
#transformers.__version__
torch.__version__


# In[3]:


# load word2vec model
#word2vec_model = Word2Vec.load("word2vec_finetuned.model")
# load BERT model
BERT_model = torch.load('BERT_finetuned.pt')
# load dataframe
df = pd.read_csv('df_5K.csv')

tokenizer = BertTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
#BERT_model = BertForMaskedLM.from_pretrained("GroNLP/bert-base-dutch-cased")
#w2v_model.intersect_word2vec_format(r"combined-160.txt", binary=False, lockf=1.0)
# https://github.com/clips/dutchembeddings

# skiplist (words that should not be corrected: names)
with open("ocr_names.txt", "rb") as fp:   # Unpickling
    ocr_names = pickle.load(fp)


# In[ ]:





# In[ ]:





# In[4]:


#with open('all_lists_tokens.txt', 'rb') as f:
#    all_lists_tokens = pickle.load(f)
    
#all_lists_tokens = ast.literal_eval(all_lists_tokens)
#vocab_BERT, vocab_word2vec, hist_expressions, modern_vocab, dictionary = all_lists_tokens

with open('homonyms.txt', 'rb') as f:
    homonyms = pickle.load(f)
with open('vocab_BERT', 'rb') as f:
    vocab_BERT = pickle.load(f)
with open('vocab_word2vec.txt', 'rb') as f:
    vocab_word2vec = pickle.load(f)
with open('hist_expressions.txt', 'rb') as f:
    hist_expressions = pickle.load(f)
with open('infrequent_expressions.txt', 'rb') as f:
    infrequent_expressions = pickle.load(f)
with open('dictionary.txt', 'rb') as f:
    dictionary = pickle.load(f)


# In[5]:


all_lists_tokens = [homonyms, vocab_BERT, vocab_word2vec, hist_expressions, infrequent_expressions, dictionary]


# In[6]:


# skiplist (words that should not be corrected: names)
with open("ocr_names.txt", "rb") as fp:   # Unpickling
    ocr_names = pickle.load(fp)

ocr_names = []
for name in ocr_names:
    if len(name) >= 5:
        ocr_names.append(name)


# In[7]:


def list_merger(lists):
    #normal_list = False
    #for elem in lists:
    #    if type(elem) != list:
    #        normal_list = True
    #if normal_list == True:
    #    return lists
    #else:
    new_list = []
    for elem in lists:
        new_list = new_list + elem
    return new_list


# In[8]:


def correct_sorted(candidates, sim_or_probs, LD): # sorts first by LD, then by similarity/probability
    paired_sorted = sorted(zip(LD,sim_or_probs,candidates),key = lambda x: (x[0],x[1]), reverse=True)
    LD,sim_or_probs,candidates = zip(*paired_sorted)
    correction = candidates[0]
    return correction
    
def correct_calculated(candidates, sim_or_probs, LD): # calculates a score from LD and normalised similarity/probability
    inv_LD = 1 - LD
    sim_or_probs = np.array(sim_or_probs)
    sim_or_probs = np.interp(sim_or_probs, (sim_or_probs.min(), sim_or_probs.max()), (0, 1)).tolist()
    score = sim_or_probs / inv_LD
    zipped_pairs = zip(score.tolist(), candidates)
    sorted_by_score = [x for _, x in sorted(zipped_pairs, reverse=True)]
    correction = sorted_by_score[0]
    return correction

def remove_stopwords(candidates, cosine, LD):
    #nltk.download('stopwords')
    stop_words = set(stopwords.words('dutch'))
    candidates_nostopwords = []
    cosine_nostopwords = []
    LD_nostopwords = []
    for i in range(len(candidates)):
        if candidates[i] not in stop_words:
            candidates_nostopwords.append(candidates[i])
            cosine_nostopwords.append(cosine[i])
            LD_nostopwords.append(LD[i])
    LD_nostopwords = np.array(LD_nostopwords)
    return candidates_nostopwords, cosine_nostopwords, LD_nostopwords


# In[9]:


# lists of all TP, FN, FP, TN detection:
homonyms_detection_list_BERT = [[],[],[],[]]
homonyms_detection_context_list_BERT = [[],[],[],[]]
histexp_detection_list_BERT = [[],[],[],[]]
histexp_detection_context_list_BERT = [[],[],[],[]]
OOV_detection_list_BERT = [[],[],[],[]]
OOV_detection_context_list_BERT = [[],[],[],[]]
infreq_detection_list_BERT = [[],[],[],[]]
infreq_detection_context_list_BERT = [[],[],[],[]]
RWE_detection_list_BERT = [[],[],[],[]]
RWE_detection_context_list_BERT = [[],[],[],[]]
all_detection_list_BERT = [[],[],[],[]]
none_detection_list_BERT = [[],[],[],[]]
none_detection_context_list_BERT = [[],[],[],[]]

# list of all right / wrong correction
homonyms_correction_list_BERT = [[],[]]
homonyms_correction_context_list_BERT = [[],[]]
histexp_correction_list_BERT = [[],[]]
histexp_correction_context_list_BERT = [[],[]]
OOV_correction_list_BERT = [[],[]]
OOV_correction_context_list_BERT = [[],[]]
infreq_correction_list_BERT = [[],[]]
infreq_correction_context_list_BERT = [[],[]]
RWE_correction_list_BERT = [[],[]]
RWE_correction_context_list_BERT = [[],[]]
all_correction_list_BERT = [[],[]]
none_correction_list_BERT = [[],[],[],[]]
none_correction_context_list_BERT = [[],[],[],[]]

#list of outputs corrected texts
new_documents = []

#list of improved and worsened
improved_all = []
worsened_all = []


# In[10]:



def calculate_result(predicted_error, actual_error):
    if actual_error == True:
        if predicted_error == True: # TP
            result = 'TP'
        if predicted_error == False: # FN
            result = 'FN'
    if actual_error == False:
        if predicted_error == True: # FP
            result = 'FP'
        if predicted_error == False: # TN
            result = 'TN'
    return result

def special_tokens_detection_word(ocr_word, gt_word, detection_list_BERT, all_lists_tokens, result): 
    homonyms, vocab_BERT, vocab_word2vec, hist_expressions, infrequent_expressions, dictionary = all_lists_tokens
    special_token = False
    homonym, hist_exp, OOV, infreq, RWE = False, False, False, False, False
    # check if word is homonym
    if gt_word in homonyms:
        homonym = True
        special_token = True
    # check if word is historical expression
    if gt_word in hist_expressions:
        hist_exp = True
        special_token = True
    # check if word is OOV
    if gt_word not in vocab_BERT:
        OOV = True
        special_token = True
    # check if word is infrequent
    if gt_word in infrequent_expressions:
        infreq = True
        special_token = True
    # check if word is RWE
    if (ocr_word in dictionary) and ((result == 'TP') or (result == 'FN')):
        RWE = True
        special_token = True
    # adding the results to the right list
    if result == 'TP': # TP = [0]
        # all = detection_lit[5]
        detection_list_BERT[5][0] += 1
        if homonym == True:  # homonyms = detection_list[0]
            detection_list_BERT[0][0] += 1
        if hist_exp == True: # hist_exp = detection_list[1]
            detection_list_BERT[1][0] += 1
        if OOV == True: # OOV = detection_list[2]
            detection_list_BERT[2][0] += 1
        if infreq == True: # infreq = detection_list[3]
            detection_list_BERT[3][0] += 1
        if RWE == True: # infreq = detection_list[4]
            detection_list_BERT[4][0] += 1
        if special_token == False: #none = detection_list[6]
            detection_list_BERT[6][0] += 1
    if result == 'FN': # FN = [1]
        # all = detection_lit[5]
        detection_list_BERT[5][1] += 1
        if homonym == True:  # homonyms = detection_list[0]
            detection_list_BERT[0][1] += 1
        if hist_exp == True: # hist_exp = detection_list[1]
            detection_list_BERT[1][1] += 1
        if OOV == True: # OOV = detection_list[2]
            detection_list_BERT[2][1] += 1
        if infreq == True: # infreq = detection_list[3]
            detection_list_BERT[3][1] += 1
        if RWE == True: # infreq = detection_list[4]
            detection_list_BERT[4][1] += 1
        if special_token == False: #none = detection_list[6]
            detection_list_BERT[6][1] += 1
    if result == 'FP': # FP = [2]
        # all = detection_lit[5]
        detection_list_BERT[5][2] += 1
        if homonym == True:  # homonyms = detection_list[0]
            detection_list_BERT[0][2] += 1
        if hist_exp == True: # hist_exp = detection_list[1]
            detection_list_BERT[1][2] += 1
        if OOV == True: # OOV = detection_list[2]
            detection_list_BERT[2][2] += 1
        if infreq == True: # infreq = detection_list[3]
            detection_list_BERT[3][2] += 1
        if RWE == True: # infreq = detection_list[4]
            detection_list_BERT[4][2] += 1
        if special_token == False: #none = detection_list[6]
            detection_list_BERT[6][2] += 1
    if result == 'TN': # TN = [3]
        # all = detection_list[5]
        detection_list_BERT[5][3] += 1
        if homonym == True:  # homonyms = detection_list[0]
            detection_list_BERT[0][3] += 1
        if hist_exp == True: # hist_exp = detection_list[1]
            detection_list_BERT[1][3] += 1
        if OOV == True: # OOV = detection_list[2]
            detection_list_BERT[2][3] += 1
        if infreq == True: # infreq = detection_list[3]
            detection_list_BERT[3][3] += 1
        if RWE == True: # infreq = detection_list[4]
            detection_list_BERT[4][3] += 1
        if special_token == False: #none = detection_list[6]
            detection_list_BERT[6][3] += 1
    return detection_list_BERT

def special_tokens_detection_context(ocr_context, gt_context, detection_list_context_BERT, all_lists_tokens, result): 
    homonyms, vocab_BERT, vocab_word2vec, hist_expressions, infrequent_expressions, dictionary = all_lists_tokens
    special_token = False
    homonym, hist_exp, OOV, infreq, RWE = False, False, False, False, False
    # check if context contains homonym
    homonym = False
    hist_exp = False
    OOV = False
    infreq = False
    RWE = False
    for word in gt_context:
        if word in homonyms:
            homonym = True
            special_token = True
        # check if word is historical expression
        if word in hist_expressions:
            hist_exp = True
            special_token = True
        # check if word is OOV
        if word not in vocab_BERT:
            OOV = True
            special_token = True
        # check if word is infrequent
        if word in infrequent_expressions:
            infreq = True
            special_token = True
        # check if word is RWE
        for i in range(len(ocr_context)):
            if (ocr_context[i] != gt_context[i]) and (ocr_context[i] in dictionary):
                RWE = True
                special_token = True
    # adding the results to the right list
    if result == 'TP': # TP = [0]
        if homonym == True:  # homonyms = detection_list[0]
            detection_list_context_BERT[0][0] += 1
        if hist_exp == True: # hist_exp = detection_list[1]
            detection_list_context_BERT[1][0] += 1
        if OOV == True: # OOV = detection_list[2]
            detection_list_context_BERT[2][0] += 1
        if infreq == True: # infreq = detection_list[3]
            detection_list_context_BERT[3][0] += 1
        if RWE == True: # infreq = detection_list[4]
            detection_list_context_BERT[4][0] += 1
        if special_token == False: #none = detection_list[6]
            detection_list_context_BERT[5][0] += 1
    if result == 'FN': # FN = [1]
        if homonym == True:  # homonyms = detection_list[0]
            detection_list_context_BERT[0][1] += 1
        if hist_exp == True: # hist_exp = detection_list[1]
            detection_list_context_BERT[1][1] += 1
        if OOV == True: # OOV = detection_list[2]
            detection_list_context_BERT[2][1] += 1
        if infreq == True: # infreq = detection_list[3]
            detection_list_context_BERT[3][1] += 1
        if RWE == True: # infreq = detection_list[4]
            detection_list_context_BERT[4][1] += 1
        if special_token == False: #none = detection_list[6]
            detection_list_context_BERT[5][1] += 1
    if result == 'FP': # FP = [2]
        if homonym == True:  # homonyms = detection_list[0]
            detection_list_context_BERT[0][2] += 1
        if hist_exp == True: # hist_exp = detection_list[1]
            detection_list_context_BERT[1][2] += 1
        if OOV == True: # OOV = detection_list[2]
            detection_list_context_BERT[2][2] += 1
        if infreq == True: # infreq = detection_list[3]
            detection_list_context_BERT[3][2] += 1
        if RWE == True: # infreq = detection_list[4]
            detection_list_context_BERT[4][2] += 1
        if special_token == False: #none = detection_list[6]
            detection_list_context_BERT[5][2] += 1
    if result == 'TN': # TN = [3]
        if homonym == True:  # homonyms = detection_list[0]
            detection_list_context_BERT[0][3] += 1
        if hist_exp == True: # hist_exp = detection_list[1]
            detection_list_context_BERT[1][3] += 1
        if OOV == True: # OOV = detection_list[2]
            detection_list_context_BERT[2][3] += 1
        if infreq == True: # infreq = detection_list[3]
            detection_list_context_BERT[3][3] += 1
        if RWE == True: # infreq = detection_list[4]
            detection_list_context_BERT[4][3] += 1
        if special_token == False: #none = detection_list[6]
            detection_list_context_BERT[5][3] += 1
    return detection_list_context_BERT
    

def special_tokens_correction_word(ocr_word, gt_word, correction_list_BERT, all_lists_tokens, result): 
    homonyms, vocab_BERT, vocab_word2vec, hist_expressions, infrequent_expressions, dictionary = all_lists_tokens
    special_token = False
    homonym, hist_exp, OOV, infreq, RWE = False, False, False, False, False
    # check if word is homonym
    if gt_word in homonyms:
        homonym = True
        special_token = True
    # check if word is historical expression
    if gt_word in hist_expressions:
        hist_exp = True
        special_token = True
    # check if word is OOV
    if gt_word not in vocab_BERT:
        OOV = True
        special_token = True
    # check if word is infrequent
    if gt_word in infrequent_expressions:
        infreq = True
        special_token = True
    # check if word is RWE
    if ocr_word in dictionary:
        RWE = True
        special_token = True
    # adding the results to the right list
    if result == 'right': # wrong = [0]
        # all = detection_lit[5]
        correction_list_BERT[5][0] += 1
        if homonym == True:  # homonyms = detection_list[0]
            correction_list_BERT[0][0] += 1
        if hist_exp == True: # hist_exp = detection_list[1]
            correction_list_BERT[1][0] += 1
        if OOV == True: # OOV = detection_list[2]
            correction_list_BERT[2][0] += 1
        if infreq == True: # infreq = detection_list[3]
            correction_list_BERT[3][0] += 1
        if RWE == True: # infreq = detection_list[4]
            correction_list_BERT[4][0] += 1
        if special_token == False: #none = detection_list[6]
            correction_list_BERT[6][0] += 1
    if result == 'wrong': # right = [1]
        # all = detection_lit[5]
        correction_list_BERT[5][1] += 1
        if homonym == True:  # homonyms = detection_list[0]
            correction_list_BERT[0][1] += 1
        if hist_exp == True: # hist_exp = detection_list[1]
            correction_list_BERT[1][1] += 1
        if OOV == True: # OOV = detection_list[2]
            correction_list_BERT[2][1] += 1
        if infreq == True: # infreq = detection_list[3]
            correction_list_BERT[3][1] += 1
        if RWE == True: # infreq = detection_list[4]
            correction_list_BERT[4][1] += 1
        if special_token == False: #none = detection_list[6]
            correction_list_BERT[6][1] += 1
    return correction_list_BERT

def special_tokens_correction_context(ocr_context, gt_context, correction_list_context_BERT, all_lists_tokens, result): 
    homonyms, vocab_BERT, vocab_word2vec, hist_expressions, infrequent_expressions, dictionary = all_lists_tokens
    special_token = False
    homonym, hist_exp, OOV, infreq, RWE = False, False, False, False, False
    # check if context contains homonym
    homonym = False
    hist_exp = False
    OOV = False
    infreq = False
    RWE = False
    for word in gt_context:
        if word in homonyms:
            homonym = True
            special_token = True
        # check if word is historical expression
        if word in hist_expressions:
            hist_exp = True
            special_token = True
        # check if word is OOV
        if word not in vocab_BERT:
            OOV = True
            special_token = True
        # check if word is infrequent
        if word in infrequent_expressions:
            infreq = True
            special_token = True
        # check if word is RWE
        for i in range(len(ocr_context)):
            if (ocr_context[i] != gt_context[i]) and (ocr_context[i] in dictionary):
                RWE = True
                special_token = True
    # adding the results to the right list
    if result == 'right': # right = [0]
        if homonym == True:  # homonyms = detection_list[0]
            correction_list_context_BERT[0][0] += 1
        if hist_exp == True: # hist_exp = detection_list[1]
            correction_list_context_BERT[1][0] += 1
        if OOV == True: # OOV = detection_list[2]
            correction_list_context_BERT[2][0] += 1
        if infreq == True: # infreq = detection_list[3]
            correction_list_context_BERT[3][0] += 1
        if RWE == True: # infreq = detection_list[4]
            correction_list_context_BERT[4][0] += 1
        if special_token == False: #none = detection_list[6]
            correction_list_context_BERT[5][0] += 1
    if result == 'wrong': # wrong = [1]
        if homonym == True:  # homonyms = detection_list[0]
            correction_list_context_BERT[0][1] += 1
        if hist_exp == True: # hist_exp = detection_list[1]
            correction_list_context_BERT[1][1] += 1
        if OOV == True: # OOV = detection_list[2]
            correction_list_context_BERT[2][1] += 1
        if infreq == True: # infreq = detection_list[3]
            correction_list_context_BERT[3][1] += 1
        if RWE == True: # infreq = detection_list[4]
            correction_list_context_BERT[4][1] += 1
        if special_token == False: #none = detection_list[6]
            correction_list_context_BERT[5][1] += 1
    return correction_list_context_BERT
    


# In[20]:


new_documents = []
#detection test BERT
def detection_and_correction_BERT(row, BERT_model, ocr_names,  all_lists_token, topn_detection=500, topn_correction=500, correction_method = 'sorted'):  # choose 'sorted'/ 'sorted_nosw', 'calculated'
    if row['set'] != 'test':
        return np.nan
    else:
        biggest_param = max(topn_detection, topn_correction)
        identifier = row['identifier']
        OCR_text = row['aligned_OCR_sentences']
        GT_text = row['aligned_GT_sentences']
        OCR_text = ast.literal_eval(OCR_text)
        GT_text = ast.literal_eval(GT_text)

        # keep track of performance detection
        homonyms_detection_BERT = [0,0,0,0]
        homonyms_detection_context_BERT = [0,0,0,0]
        histexp_detection_BERT = [0,0,0,0]
        histexp_detection_context_BERT = [0,0,0,0]
        OOV_detection_BERT = [0,0,0,0]
        OOV_detection_context_BERT = [0,0,0,0]
        infreq_detection_BERT = [0,0,0,0]
        infreq_detection_context_BERT = [0,0,0,0]
        RWE_detection_BERT = [0,0,0,0]
        RWE_detection_context_BERT = [0,0,0,0]
        all_detection_BERT = [0,0,0,0]
        none_detection_BERT = [0,0,0,0]
        none_detection_context_BERT = [0,0,0,0]
        
        # keep track of performance correction right / wrong
        homonyms_correction_BERT = [0,0]
        homonyms_correction_context_BERT = [0,0]
        histexp_correction_BERT = [0,0]
        histexp_correction_context_BERT = [0,0]
        OOV_correction_BERT = [0,0]
        OOV_correction_context_BERT = [0,0]
        infreq_correction_BERT = [0,0]
        infreq_correction_context_BERT = [0,0]
        RWE_correction_BERT = [0,0]
        RWE_correction_context_BERT = [0,0]
        all_correction_BERT = [0,0]
        none_correction_BERT = [0,0]
        none_correction_context_BERT = [0,0]
        
        # create lists that save evaluation scores for this documents
        detection_list_BERT = [homonyms_detection_BERT, histexp_detection_BERT, OOV_detection_BERT, infreq_detection_BERT, RWE_detection_BERT, all_detection_BERT, none_detection_BERT]
        detection_list_context_BERT = [homonyms_detection_context_BERT, histexp_detection_context_BERT, OOV_detection_context_BERT, infreq_detection_context_BERT, RWE_detection_context_BERT, none_detection_context_BERT]
        correction_list_BERT = [homonyms_correction_BERT, histexp_correction_BERT, OOV_correction_BERT, infreq_correction_BERT, RWE_correction_BERT, all_correction_BERT, none_correction_BERT]
        correction_list_context_BERT = [homonyms_correction_context_BERT, histexp_correction_context_BERT, OOV_correction_context_BERT, infreq_correction_context_BERT, RWE_correction_context_BERT, none_correction_context_BERT]
        
        improved = 0 # when actual error is detected, and corrected rightly
        worsened  = 0 # when actual non error is wrongfully detected, and corrected wrongly
        
        # create corrected file
        new_document = []
        for s in range(len(OCR_text)): # for each sentence
            for i in range(len(OCR_text[s])): # for each word in a sentence
                if (OCR_text[s][i] in ocr_names) or (OCR_text[s][i].isalpha() == False) or (len(OCR_text[s][i]) <= 2)  or (GT_text[s][i] == 'REMOVED'):
                    # add word to document if left unchanged
                    #print('skipped')
                    new_document.append(OCR_text[s][i])
                    continue
                sentence = copy.deepcopy(OCR_text[s])
                gt_sentence = copy.deepcopy(GT_text[s])
                sentence[i] = '[MASK]'
                context = copy.deepcopy(sentence)
                del context[i]
                sentence = ' '.join(sentence)
                GT_context = gt_sentence
                del GT_context[i]
                for t in range(len(context)):
                    if any(str.isdigit(c) for c in context[t]) == True:
                        context[t] = '%NUMBER%'
                    elif context[t] in ocr_names:
                        context[t] = '%NNP%'
                # generate list of all candidates
                whole_list_candidates = []
                whole_list_probabilities = []
                pipe = pipeline('fill-mask', model=BERT_model, tokenizer = tokenizer, top_k=biggest_param, device = 0)
                for res in pipe(sentence):
                    whole_list_candidates.append(res['token_str'].replace(' ', ''))
                    whole_list_probabilities.append(res['score'])
                # remove punctuation except for hyphen from candidates
                whole_list_candidates = [re.sub(r'[^\w\d\s\-]+', '', x) for x in whole_list_candidates]
                whole_list_candidates = [x.lower() for x in whole_list_candidates]
                # score down for detection task
                candidates = copy.deepcopy(whole_list_candidates[:topn_detection])
                probabilities = copy.deepcopy(whole_list_probabilities[:topn_detection])
                #calculate positions detection task
                # determine if token is predicted error or not
                if OCR_text[s][i] in candidates:
                    predicted_error = False
                elif OCR_text[s][i] not in candidates:
                    predicted_error = True
                # determine if token is actual error or not
                if OCR_text[s][i] != GT_text[s][i]:
                    actual_error = True
                elif OCR_text[s][i] == GT_text[s][i]:
                    actual_error = False
                result_det = calculate_result(predicted_error, actual_error)
                # evaluate detection
                detection_list_BERT = special_tokens_detection_word(OCR_text[s][i], GT_text[s][i], detection_list_BERT, all_lists_token, result_det)
                detection_context_list_BERT = special_tokens_detection_context(context, GT_context, detection_list_context_BERT, all_lists_tokens, result_det)
                # return detection evaluation values:


                # place old detection evaluation

                # correction evaluation
                if actual_error == True:
                    candidates = copy.deepcopy(whole_list_candidates[:topn_correction])
                    probabilities = copy.deepcopy(whole_list_probabilities[:topn_correction])
                    # calculate positions detection task
                    # try two correction methods
                    # first calculate the normalized LDs:
                    LD = np.array([fuzz.ratio(OCR_text[s][i], word)/100 for word in candidates])
                    # try sorting method
                    if correction_method == 'sorted':
                        correction = correct_sorted(candidates, probabilities, LD)
                    elif correction_method == 'sorted_nosw':
                    # try again the sorting methods, but without stopwords
                        candidates_nostopwords, cosine_nostopwords, LD_nostopwords = remove_stopwords(candidates, probabilities, LD)
                        correction = correct_sorted(candidates_nostopwords, cosine_nostopwords, LD_nostopwords)
                    # try score calculation method
                    elif correction_method == 'calculated':
                        correction = correct_calculated(candidates, probabilities, LD)
                    # evaluation
                    if correction == GT_text[s][i]:
                        result_cor = 'right'
                    elif correction != GT_text[s][i]:
                        result_cor = 'wrong'
                    correction_list_BERT = special_tokens_correction_word(OCR_text[s][i], GT_text[s][i], detection_list_BERT, all_lists_token, result_cor)
                    correction_context_list_BERT = special_tokens_correction_context(context, GT_context, correction_list_context_BERT, all_lists_tokens, result_cor)

                    # place old correction evaluation

                # perform whole task
                # first, add OCR-word to file if skipped (see above)
                # add word to document if not detected as an error
                if predicted_error == False:
                    new_document.append(OCR_text[s][i])
                    continue
                # if predicted to be an error, perform correction:
                if actual_error == True:
                    correction = correction # correction was already created
                elif actual_error == False:
                    candidates = copy.deepcopy(whole_list_candidates[:topn_detection])
                    probabilities = copy.deepcopy(whole_list_probabilities[:topn_detection])
                # first calculate the normalized LDs:
                LD = np.array([fuzz.ratio(OCR_text[s][i], word)/100 for word in candidates])
                # try sorting method
                if correction_method == 'sorted':
                    correction = correct_sorted(candidates, probabilities, LD)
                elif correction_method == 'sorted_nosw':
                # try again the sorting methods, but without stopwords
                    candidates_nostopwords, cosine_nostopwords, LD_nostopwords = remove_stopwords(candidates, probabilities, LD)
                    correction = correct_sorted(candidates_nostopwords, cosine_nostopwords, LD_nostopwords)
                # try score calculation method
                elif correction_method == 'calculated':
                    correction = correct_calculated(candidates, probabilities, LD)
                if correction == GT_text[s][i]:
                        result_cor = 'right'
                elif correction != GT_text[s][i]:
                        result_cor = 'wrong'
                #print(GT_text[s][i])
                #print(correction)
                new_document.append(correction)

                if (result_det == 'TP') and (result_cor == 'right'):
                    improved += 1
                elif (result_det == 'FP') and (result_cor == 'wrong'):
                    worsened += 1
                
        improved_all.append(improved)
        worsened_all.append(worsened)
            
        new_document = (' ').join(new_document)
        new_document = re.sub(' +', ' ', new_document)
        new_documents.append(new_document)
        
        
        for k in range(len(detection_list_BERT[0])): # for each result: 0 = TP, 1 = TN, 2 = FP, 3 = TN
                # homonyms = index 0 in detection_list_BERT    
                homonyms_detection_list_BERT[k].append(detection_list_BERT[0][k])
                # hist_exp = index 1
                histexp_detection_list_BERT[k].append(detection_list_BERT[1][k])
                # OOV = index 2
                OOV_detection_list_BERT[k].append(detection_list_BERT[2][k])
                # infreq = index 3
                infreq_detection_list_BERT[k].append(detection_list_BERT[3][k])
                # RWE = index 4
                RWE_detection_list_BERT[k].append(detection_list_BERT[4][k])
                # all = index 5
                all_detection_list_BERT[k].append(detection_list_BERT[5][k])
                # non = index 6
                none_detection_list_BERT[k].append(detection_list_BERT[6][k])
        for k in range(len(detection_list_context_BERT[0])): # for each result: 0 = TP, 1 = TN, 2 = FP, 3 = TN
                # homonyms = index 0 in detection_list_BERT    
                homonyms_detection_context_list_BERT[k].append(detection_context_list_BERT[0][k])
                # hist_exp = index 1
                histexp_detection_context_list_BERT[k].append(detection_context_list_BERT[1][k])
                # OOV = index 2
                OOV_detection_context_list_BERT[k].append(detection_context_list_BERT[2][k])
                # infreq = index 3
                infreq_detection_context_list_BERT[k].append(detection_context_list_BERT[3][k])
                # RWE = index 4
                RWE_detection_context_list_BERT[k].append(detection_context_list_BERT[4][k])
                # non = index 5
                none_detection_context_list_BERT[k].append(detection_context_list_BERT[5][k])
    
        
        # return correction evaluation values:
        for k in range(2): # for each result: 0 = right, 1 = wrong
                # homonyms = index 0 in detection_list_BERT    
                homonyms_correction_list_BERT[k].append(correction_list_BERT[0][k])
                # hist_exp = index 1
                histexp_correction_list_BERT[k].append(correction_list_BERT[1][k])
                # OOV = index 2
                OOV_correction_list_BERT[k].append(correction_list_BERT[2][k])
                # infreq = index 3
                infreq_correction_list_BERT[k].append(correction_list_BERT[3][k])
                # RWE = index 4
                RWE_correction_list_BERT[k].append(correction_list_BERT[4][k])
                # all = index 5
                all_correction_list_BERT[k].append(correction_list_BERT[5][k])
                # non = index 6
                none_correction_list_BERT[k].append(correction_list_BERT[6][k])
        for k in range(2): # for each result: 0 = right, 1 = wrong
                # homonyms = index 0 in detection_list_BERT    
                homonyms_correction_context_list_BERT[k].append(correction_context_list_BERT[0][k])
                # hist_exp = index 1
                histexp_correction_context_list_BERT[k].append(correction_context_list_BERT[1][k])
                # OOV = index 2
                OOV_correction_context_list_BERT[k].append(correction_context_list_BERT[2][k])
                # infreq = index 3
                infreq_correction_context_list_BERT[k].append(correction_context_list_BERT[3][k])
                # RWE = index 4
                RWE_correction_context_list_BERT[k].append(correction_context_list_BERT[4][k])
                # non = index 5
                none_correction_context_list_BERT[k].append(correction_context_list_BERT[5][k])
        
        
#for index, row in df.iterrows():
#    detection_word2vec(row)
# df.loc[70]
fake_test_list_GT_aligned = """12 Een koekenpan of kortweg 879 pan is een platte pan met een lang handvat.
De pan ontleent zijn naam aan het feit dat in zo'n pan 12 pannenkoeken worden gebakken. Ook ander voedsel, zoals vlees, wordt in een koekenpan gebraden 12 coninghs-merck"""
fake_test_list_OCR_aligned = """12 Een hoekenpan of kortweg 879 pan is een platte pan met een hang handvat.
De pan ontleent zijn naam haan het feit dat in zo'n pan 12 pannenkoeken horden gebakken. Ook ander voedsel, zoals vlees, word in een hoekenpan gebraden 12 coninghs-merck"""
fake_test_list_GT_aligned = fake_test_list_GT_aligned.split('.')
fake_test_list_OCR_aligned = fake_test_list_OCR_aligned.split('.')
fake_test_list_GT_aligned = [x.split(' ') for x in fake_test_list_GT_aligned]
fake_test_list_OCR_aligned = [x.split(' ') for x in fake_test_list_OCR_aligned]
d = {'identifier': ['111'], 'aligned_OCR_sentences': [str(fake_test_list_OCR_aligned)], 'aligned_GT_sentences': [str(fake_test_list_GT_aligned)], 'set': ['test'], 'century': ['1600s'], 'source': ['Meertens']}

#df_probeer = pd.DataFrame(data=d)
#for index, row in df.loc[df['set'].isin(['test'])].iterrows():
for index, row in df.iterrows():
    if index%1000 == 0:
        print(index)
    detection_and_correction_BERT(row, BERT_model, ocr_names, all_lists_tokens)  # choose 'sorted'/


# In[ ]:


#print(new_documents)


# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


#print(homonyms_detection_list_BERT)
#print(homonyms_detection_context_list_BERT)
#print(histexp_detection_list_BERT)
#print(histexp_detection_context_list_BERT)
#print(OOV_detection_list_BERT)
#print(OOV_detection_context_list_BERT)
#print(infreq_detection_list_BERT)
#print(infreq_detection_context_list_BERT)
#print(RWE_detection_list_BERT)
#print(RWE_detection_context_list_BERT)
#print(all_detection_list_BERT)
#print(none_detection_list_BERT)
#print(none_detection_context_list_BERT)


# In[ ]:





# In[17]:


d = {'homonyms_detection TP': homonyms_detection_list_BERT[0], 'homonyms_detection FN': homonyms_detection_list_BERT[1], 'homonyms_detection FP': homonyms_detection_list_BERT[2], 'homonyms_detection TN': homonyms_detection_list_BERT[3],     'homonyms_detection context TP': homonyms_detection_context_list_BERT[0], 'homonyms_detection context FN': homonyms_detection_context_list_BERT[1], 'homonyms_detection context FP': homonyms_detection_context_list_BERT[2], 'homonyms_detection context TN': homonyms_detection_context_list_BERT[3],     'histexp_detection TP': histexp_detection_list_BERT[0], 'histexp_detection FN': histexp_detection_list_BERT[1], 'histexp_detection FP': histexp_detection_list_BERT[2], 'histexp_detection TN': histexp_detection_list_BERT[3],     'histexp_detection context TP': histexp_detection_context_list_BERT[0], 'histexp_detection context FN': histexp_detection_context_list_BERT[1], 'histexp_detection context FP': histexp_detection_context_list_BERT[2], 'histexp_detection context TN': histexp_detection_context_list_BERT[3],     'OOV_detection TP': OOV_detection_list_BERT[0], 'OOV_detection FN': OOV_detection_list_BERT[1], 'OOV_detection FP': OOV_detection_list_BERT[2], 'OOV_detection TN': OOV_detection_list_BERT[3],     'OOV_detection context TP': OOV_detection_context_list_BERT[0], 'OOV_detection context FN': OOV_detection_context_list_BERT[1], 'OOV_detection context FP': OOV_detection_context_list_BERT[2], 'OOV_detection context TN': OOV_detection_context_list_BERT[3],     'infreq_detection TP': infreq_detection_list_BERT[0], 'infreq_detection FN': infreq_detection_list_BERT[1], 'infreq_detection FP': infreq_detection_list_BERT[2], 'infreq_detection TN': infreq_detection_list_BERT[3],     'infreq_detection context TP': infreq_detection_context_list_BERT[0], 'infreq_detection context FN': infreq_detection_context_list_BERT[1], 'infreq_detection context FP': infreq_detection_context_list_BERT[2], 'infreq_detection context TN': infreq_detection_context_list_BERT[3],     'RWE_detection TP': RWE_detection_list_BERT[0], 'RWE_detection FN': RWE_detection_list_BERT[1], 'RWE_detection FP': RWE_detection_list_BERT[2], 'RWE_detection TN': RWE_detection_list_BERT[3],     'RWE_detection context TP': RWE_detection_context_list_BERT[0], 'RWE_detection context FN': RWE_detection_context_list_BERT[1], 'RWE_detection context FP': RWE_detection_context_list_BERT[2], 'RWE_detection context TN': RWE_detection_context_list_BERT[3],     'all_detection TP': all_detection_list_BERT[0], 'all_detection FN': all_detection_list_BERT[1], 'all_detection FP': all_detection_list_BERT[2], 'all_detection TN': all_detection_list_BERT[3],     'none_detection TP': none_detection_list_BERT[0], 'none_detection FN': none_detection_list_BERT[1], 'none_detection FP': none_detection_list_BERT[2], 'none_detection TN': none_detection_list_BERT[3],     'none_detection context TP': none_detection_context_list_BERT[0], 'none_detection context FN': none_detection_context_list_BERT[1], 'none_detection context FP': none_detection_context_list_BERT[2], 'none_detection context TN': none_detection_context_list_BERT[3],     'identifier': list(df[df["set"] == 'test']['identifier']), 'century': list(df[df["set"] == 'test']['century']), 'source': list(df[df["set"] == 'test']['source'])  }
BERT_detection = pd.DataFrame(data=d)

#BERT_detection


# In[ ]:


d = {'homonyms_correction right': homonyms_correction_list_BERT[0], 'homonyms_correction wrong': homonyms_correction_list_BERT[1],    'homonyms_correction context right': homonyms_correction_context_list_BERT[0], 'homonyms_correction context wrong': homonyms_correction_context_list_BERT[1],     'histexp_correction right': histexp_correction_list_BERT[0], 'histexp_correction wrong': histexp_correction_list_BERT[1],     'histexp_correction context right': histexp_correction_context_list_BERT[0], 'histexp_correction context wrong': histexp_correction_context_list_BERT[1],     'OOV_correction right': OOV_correction_list_BERT[0], 'OOV_correction wrong': OOV_correction_list_BERT[1],    'OOV_correction context right': OOV_correction_context_list_BERT[0], 'OOV_correction context wrong': OOV_correction_context_list_BERT[1],    'infreq_correction right': infreq_correction_list_BERT[0], 'infreq_correction wrong': infreq_correction_list_BERT[1],    'infreq_correction context right': infreq_correction_context_list_BERT[0], 'infreq_correction context wrong': infreq_correction_context_list_BERT[1],     'RWE_correction right': RWE_correction_list_BERT[0], 'RWE_correction wrong': RWE_correction_list_BERT[1],    'RWE_correction context right': RWE_correction_context_list_BERT[0], 'RWE_correction context wrong': RWE_correction_context_list_BERT[1],    'all_correction right': all_correction_list_BERT[0], 'all_correction wrong': all_correction_list_BERT[1],    'none_correction right': none_correction_list_BERT[0], 'none_correction wrong': none_correction_list_BERT[1],    'none_correction context right': none_correction_context_list_BERT[0], 'none_correction context wrong': none_correction_context_list_BERT[1],      'identifier': list(df[df["set"] == 'test']['identifier']), 'century': list(df[df["set"] == 'test']['century']), 'source': list(df[df["set"] == 'test']['source'])}
BERT_correction = pd.DataFrame(data=d)

#BERT_correction


# In[ ]:


detection_categories_BERT = "homonyms_detection_BERT, histexp_detection_BERT, OOV_detection_BERT, infreq_detection_BERT, RWE_detection_BERT, all_detection_BERT, none_detection_BERT, homonyms_detection context_BERT, histexp_detection context_BERT, OOV_detection context_BERT, infreq_detection context_BERT, RWE_detection context_BERT, none_detection context_BERT".replace('_BERT', '').split(', ')

for category in detection_categories_BERT:
    precisions = []
    recalls = []
    F1s = []
    accuracies = []

    def calc_scores(row, category):
        TP, FN, FP, TN = int(row[f'{category} TP']), int(row[f'{category} FN']),  int(row[f'{category} FP']),  int(row[f'{category} TN']),    
        try:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1 = 2*((precision*recall)/(precision+recall))
        except ZeroDivisionError:
            if (TP == 0) and (FP == 0) and (FN == 0):
                precision = recall = F1 = 1
            elif (TP == 0) and ((FP > 0) or (FN > 0)):
                precision = recall = F1 = 0 
        try:
            accuracy = (TP + TN)/(TP + TN + FP + FN)
        except ZeroDivisionError:
            accuracy = np.nan
        
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(F1)
        accuracies.append(accuracy)
    
    for index, row in BERT_detection.iterrows():
        calc_scores(row, category)
    
    BERT_detection[f'{category} precision'] = precisions
    BERT_detection[f'{category} recall'] = recalls
    BERT_detection[f'{category} F1'] = F1s
    BERT_detection[f'{category} accuracy'] = accuracies


# In[ ]:


correction_categories_BERT = "homonyms_correction_BERT, histexp_correction_BERT, OOV_correction_BERT, infreq_correction_BERT, RWE_correction_BERT, all_correction_BERT, none_correction_BERT, homonyms_correction context_BERT, histexp_correction context_BERT, OOV_correction context_BERT, infreq_correction context_BERT, RWE_correction context_BERT, none_correction context_BERT".replace('_BERT', '').split(', ')

for category in correction_categories_BERT:
    right, wrong = np.array(BERT_correction[f'{category} right']), np.array(BERT_correction[f'{category} wrong'])    
    #try:
    accuracy = right/(right+wrong)
    #except ZeroDivisionError:
    #    accuracy = np.nan*len(w2v_correction)
    BERT_correction[f'{category} accuracy'] = accuracy
    BERT_correction[f'{category} total'] = right + wrong


# In[ ]:


pd.set_option('display.max_columns', None)
#BERT_correction


# In[ ]:


gt_orgs = list(df[df["set"] == 'test']['gt text'])
WER_orgs = list(df[df["set"] == 'test']['WER matched sentences'])
CER_orgs = list(df[df["set"] == 'test']['CER matched sentences'])


# In[ ]:





# In[ ]:


#test_df = BERT_detection.filter(regex='homonyms|OOV|all').columns


# In[ ]:


d = {'corrected document': new_documents, 'gt text': gt_orgs,'identifier': list(df[df["set"] == 'test']['identifier']), 'century': list(df[df["set"] == 'test']['century']), 'source': list(df[df["set"] == 'test']['source']),     'improved': improved_all, 'worsened': worsened_all, 'old WER': WER_orgs, 'old CER': CER_orgs}
whole_task_BERT = pd.DataFrame(data=d)


# In[ ]:





# In[ ]:





# In[ ]:


jar_file = "ocrevalUAtion-1.3.4-jar-with-dependencies.jar"

def evaluation(index, row):
    ID = row['identifier']
    page = 'None'
    corrected_OCR = re.sub(' +', ' ', str(row['corrected document'].replace('.', '')))
    gt_text = re.sub(' +', ' ', str(row['gt text'].replace('.', '')))
    filename_ocr = f"{ID}_{page}_OCR.txt"
    #file_ocr = open(os.path.join(save_path, filename),"w+", encoding="utf-8")
    file_ocr = open(filename_ocr,"w+", encoding="utf-8")
    file_ocr.write(corrected_OCR)
    file_ocr.close()
    
    filename_gt = f"{ID}_{page}_GT.txt"
    #file_gt = open(os.path.join(save_path, filename),"w+", encoding="utf-8")
    file_gt = open(filename_gt,"w+", encoding="utf-8")
    file_gt.write(gt_text)
    file_gt.close()
    
    #output = ID + '_' + page + ".html"
    output = f"{ID}_{page}.html"
    
    #process = subprocess.call("/home/nvanthof/jdk-16.0.1/bin/java -cp " + jar_file  + " eu.digitisation.Main -gt " + filename_gt + " -ocr "+ filename_ocr +" -o " + output + "")
    #os.system("/home/nvanthof/jdk-16.0.1/bin/java -cp /home/nvanthof/ocrevalUAtion-1.3.4-jar-with-dependencies.jar eu.digitisation.Main -gt /home/nvanthof/ddd.010728187.mpeg21.a0005_None_GT.txt -ocr /home/nvanthof/ddd.010728187.mpeg21.a0005_None_OCR.txt  -o /home/nvanthof/OUTPUT2.html")
    #command = f"/home/nvanthof/jdk-16.0.1/bin/java -cp /home/nvanthof/ocrevalUAtion-1.3.4-jar-with-dependencies.jar eu.digitisation.Main -gt /home/nvanthof/{filename_gt} -ocr /home/nvanthof/{filename_ocr}  -o /home/nvanthof/{output}"
    command = f"/usr/bin/java -cp /home/nynkegpu/ocrevalUAtion-1.3.4-jar-with-dependencies.jar eu.digitisation.Main -gt /home/nynkegpu/{filename_gt} -ocr /home/nynkegpu/{filename_ocr}  -o /home/nynkegpu/{output}"
    os.system(command)
    sleep(5)
    
    soup = BeautifulSoup(open(output, encoding='utf-8'))
    table = soup.find("table", attrs={'border': '1'})
    # Split the filename, and extract the identifier and pagenr together as identifier 
    # Find the first table (this is the table in which the scores are stored)
    # Find the tags in which 'CER', 'WER', and 'WER (order independent)' are stored and take the next tag to get the score 
    cer = table.find('td', text='CER')
    cerScore = cer.findNext('td')
    wer = table.find('td', text='WER')
    werScore = wer.findNext('td')
    werOI = table.find('td', text='WER (order independent)')
    werOIScore = werOI.findNext('td')
    
    os.remove(filename_gt)
    os.remove(filename_ocr)
    os.remove(output)
    return float(cerScore.text), float(werScore.text)   
    
    return cerScore.text, werScore.text

for index, row in whole_task_BERT.iterrows():
    if index%1000 == 0:
        print(index)
    whole_task_BERT.at[index, 'CER after correction'], whole_task_BERT.at[index, 'WER after correction'] = evaluation(index, row)
    


# In[ ]:


whole_task_BERT['WER reduced'] = whole_task_BERT['old WER'] - whole_task_BERT['WER after correction']
whole_task_BERT['CER reduced'] = whole_task_BERT['old CER'] - whole_task_BERT['CER after correction']


# In[ ]:


#whole_task_BERT


# In[ ]:


# save dataframes
# detection dataframe
BERT_detection.to_csv('detection_test_5K_BERT_sorted.csv')
# correction dataframe
BERT_correction.to_csv('correction_test_5K_BERT_sorted.csv')
# whole task dataframe
whole_task_BERT.to_csv('whole_task_test_5K_BERT_sorted.csv')


# In[ ]:





# In[ ]:




