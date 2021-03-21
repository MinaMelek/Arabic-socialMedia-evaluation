# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 09:52:47 2021

@author: Mina.Melek
"""

# ==== Helper Methods =====
import os
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.stem.isri import ISRIStemmer # arabic stemming
# from qalsadi import lemmatizer
from tashaphyne.stemming import ArabicLightStemmer
from pyarabic.araby import strip_tashkeel
import contractions
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Dropout
from gensim.models.callbacks import CallbackAny2Vec

st = ISRIStemmer()
# lem = lemmatizer.Lemmatizer()
stm = ArabicLightStemmer()
nltk.download('stopwords')
contractions.add("I'd been", "I had been")

# -- specific data cleaning
# remove repeated sentences in a single text
def reduce_sent(x):
    while True:
        d = len(x)
        x = re.sub(r'\b([\w\s]+)( \1\b)+', r'\1', x)
        if len(x) == d:
            return x
            
def remove_longation(text, prefix=False):
    # English
    text = contractions.fix(text)
    text = text.lower()
    # remove tashkeel
    text = strip_tashkeel(text)
    # remove repeated
    text = re.sub(r'(.)\1+', r'\1\1', text)
    text = text.replace('اا', 'ا').replace('وو', 'و').replace('يي', 'ي').replace('رر', 'ر')
    text = re.sub(r'((?<=\A)|(?<=\s))هه(?=\s|\Z)', r'ضحك', text)
    text = re.sub(r'ـ', '', text)
    # remove punctuations
    text = re.sub(r'[^\w\sؠ-ي١-٩]', ' ', text) # replace \w with \d to filter english letters
    text = re.sub(r'_', ' ', text)
    # unifying similar letters
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub('علے', 'على ', text)
    text = re.sub(r'ے', '', text)
    text = re.sub("[ؠىي]", "ي", text)
    # text = re.sub("يئ", "يء", text)
    text = re.sub("[ئؽؾؿ]", "ئ", text)# ء
    # text = re.sub("ؤ", "ء", text)
    # text = re.sub("ة", "ه", text)
    text = re.sub("[ؼػگ]", "ك", text)
    # separate negation signs
    text = re.sub(r'\bما((\w+))ش\b', r'ما \1', text)
    text = re.sub(r'((?<=\A)|(?<=\s)|(?<=و))مش\b', r' ما', text)
    text = re.sub(r'((?<=\A)|(?<=\s)|(?<=و))ليش\b', r' ليه', text)
    text = re.sub(r'((?<=\A)|(?<=\s)|(?<=و))يا(?=\w)', r' يا ', text)
    text = re.sub('شاءالله', ' شاء الله', text)
    text = re.sub('شا الله', ' شاء الله', text)
    # remove slang begin marks
    text = re.sub(r'((?<=\A)|(?<=\s))ه(?!نال?ك( |\Z))(?=[ايتن]\w{2,})', '', text)
    # text = re.sub(r'((?<=\A)|(?<=\s))ب(?=[ايت])', '', text)
    # text = re.sub(r'((?<=\A)|(?<=\s))ا(?=[يت])', '', text)
    # remove repeated sentences in a single text
    # text = reduce_sent(text) #TODO long time
    # remove prefix
    if prefix:
        text = re.sub('((?<=\s)|(?<=\A))(\w?ال)(?=\w{3,})', '', text)
        text = re.sub('((?<=\s)|(?<=\A))(لل)(?=\w{3,})', '', text)
    return " ".join(text.split())

# preprocessing functions
def get_stopwords():
    cachedStopWords = stopwords.words("arabic") + st.stop_words
    [cachedStopWords.remove(e) for e in ['ما', 'كم', 'كم', 'بكم', 'بكم', 'مساء', 'امسى', 'أمسى', 'كانت']]
    # print(cachedStopWords)
    cachedStopWords = sorted(set(remove_longation(" ".join(cachedStopWords)).split(" ") + ['من' , 'و']))
    cachedStopWords.remove('كان')
    # print(dict(enumerate(cachedStopWords)))
    return cachedStopWords

def remove_stopwords(in_msg_words, stopWords=None, split=False, progress_per=None):
    if not stopWords: 
        stopWords = get_stopwords()

    out_msg_words = in_msg_words.copy()
    for idx in in_msg_words.index:
        #go through each word in each msg_words row, remove stopwords, and set them on the index.
        if split:
            out_msg_words[idx] = [word for word in in_msg_words[idx].split() if word not in stopWords]
        else:
            out_msg_words[idx] = " ".join([word for word in in_msg_words[idx].split() if word not in stopWords])
        
        #print logs to monitor output
        if not progress_per:
            continue
        if idx % progress_per == 0:
            print('\rc = ' + str(idx) + ' / ' + str(len(in_msg_words)))
    return out_msg_words

# def remove_stopwords(in_msg_words, stopWords=None, split=False):
#     if not stopWords: 
#         stopWords = get_stopwords()
    
#     if split:
#         out_msg_words = in_msg_words.apply(lambda x: re.sub(r'\b('+"|".join(stopWords)+r')(\W|\Z)', '', str(x)).split())
#     else:
#         out_msg_words = in_msg_words.apply(lambda x: re.sub(r'\b('+"|".join(stopWords)+r')(\W|\Z)', '', str(x)))
        
#     return out_msg_words

def stem(message):
    return message.apply(lambda x: " ".join([st.stem(w) for w in remove_longation(str(x)).split()]))

def lemma(message):
    return message.apply(lambda x: " ".join([stm.light_stem(w) for w in x.split()]))

def clean_df(message):
    return message.astype(str).apply(remove_longation)

def get_vec(n_model,dim, token):
    vec = np.zeros(dim)
    is_vec = False
    if token not in n_model.wv:
        _count = 0
        is_vec = True
        for w in token.split("_"):
            if w in n_model.wv:
                _count += 1
                vec += n_model.wv[w]
        if _count > 0:
            vec = vec / _count
    else:
        vec = n_model.wv[token]
    return vec

def calc_vec(pos_tokens, neg_tokens, n_model, dim):
    vec = np.zeros(dim)
    for p in pos_tokens:
        vec += get_vec(n_model,dim,p)
    for n in neg_tokens:
        vec -= get_vec(n_model,dim,n)
    
    return vec   

## -- Retrieve all ngrams for a text in between a specific range
def get_all_ngrams(text, nrange=3):
    text = re.sub(r'[\,\.\;\(\)\[\]\_\+\#\@\!\?\؟\^]', ' ', text)
    tokens = [token for token in text.split(" ") if token.strip() != ""]
    if len(tokens) < 2:
        return tokens
    ngs = []
    for n in range(2,nrange+1):
        ngs += [ng for ng in ngrams(tokens, n)]
    return tokens + ["_".join(ng) for ng in ngs if len(ng)>0 ]

## -- Retrieve all ngrams for a text in a specific n
def get_ngrams(text, n=2):
    text = re.sub(r'[\,\.\;\(\)\[\]\_\+\#\@\!\?\؟\^]', ' ', text)
    tokens = [token for token in text.split(" ") if token.strip() != ""]
    ngs = [ng for ng in ngrams(tokens, n)]
    return ["_".join(ng) for ng in ngs if len(ng)>0 ]

## -- filter the existed tokens in a specific model
def get_existed_tokens(tokens, n_model):
    return [tok for tok in tokens if tok in n_model.wv ]

## -- visualize confusion matrix using seaborn
def plot_cm(y_true, y_pred, labels=None, figsize=None):
    if not labels:
        labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)

## -- save train/val data
def save_train_val_data(x_train, y_train, x_val, y_val, int_category, category_int):
#    np.save('./data/x_train_np.npy', x_train)
#    np.save('./data/y_train_np.npy', y_train)
#    np.save('./data/x_val_np.npy', x_val)
#    np.save('./data/y_val_np.npy', y_val)
#    np.savez_compressed('data/All_Data_np.npz',
#                        x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    filename = os.path.join('data', 'train_val_data.dat')
    with open(filename, 'wb') as handle:
        pickle.dump(x_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(x_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(y_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(int_category, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(category_int, handle, protocol=pickle.HIGHEST_PROTOCOL)

## -- load train/val data
def read_train_val_data():
    """Instead of running the entire pipeline at all times."""
    filename = os.path.join('data', 'train_val_data.dat') ##@
    with open(filename, 'rb') as handle:
        x_train = pickle.load(handle)
        y_train = pickle.load(handle)
        x_val = pickle.load(handle)
        y_val = pickle.load(handle)
        int_category = pickle.load(handle)
        category_int = pickle.load(handle)
    return (x_train, y_train), (x_val, y_val), int_category, category_int
    
    
## -- save and load keras model
def save_keras_model(model, parameters, model_name, path='models'):
    data_path = os.path.join(path, model_name)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    model.save_weights(os.path.join(data_path, model_name+'_weights.hdf5'))
    with open(os.path.join(data_path, model_name+'_architecture.json'), 'w') as f:
        f.write(model.to_json())
    with open(os.path.join(data_path, model_name+'_parameters.pkl'), 'wb') as fid:
        pickle.dump(parameters, fid)    
        
def load_keras_model(model_name, path='models'):
    data_path = os.path.join(path, model_name)
    with open(os.path.join(data_path, model_name+'_parameters.pkl'), 'rb') as fid:
        parameters = pickle.load(fid)
    with open(os.path.join(data_path, model_name+'_architecture.json'), 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(os.path.join(data_path, model_name+'_weights.hdf5'))
    return model, parameters
    
## -- saver class for Word2vec callback

class EpochSaver(CallbackAny2Vec):

    def __init__(self, n_epochs, per_epoch=10, path='.'):
        self.n_epochs = n_epochs
        self.per_epoch = per_epoch
        self.savedir = path
        self.epoch = 0
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir, exist_ok=True)
        print('Starting training over {} epochs, saving checkpoints every {} epochs.'.format(self.n_epochs, self.per_epoch))

    def on_epoch_begin(self, model):
        print("Epoch {}/{} started ... ".format(self.epoch + 1, self.n_epochs))

    def on_epoch_end(self, model):
        if self.per_epoch is None:
            pass
        elif (self.epoch + 1) % self.per_epoch == 0:
            savepath = os.path.join(self.savedir, "w2v_model_epoch{}.gz".format(self.epoch + 1))
            model.save(savepath)
            print(
                "Epoch {} saved at '{}',".format(self.epoch + 1, savepath), 
                end='\t'
                )
            if os.path.isfile(os.path.join(self.savedir, "w2v_model_epoch{}.gz".format(self.epoch + 1 - self.per_epoch))):
                print("previous model deleted ", end='')
                for f in glob.glob(os.path.join(self.savedir, "w2v_model_epoch{}.gz*".format(self.epoch + 1 - self.per_epoch))):
                    os.remove(f) #!rm -rf $f
            print()
        self.epoch += 1

## -- Word2Vec vectorizer class for classification
        
class WordVecVectorizer(object):
    def __init__(self, word2vec, max_len=50):
        self.word2vec = word2vec
        self.dim = word2vec.vector_size
        self.max_len = max_len

    def fit(self, X, y):
        return self    

    def transform(self, X):
        """
        Transforms a document to one vector of a size = word2vec.vector_size (dim), 
        by taking the mean of th vectors of every word.
        The output size will be of shape (X.shape[0], self.dim).
        """
        return np.array([
            np.mean([self.word2vec[w] for w in texts.split() if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for texts in X
        ])
        
    def instance_transform(self, X):
        """
        Transforms a document to a 3D array based on the word2vec vector transformation of every instance (word),
        while the words that exceeds the maximum allowable length (max_len) will be removed.
        If the sentance is less than max_len, it will be padded with zeros.
        The output size will be of shape (X.shape[0], self.max_len, self.dim).
        """
        return np.array([
            [self.word2vec[w] if w in self.word2vec else np.zeros(self.dim) for i, w in enumerate(texts.split()) if i<self.max_len] + [np.zeros(self.dim)]*(self.max_len-min(self.max_len, len(texts.split())))
            for texts in X
        ])