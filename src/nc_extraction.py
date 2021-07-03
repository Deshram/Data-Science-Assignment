#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import spacy
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.corpus import words


# In[2]:


#TO-DO: Chnage the directory below to atleast 20 sample articles
files = os.listdir('663_webhose-2015-09-new_20170904095535')


# In[3]:


articles = []
for i in files[:20]:
    with open('663_webhose-2015-09-new_20170904095535/'+i, 'rb') as f:
        file = json.load(f)
        articles.append(file['text'])
        


# In[23]:


class Model:
    
    def __init__(self):
        pass
        
    #It prerocesses the input news articles
    def preprocess(self,text: str) -> str:
        
        text = text.lower()    #converting the text into lowercase
        text = re.sub('[^ a-z]', ' ', text)    #keeping only letters in teh text
        text = re.sub('\s\s+', ' ', text)    #removing extra tabs

        #removing stopwords
        text = text.split(' ')
        stop_words = set(stopwords.words('english'))
        text = [word for word in text if word not in stop_words]
        text = " ".join(text)
        
        return text

    #Noun Chunk extraction engine: using spacy in built feature
    def get_noun_chunks(self, text: str) -> list:
        
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        noun_chunks = [i.text for i in list(doc.noun_chunks) if (len(i.text.split(' ')) == 2) or (len(i.text.split(' ')) == 3)]
        return noun_chunks

    #Calulating tf-idf score of the whole corpus  (will be used to select top noun chunks)
    def tf_idf(self, articles: list) -> (list,list) :
       
        vectorizer = TfidfVectorizer(ngram_range=(2,3))
        tfidf_scores = vectorizer.fit_transform(articles)
        tfidf_scores = tfidf_scores.toarray()
        tfidf_scores = tfidf_scores.sum(axis=0)
        
        #getting total words in corpus 
        vocab = vectorizer.get_feature_names()
        return tfidf_scores, vocab

    #Checks if the given chunk has english words and makes sense
    def if_word(self, text: str) -> bool:
        text = text.split(' ')
        n_words = len(text)
        text = [word for word in text if word in set(words.words())]
        if len(text)<n_words:
            return True
        else:
            return False

    #Eliminating bad noun chunks
    def postprocess(self, noun_chunks: list) -> list:
        noun_chunks = list(set(noun_chunks))      #removing duplicate chunks
#         noun_chunks = [chunks for chunks in noun_chunks if self.if_word(chunks)]      #keeping the words that make sense
        return noun_chunks
    
    #The main pipeline 
    def __call__(self,articles: list) -> list:
        noun_chunks = []
        
        print("preprocessing and nc extracting started")
        
        #preprocessing and getting noun chunks
        for article in articles:
            article = self.preprocess(article)
            noun_chunks.extend(self.get_noun_chunks(article))   
        print("preprocessing and nc extracting done..")
        
        #postprocessing the noun chunks
        print("post-processing of noun_chunks started")
        noun_chunks = self.postprocess(noun_chunks)
        print("post-processing of noun_chunks done")
        
        #calculating tfidf 
        tfidf_scores, vocab = self.tf_idf(articles)
        
        #getting tfidf score fir
        noun_chunks_score = []
        for i in noun_chunks:
            try:
                idx = vocab.index(i)
                noun_chunks_score.append([i,tfidf_scores[idx]])
            except:
                pass

        noun_chunks_score = sorted(noun_chunks_score, key=lambda x:x[1])[::-1][:10]
        return [i[0] for i in noun_chunks_score]


# In[24]:


model = Model()  #instantiating the model
noun_chunks = model(articles)
noun_chunks


# In[20]:




