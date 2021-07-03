import spacy
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.corpus import words
from flask import Flask, jsonify, request

app = Flask(__name__)

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
        
        print("*"*100)
        print("preprocessing and nc extracting started")
        
        #preprocessing and getting noun chunks
        for article in articles:
            article = self.preprocess(article)
            noun_chunks.extend(self.get_noun_chunks(article))   
        print("preprocessing and nc extracting done..")
        print("*"*100)

        #postprocessing the noun chunks
        print("post-processing of noun_chunks started")
        print("*"*100)
        noun_chunks = self.postprocess(noun_chunks)
        print("post-processing of noun_chunks done")
        print("*"*100)
        
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

@app.route('/nc_keyword_extraction', methods=['POST'])
def nc_keyword_extraction():
	articles = request.get_json()['data']
	model = Model()
	noun_chunks = model(articles)
	return jsonify({"noun_chunks": {"nc": noun_chunks}})	

if __name__ == "__main__":
    app.run(debug=True)