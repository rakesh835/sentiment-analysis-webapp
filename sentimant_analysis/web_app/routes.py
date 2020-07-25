from web_app import app

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras 
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from nltk.stem.porter import *

import os
import glob

import numpy as np
import csv
import pickle


model = load_model('sentiment_analysis1595505698.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

upload_dir="web_app/static/uploaded"
app.config['UPLOAD_FOLDER'] = upload_dir


@app.route('/')
def upload_f():
    return render_template('pred.html', ss='')

def finds(sentiment):
    
    padding="post"
    max_length=200
    trunc_type="post"
    vocab_size=30000
    oov_tok="<OOV_tok>"
    
    #model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    #tokenizer=Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    test_sequence=tokenizer.texts_to_sequences(sentiment)
    testing=pad_sequences(test_sequence, maxlen=max_length, padding=padding, truncating=trunc_type)

    output=model.predict(testing)
    print(output)
    if output[0][0]>=0.6:
        ss="Positive"
    elif output[0][0]<=0.4:
        ss="Negaive"
    else:
        ss="Neutral"

    return ss, output[0][0]

@app.route('/prediction', methods = ['GET', 'POST'])
def writeSentiment():

    if request.method == 'POST':
        sentiment_user = request.form['review']
        sentiment=sentiment_user.split(" ", 0)
        print("review:- ", sentiment)



        ss, value=finds(sentiment)

        if value >=0.5:
            result=1
        else:
            result=0

        filename="reviews.csv"

        with open (filename, "a") as file:
            writer=csv.writer(file, delimiter=",")
            writer.writerow((sentiment_user, result))
        
        return render_template('pred.html', ss = "Your review is "+ss)







