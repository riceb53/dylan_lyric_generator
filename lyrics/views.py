from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np 
import pandas as pd
import time
import pdb

def index(request):
    start = time.time()
    # pdb.set_trace()
    model = tf.keras.models.load_model('lyrics/dylan_model.h5')
    # model = load_model('mysite/polls/chatbot_model.h5')


    # print(model)

    # re-get lyrics and tokens which weren't saved
    cleaned_lyrics = []
    cleanest_lyrics = []

    df = pd.read_csv('lyrics/clear.csv')
    # print(df)    
    # df = df.head(5)
    # print(len(df.index))
    for index, row in df.iterrows():    
        lowered_and_split_lyrics = row['lyrics'].lower().split("\n")
        cleaned_lyrics = list(filter(None, lowered_and_split_lyrics))
        # print(cleaned_lyrics)
        # break
        cleanest_lyrics += cleaned_lyrics

    corpus = cleanest_lyrics

    # print(corpus)



    # make tokenizer
    tokenizer = Tokenizer()

    corpus = cleanest_lyrics

    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # print(tokenizer.word_index)
    # print(total_words)




    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # pad sequences 
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    # xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

    # ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


    # print(xs[6])
    # print(tokenizer.word_index['one'])



    seed_text =  request.GET.get('search') 
    next_words = int(request.GET.get('count'))

    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)

    end = time.time()
    print ("Time elapsed:", end - start)

    context = {
        'lyrics': seed_text
    }


    return render(request, 'lyrics/index.html', context)


