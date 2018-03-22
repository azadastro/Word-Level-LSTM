

import nltk
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization

import numpy as np
import random
import sys
import os

path = "woolfv-waltersickert-00-t.txt"

try:
    raw_text = open(path).read().lower()
except UnicodeDecodeError:
    import codecs
    raw_text = codecs.open(path, encoding='utf-8').read().lower()

print('corpus length:', len(raw_text))

chars = set(raw_text)
new_text = ""
for i in range(0,len(raw_text)):
    if ord(raw_text[i]) == 32 or ord(raw_text[i]) == ord('\n'):
        new_text += raw_text[i]
    if ord(raw_text[i]) > 96 and ord(raw_text[i]) < 123:
        new_text += raw_text[i]
    if ord(raw_text[i]) > 47 and ord(raw_text[i]) < 58:
        new_text += raw_text[i]

list_words = nltk.word_tokenize(new_text)
words = set(list_words)

print("total number of unique words",len(words))
print("total number of unique chars", len(chars))

word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

maxlen = 10
step = 2
print("maxlen:", maxlen, "step:", step)
sentences = []
next_words = []
sentences2=[]

for i in range(0, len(list_words)-maxlen, step):
    sentences2 = ' '.join(list_words[i: i + maxlen])
    sentences.append(sentences2)
    next_words.append((list_words[i + maxlen]))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
y = np.zeros((len(sentences), len(words)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence.split()):
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1

print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(words))))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(len(words)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

if os.path.isfile('weights'):
    print('Loading weights...')
    model.load_weights('weights')

# train the model, output generated text after each iteration
for iteration in range(1, 10000):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=256, epochs=1)
    model.save_weights('weights', overwrite=True)

    start_index = random.randint(0, len(list_words) - maxlen - 1)

    generated = ''
    sentence = list_words[start_index: start_index + maxlen]
    generated += ' '.join(sentence)
    print('----- Generating with seed: "', sentence, '"')
    print()
    sys.stdout.write(generated)
    print()

    for i in range(100):
        x = np.zeros((1, maxlen, len(words)))
        for t, word in enumerate(sentence):
            x[0, t, word_indices[word]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = np.argmax(preds)
        next_word = indices_word[next_index]
        generated += next_word
        del sentence[0]
        sentence.append(next_word)
        sys.stdout.write(' ')
        sys.stdout.write(next_word)
        sys.stdout.flush()
    print()
