import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# Make sure model folder exists
os.makedirs("model", exist_ok=True)

# Load data
with open('data/english.csv', encoding='utf-8') as f:
    english = [line.strip().lower() for line in f.readlines()]
with open('data/french.csv', encoding='utf-8') as f:
    french = ['startseq ' + line.strip().lower() + ' endseq' for line in f.readlines()]
with open('data/hindi.csv', encoding='utf-8') as f:
    hindi = ['startseq ' + line.strip().lower() + ' endseq' for line in f.readlines()]

# Tokenization
def tokenize(texts):
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return tokenizer, sequences

eng_tokenizer, eng_seq = tokenize(english)
fr_tokenizer, fr_seq = tokenize(french)
hi_tokenizer, hi_seq = tokenize(hindi)

maxlen_eng = max(len(s) for s in eng_seq)
maxlen_fr = max(len(s) for s in fr_seq)
maxlen_hi = max(len(s) for s in hi_seq)

X_eng = pad_sequences(eng_seq, maxlen=maxlen_eng, padding='post')
Y_fr = pad_sequences(fr_seq, maxlen=maxlen_fr, padding='post')
Y_hi = pad_sequences(hi_seq, maxlen=maxlen_hi, padding='post')

# Save tokenizers
with open('model/eng_tokenizer.pkl', 'wb') as f:
    pickle.dump(eng_tokenizer, f)
with open('model/fr_tokenizer.pkl', 'wb') as f:
    pickle.dump(fr_tokenizer, f)
with open('model/hi_tokenizer.pkl', 'wb') as f:
    pickle.dump(hi_tokenizer, f)

# Targets
Y_fr_input = Y_fr[:, :-1]
Y_fr_target = np.expand_dims(Y_fr[:, 1:], -1)

Y_hi_input = Y_hi[:, :-1]
Y_hi_target = np.expand_dims(Y_hi[:, 1:], -1)

# Build Model
def build_model(vocab_inp, vocab_out, len_inp, len_out):
    encoder_inputs = Input(shape=(len_inp,), name="encoder_inputs")
    enc_emb = Embedding(vocab_inp, 256, mask_zero=True, name="encoder_embedding")(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(256, return_state=True, name="encoder_lstm")(enc_emb)

    decoder_inputs = Input(shape=(len_out,), name="decoder_inputs")
    dec_emb = Embedding(vocab_out, 256, mask_zero=True, name="decoder_embedding")(decoder_inputs)
    decoder_lstm, _, _ = LSTM(256, return_sequences=True, return_state=True, name="decoder_lstm")(
        dec_emb, initial_state=[state_h, state_c])
    decoder_outputs = Dense(vocab_out, activation='softmax', name="output_dense")(decoder_lstm)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

vocab_eng = len(eng_tokenizer.word_index) + 1
vocab_fr = len(fr_tokenizer.word_index) + 1
vocab_hi = len(hi_tokenizer.word_index) + 1

# ---------- Train French ----------
model_fr = build_model(vocab_eng, vocab_fr, maxlen_eng, maxlen_fr - 1)
model_fr.fit([X_eng, Y_fr_input], Y_fr_target, batch_size=64, epochs=200, validation_split=0.1)
model_fr.save('model/model_en_fr.h5')

# Save Encoder-Decoder French
encoder_inputs_fr = model_fr.get_layer("encoder_inputs").input
enc_emb_fr = model_fr.get_layer("encoder_embedding")(encoder_inputs_fr)
_, state_h_fr, state_c_fr = model_fr.get_layer("encoder_lstm")(enc_emb_fr)
encoder_model_fr = Model(encoder_inputs_fr, [state_h_fr, state_c_fr])
encoder_model_fr.save("model/encoder_fr.h5")

decoder_inputs_fr = model_fr.get_layer("decoder_inputs").input
decoder_state_input_h = Input(shape=(256,), name="input_h")
decoder_state_input_c = Input(shape=(256,), name="input_c")
dec_emb_fr = model_fr.get_layer("decoder_embedding")(decoder_inputs_fr)
decoder_lstm = model_fr.get_layer("decoder_lstm")
decoder_outputs_fr, state_h, state_c = decoder_lstm(
    dec_emb_fr, initial_state=[decoder_state_input_h, decoder_state_input_c])
decoder_dense = model_fr.get_layer("output_dense")
decoder_outputs_fr = decoder_dense(decoder_outputs_fr)

decoder_model_fr = Model(
    [decoder_inputs_fr, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs_fr, state_h, state_c]
)
decoder_model_fr.save("model/decoder_fr.h5")

# ---------- Train Hindi ----------
model_hi = build_model(vocab_eng, vocab_hi, maxlen_eng, maxlen_hi - 1)
model_hi.fit([X_eng, Y_hi_input], Y_hi_target, batch_size=64, epochs=200, validation_split=0.1)
model_hi.save('model/model_en_hi.h5')

# Save Encoder-Decoder Hindi
encoder_inputs_hi = model_hi.get_layer("encoder_inputs").input
enc_emb_hi = model_hi.get_layer("encoder_embedding")(encoder_inputs_hi)
_, state_h_hi, state_c_hi = model_hi.get_layer("encoder_lstm")(enc_emb_hi)
encoder_model_hi = Model(encoder_inputs_hi, [state_h_hi, state_c_hi])
encoder_model_hi.save("model/encoder_hi.h5")

decoder_inputs_hi = model_hi.get_layer("decoder_inputs").input
decoder_state_input_h_hi = Input(shape=(256,), name="input_h")
decoder_state_input_c_hi = Input(shape=(256,), name="input_c")
dec_emb_hi = model_hi.get_layer("decoder_embedding")(decoder_inputs_hi)
decoder_lstm_hi = model_hi.get_layer("decoder_lstm")
decoder_outputs_hi, state_h_hi2, state_c_hi2 = decoder_lstm_hi(
    dec_emb_hi, initial_state=[decoder_state_input_h_hi, decoder_state_input_c_hi])
decoder_dense_hi = model_hi.get_layer("output_dense")
decoder_outputs_hi = decoder_dense_hi(decoder_outputs_hi)

decoder_model_hi = Model(
    [decoder_inputs_hi, decoder_state_input_h_hi, decoder_state_input_c_hi],
    [decoder_outputs_hi, state_h_hi2, state_c_hi2]
)
decoder_model_hi.save("model/decoder_hi.h5")
