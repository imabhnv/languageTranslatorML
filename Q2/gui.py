import tkinter as tk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import os

# Load tokenizers
with open('model/eng_tokenizer.pkl', 'rb') as f:
    eng_tokenizer = pickle.load(f)
with open('model/fr_tokenizer.pkl', 'rb') as f:
    fr_tokenizer = pickle.load(f)
with open('model/hi_tokenizer.pkl', 'rb') as f:
    hi_tokenizer = pickle.load(f)

# Reverse lookup
reverse_fr_index = {i: w for w, i in fr_tokenizer.word_index.items()}
reverse_hi_index = {i: w for w, i in hi_tokenizer.word_index.items()}
start_fr = fr_tokenizer.word_index['startseq']
end_fr = fr_tokenizer.word_index['endseq']
start_hi = hi_tokenizer.word_index['startseq']
end_hi = hi_tokenizer.word_index['endseq']

# Load models from model/ folder
encoder_fr = load_model(os.path.join('model', 'encoder_fr.h5'))
decoder_fr = load_model(os.path.join('model', 'decoder_fr.h5'))
encoder_hi = load_model(os.path.join('model', 'encoder_hi.h5'))
decoder_hi = load_model(os.path.join('model', 'decoder_hi.h5'))

# Dynamically get maxlen
maxlen_eng = encoder_fr.input_shape[1]
maxlen_fr = decoder_fr.input_shape[0][1]
maxlen_hi = decoder_hi.input_shape[0][1]

# Decode function
def decode_sequence(input_seq, encoder_model, decoder_model, start_token, end_token, reverse_index, maxlen_out):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_token

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_index.get(sampled_token_index, '')

        if sampled_word == 'endseq' or len(decoded_sentence.split()) > maxlen_out:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()

# GUI
root = tk.Tk()
root.title("Dual Language Translator (English → French & Hindi)")

tk.Label(root, text="Enter English Sentence:").pack()
entry = tk.Entry(root, width=80)
entry.pack()

fr_output = tk.Label(root, text="French Translation:")
fr_output.pack()

hi_output = tk.Label(root, text="Hindi Translation:")
hi_output.pack()

def translate_text():
    text = entry.get().lower().strip()
    if len(text) < 10:
        fr_output.config(text="French Translation: Upload again")
        hi_output.config(text="Hindi Translation: Upload again")
        return

    seq = eng_tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=maxlen_eng, padding='post')

    fr_translation = decode_sequence(seq, encoder_fr, decoder_fr, start_fr, end_fr, reverse_fr_index, maxlen_fr)
    hi_translation = decode_sequence(seq, encoder_hi, decoder_hi, start_hi, end_hi, reverse_hi_index, maxlen_hi)

    fr_output.config(text=f"French Translation: {fr_translation}")
    hi_output.config(text=f"Hindi Translation: {hi_translation}")

tk.Button(root, text="Translate", command=translate_text).pack()
root.mainloop()
