# 🌐 Dual Language Translator (English ➝ French & Hindi)

A deep learning-based sequence-to-sequence model that **translates English sentences into both French and Hindi** simultaneously. Built using **TensorFlow** and **Keras**, this project utilizes **LSTM-based encoder-decoder architectures** with separate models for each language. It includes a **GUI built with Tkinter** for real-time translation.

---

## 🚀 Features

- 🔤 Translates English into **two languages**: French and Hindi.
- ⚡ Trained using **parallel corpora** (500+ aligned sentence pairs).
- 💡 Uses `startseq` / `endseq` markers for better decoder alignment.
- ✅ Validates minimum input length for meaningful translation.
- 📊 Efficient sequence padding and tokenization.
- 🖥️ GUI using `Tkinter` for easy user interaction.
- 🧠 Trained LSTM models for accurate sentence-level translation.

---

## 🛠️ Tech Stack

| Component      | Technology |
|----------------|------------|
| Programming    | Python 3.x |
| Deep Learning  | TensorFlow / Keras |
| GUI            | Tkinter |
| Data Handling  | Numpy, Pickle |
| Tokenization   | Keras Tokenizer |
| Input Format   | CSV Files (Parallel Sentences) |

---

## 📁 Directory Structure

DualLanguageTranslator/
│
├── data/
│ ├── english.csv
│ ├── french.csv
│ └── hindi.csv
│
├── model/
│ ├── model_en_fr.h5
│ ├── model_en_hi.h5
│ ├── encoder_fr.h5
│ ├── decoder_fr.h5
│ ├── encoder_hi.h5
│ ├── decoder_hi.h5
│
├── eng_tokenizer.pkl
├── fr_tokenizer.pkl
├── hi_tokenizer.pkl
│
├── training.py
├── dual_language_gui.py
└── README.md

markdown
Always show details

Copy

---

## ⚙️ How it Works

1. **Preprocessing**  
   - All sentences are lowercased.
   - French and Hindi lines are prepended with `"startseq"` and appended with `"endseq"`.
   - Sentences are tokenized using `Tokenizer(filters='')`.

2. **Model Training**  
   - Two separate models are trained:
     - `English ➝ French`
     - `English ➝ Hindi`
   - Encoder captures context; Decoder generates target sentence step-by-step.
   - Each decoder uses initial states from the encoder.

3. **Model Saving**  
   - Both the `encoder` and `decoder` parts of each model are saved separately.
   - Tokenizers are also pickled for later inference use.

4. **Inference / GUI**  
   - Input English is tokenized and padded.
   - Encoder generates states.
   - Decoder generates words one-by-one until `endseq` is predicted.

---

## 🧪 Requirements

Install the necessary dependencies:
```bash
pip install tensorflow numpy
Optional:

bash
Always show details

Copy
pip install pillow
💻 How to Run
➤ Train the Models
bash
Always show details

Copy
python training.py
➤ Run the GUI Translator
bash
Always show details

Copy
python dual_language_gui.py
✅ Input Criteria
English sentence must be at least 10 characters long.

If not, GUI will prompt: "Upload again".

⚠️ Notes
Avoid using punctuation like “ ” (smart quotes) in CSVs. Stick to plain " or no quotes.

Keep sentences aligned line-by-line across all three files.

📌 Sample
Input:
the united states is usually chilly during july , and it is usually freezing in november .

French Output:
les états-unis sont généralement froids en juillet et il se congèle généralement en novembre

Hindi Output:
संयुक्त राज्य अमेरिका आमतौर पर जुलाई के दौरान ठंडा होता है और यह आमतौर पर नवंबर में जमता है

🧠 Future Improvements
Add support for speech-to-text input.

Support additional languages (Spanish, German).

Use Transformer models for higher accuracy.

🏁 License
This project is for academic and learning purposes. Use freely with credits. ✨

Made with ❤️ by [Abhinav Varshney]