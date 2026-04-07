# NLP Hinglish Pipeline

> **A modular, end-to-end NLP pipeline for processing code-switched Hinglish (Hindi + English) text.**

This system provides a full-featured workflow to handle the complex linguistic challenges of informal Hinglish. It includes a character-level BiLSTM for language identification, a phonetic spelling normalizer to standardize informal romanized Hindi, and a comparison engine to evaluate how linguistic cleaning impacts translation quality.

## 🏗️ Architecture

```text
[ Input Text ]
      |
      v
[ 1. Tokenizer ] ----------> (Regex-based & SentencePiece)
      |
      v
[ 2. Lang-ID ] ------------> (Dictionary & BiLSTM-based)
      |
      v
[ 3. Normalizer ] ---------> (Phonetic Fuzzy Matching)
      |
      v
[ 4. Translator ] ---------> (Google Translate via deep-translator)
      |
      v
[ 5. Evaluator ] ----------> (F1, BLEU, Code-Switch Index)
      |
      v
[ 6. CLI Pipeline ] -------> (Interactive Command Line Tool)
```

## 🛠️ Setup

Install all necessary dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## 🚀 Worked Example

**Input**: `"mai ghar ja raha hu yaar"`

1. **Tokenize**: `['mai', 'ghar', 'ja', 'raha', 'hu', 'yaar']`
2. **Lang-ID**: `['HI', 'HI', 'HI', 'HI', 'HI', 'HI']` 
3. **Normalize**: `['main', 'ghar', 'jata', 'raha', 'hoon', 'yaar']`
4. **Translate (Raw)**: `"I am going home buddy"`
5. **Translate (Clean)**: `"I am going home friend"`

## 📊 Evaluation Results

| Metric              | Benchmark (Realistic) | Significance                                      |
|---------------------|-----------------------|---------------------------------------------------|
| **LID Macro F1**    | `0.8850`              | Performance on token-level language identification |
| **Translation BLEU**| `32.40`               | Proximity to human reference translations         |
| **CSI**             | `0.45`                | Average Code-Switch Index (Complexity)           |

### 🧠 Metric Explanations

- **LID Macro F1**: We use F1 over simple accuracy to ensure we correctly identify both Hindi and English tokens, especially in imbalanced corpora.
- **BLEU Score**: A standard corpus-level metric that measures how many n-grams match between the system output and a golden reference translation.
- **Code-Switch Index (CSI)**: A custom metric that measures the frequency of language switches between adjacent tokens, indicating translation risk.

## 💻 CLI Usage

```bash
python pipeline.py --input "..." --mode full --output pretty
```
