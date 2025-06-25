#  Information Retrieval-Based Chatbot for Malay Main Dish Recipes

This is a **Bahasa Melayu** recipe question-answering chatbot that helps users retrieve specific information about traditional Malay dishes. It combines **intent classification** using BERT embeddings with **fuzzy string matching** to retrieve recipes from a structured dataset.


## Features
- **Intent Classification** (e.g., bahan, masa, petua, penerangan)
- **Information Retrieval** using **fuzzy matching** on recipe titles
- **Square bracket recipe name support** for precise queries
- **Streamlit-powered chatbot UI**
- Tailored for **Bahasa Melayu** culinary queries

## Project Structure
```
.
├── app.py                          # Main Streamlit chatbot application
├── Intent.json                    # Intent labels and example patterns
├── Intent-classifier-training.ipynb  # Classifier training notebook
├── trained_models/
│   ├── intent_classifier.pkl      # Trained intent classification model
│   ├── label_encoder.pkl          # Encoded intent labels
│   └── Preprocess_Data.json       # Recipe dataset used for retrieval
├── requirements.txt               # Project dependencies
```

## Getting Started

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Ensure Trained Model Files Exist
Place the following inside a `trained_models/` directory:

- `intent_classifier.pkl`
- `label_encoder.pkl`
- `Preprocess_Data.json`

### 3. Launch the Chatbot
```bash
streamlit run app.py
```

## How This Chatbot Works
![tnl (6)](https://github.com/user-attachments/assets/c475e5ca-b7f3-4e3e-998c-4b4bb95c3329)

### Step 1: Intent Detection
- Uses `mesolitica/bert-base-standard-bahasa-cased` to generate text embeddings.
- A trained classifier (e.g., Logistic Regression) determines the user’s intent (e.g., asking for ingredients, preparation time).

### Step 2: Recipe Retrieval (Information Retrieval)
- If the user includes a recipe name in `[square brackets]`, it triggers a strict fuzzy matching routine.
- Without brackets, the chatbot performs **approximate fuzzy matching** using `rapidfuzz` to find the closest recipe titles.

### Step 3: Response Generation
- Returns specific fields from the matched recipe depending on the detected intent (e.g., cooking time, ingredients, steps, tips).


## Example Questions

- `Apa bahan untuk [rendang ayam]?`
- `Berapa lama masa penyediaan [kari ayam istimewa]?`
- `Bagaimana cara memasak [ayam masak merah]?`
- `Ada petua untuk masak [sambal belacan padu]?`
- `Terima kasih`


## Supported Intents
| Intent Tag         | Contoh Soalan                                               |
|--------------------|-------------------------------------------------------------|
| `penerangan`       | `Apa itu [kari ayam istimewa]?`                            |
| `masa_penyediaan`  | `Berapa lama masa penyediaan [rendang ayam]?`              |
| `masa_memasak`     | `Tempoh memasak [kari ayam istimewa] berapa minit?`        |
| `jumlah_masa`      | `Berapa jumlah masa keseluruhan untuk [ayam masak hitam]?` |
| `bahan_bahan`      | `Apa bahan diperlukan untuk [kangkung goreng belacan]?`    |
| `cara_memasak`     | `Bagaimana cara memasak [laksam sedap]?`                   |
| `petua_panduan`    | `Ada petua untuk masakan [ayam masak merah]?`              |
| `bertegur-sapa`    | `Hai`, `Assalamualaikum`, `Selamat Sejahtera`              |
| `selamat-tinggal`  | `Bye`, `Jumpa lagi`                                        |
| `terima-kasih`     | `Terima kasih`, `Thanks!`                                  |


## Dependencies

From `requirements.txt`:

```
streamlit
transformers
torch
rapidfuzz
scikit-learn
joblib
```

## Training

You can train or retrain the intent classifier using the Jupyter notebook:  
📄 **`Intent-classifier-training.ipynb`**

### Steps:

1. **Load `Intent.json`**  
   The notebook parses all patterns and intent tags into input–label pairs for training.

2. **Generate Sentence Embeddings**  
   For each text pattern, sentence embeddings are generated using the **`mesolitica/bert-base-standard-bahasa-cased`** model from Hugging Face.

3. **Label Encoding**  
   Intent tags (e.g., `bahan_bahan`, `cara_memasak`) are encoded using `LabelEncoder` from `sklearn.preprocessing`.

4. **Train the Classifier**  
   A `LogisticRegression` model (or any scikit-learn classifier) is trained on the embeddings and encoded labels.

5. **Save the Trained Models**  
   Use `joblib` to save:
   - The classifier → `intent_classifier.pkl`
   - The label encoder → `label_encoder.pkl`

   Example:
   ```python
   import joblib

   joblib.dump(classifier_model, 'trained_models/intent_classifier.pkl')
   joblib.dump(label_encoder, 'trained_models/label_encoder.pkl')
   ```

6. **Use in Inference**  
   These models will be automatically loaded in `app.py` and used to classify user questions at runtime.


## Notes

- Use square brackets (`[ ]`) to specify a **recipe name**.
- Queries must be in **Bahasa Melayu**.
- Fuzzy matching helps find close recipe names even if the spelling isn't exact.


## Acknowledgements

- BERT embeddings from [Mesolitica](https://huggingface.co/mesolitica)
- Fuzzy search by [RapidFuzz](https://github.com/maxbachmann/RapidFuzz)
- Built with [Streamlit](https://streamlit.io)  
