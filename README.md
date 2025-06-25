#  Information Retrieval-Based Chatbot for Malay Main Dish Recipes

This is a **Bahasa Melayu** recipe question-answering chatbot that helps users retrieve specific information about traditional Malay dishes. It combines **intent classification** using BERT embeddings with **fuzzy string matching** to retrieve recipes from a structured dataset.

---

## Features
- **Intent Classification** (e.g., bahan, masa, petua, penerangan)
- **Information Retrieval** using **fuzzy matching** on recipe titles
- **Square bracket recipe name support** for precise queries
- **Streamlit-powered chatbot UI**
- Tailored for **Bahasa Melayu** culinary queries

---

## Project Structure
```
.
├── app.py                          # Main Streamlit chatbot application
├── Intent.json                    # Intent labels and example patterns (used for training the intent classifier)
├── Intent-classifier-training.ipynb  # Classifier training notebook (trained using Kaggle Platform)
├── intent_classifier.pkl      # Trained intent classification model
├── label_encoder.pkl          # Encoded intent labels
├── Preprocess_Data.json       # Recipe dataset used for retrieval
├── requirements.txt               # Project dependencies
```
---

## Getting Started

## Run the Chatbot Online

You can run this chatbot online for free using **Streamlit Cloud**  — no installation needed, just open it in your browser and start chatting! Please follow the steps below to run the chatbot.

### Step-by-Step: Deploy to Streamlit Cloud

### 1. Prepare Your Files

Ensure these files are inside your project folder:

```
app.py
Intent.json
Intent-classifier-training.ipynb
intent_classifier.pkl
label_encoder.pkl
Preprocess_Data.json
requirements.txt
```

### 2. Upload to GitHub

1. Go to [https://github.com/](https://github.com/)
2. Create a new repository (e.g., `recipe-chatbot`)
3. Upload all the files to the repository


### 3. Deploy on Streamlit Cloud

1. Visit [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Click **“Sign in with GitHub”**
3. Click **“New App”**
4. Choose your GitHub repository
5. Set:
   - **Main file:** `app.py`
   - **Branch:** `main`
6. Click **“Deploy”**

🤩 Your chatbot is now online and ready to use!

---

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

---

## Example Questions

- `Apa bahan untuk [rendang ayam]?`
- `Berapa lama masa penyediaan [kari ayam istimewa]?`
- `Bagaimana cara memasak [ayam masak merah]?`
- `Ada petua untuk masak [sambal belacan padu]?`
- `Terima kasih`

---

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

---

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

---

## Training

You can train or retrain the intent classifier using the **Kaggle Notebook**:  
📄 **`Intent-classifier-training.ipynb`**

### Step-by-Step: Retrain the Intent Classifier on Kaggle

#### 1. Upload Dataset and Notebook to Kaggle

1. Go to [https://www.kaggle.com/](https://www.kaggle.com/)
2. Click `+ Create > Dataset` → Upload **`Intent.json`** → Click **Create**
3. Click `+ Create > New Notebook`
4. Click **“File” > “Import Notebook”** → Upload **`Intent-classifier-training.ipynb`**
5. On the right sidebar, click **“Add Data”** → Select the dataset you uploaded in Step 2

#### 2. Run the Notebook

- Load `Intent.json` using:

```python
import json

with open('/kaggle/input/path-to-your-uploaded-dataset(Intent.json)', 'r', encoding='utf-8') as f:
    data = json.load(f)
```

- The notebook includes:
  - Pattern–intent extraction
  - BERT embedding generation (`mesolitica/bert-base-standard-bahasa-cased`)
  - Label encoding using `LabelEncoder`
  - Classifier training (e.g., `LogisticRegression`)
  - Saving trained models using `joblib`

#### 3. Save the Trained Models

Add this to the end of the notebook:

```python
import joblib

joblib.dump(classifier_model, 'intent_classifier.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
```

After the notebook finishes:

1. Go to the **right sidebar** under **“Output”**
2. Click the **three dots** next to each file:
   - `intent_classifier.pkl`
   - `label_encoder.pkl`
3. Click **“Download”** to save both to your computer

---

## Notes

- Use square brackets (`[ ]`) to specify a **recipe name**.
- Queries must be in **Bahasa Melayu**.
- Fuzzy matching helps find close recipe names even if the spelling isn't exact.

---

## Acknowledgements

- BERT embeddings from [Mesolitica](https://huggingface.co/mesolitica)
- Model training with [Scikit-learn](https://scikit-learn.org/)
- Fuzzy search by [RapidFuzz](https://github.com/maxbachmann/RapidFuzz)
- Built with [Streamlit](https://streamlit.io)  
