import streamlit as st
import json
import joblib
import numpy as np
import torch
import re
from transformers import AutoTokenizer, AutoModel
from rapidfuzz import process

# === Load Models and Tokenizer ===
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("mesolitica/bert-base-standard-bahasa-cased")
    model = AutoModel.from_pretrained("mesolitica/bert-base-standard-bahasa-cased").eval().to("cpu")
    clf = joblib.load("trained_models/intent_classifier.pkl")
    label_encoder = joblib.load("trained_models/label_encoder.pkl")
    with open("trained_models/Preprocess_Data.json", "r", encoding="utf-8") as f:
        recipes = json.load(f)
    return tokenizer, model, clf, label_encoder, recipes

tokenizer, model, clf, label_encoder, recipes = load_models()

# === Embedding Function ===
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        masked = hidden * mask
        summed = masked.sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        mean = summed / counts
    return mean[0].numpy()

# === Extract recipe name ===
def extract_recipe_name(text):
    match = re.search(r"\[(.*?)\]", text)
    return match.group(1).strip() if match else None

def find_top_recipe_matches(user_input, recipes, limit=3, threshold=0):
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # remove special chars (but keep space)
        text = re.sub(r"\d+", "", text)      # remove numbers
        return text.strip()

    titles = [r["Tajuk"] for r in recipes]
    normalized_titles = [normalize_text(t) for t in titles]
    normalized_input = normalize_text(user_input)
    matches = process.extract(normalized_input, normalized_titles, limit=limit)
    return [(titles[i], score) for (match, score, i) in matches]

def generate_response(user_input):
    cleaned_input = re.sub(
    r"\[.*?\]",
    lambda m: "[" + 
        re.sub(r"[^\w\s]", "", re.sub(r"\d+", "", m.group(0))).lower() + "]",
    user_input
    )
    embed = get_embedding(cleaned_input).reshape(1, -1)
    pred_class = clf.predict(embed)[0]
    intent = label_encoder.inverse_transform([pred_class])[0]

    # === Respond immediately for non-recipe intents ===
    if intent == "bertegur-sapa":
        return "Selamat sejahtera! Ada apa saya boleh bantu?"
    elif intent == "selamat-tinggal":
        return "Jumpa lagi! Semoga hari anda baik!"
    elif intent == "terima-kasih":
        return "Sama-sama! Gembira dapat membantu 😊"

    target_name = None  # initialize

    # === Extract recipe name ===
    extracted_name = extract_recipe_name(user_input)
    if extracted_name:
        normalized_extracted = re.sub(r"[^\w\s]", "", extracted_name.lower())
        normalized_extracted = re.sub(r"\d+", "", normalized_extracted).strip()
    
        for r in recipes:
            normalized_title = re.sub(r"[^\w\s]", "", r["Tajuk"].lower())
            normalized_title = re.sub(r"\d+", "", normalized_title).strip()
    
            if normalized_extracted == normalized_title:
                target_name = r["Tajuk"]
                break

        else:
            matches = find_top_recipe_matches(extracted_name, recipes, limit=3)
            top_score = matches[0][1] if matches else 0

            if top_score == 100:
                target_name = matches[0][0]
            elif 75 <= top_score < 100:
                suggestions = ", ".join(f"'{name}'" for name, _ in matches[:3])
                return f"Adakah anda maksudkan salah satu daripada ini: {suggestions}?"
            elif 70 <= top_score < 75:
                target_name = matches[0][0]
            else:
                return "Maaf, saya tidak jumpa resepi yang sepadan."
    else:
        matches = find_top_recipe_matches(user_input, recipes, limit=3)
        top_score = matches[0][1] if matches else 0

        if top_score == 100:
            target_name = matches[0][0]
        elif 85 <= top_score < 100:
            # Return full recipe info
            recipe = next((r for r in recipes if r["Tajuk"].strip().lower() == matches[0][0].strip().lower()), None)
            if recipe:
                tajuk = recipe.get("Tajuk", "Resepi")
                penerangan = recipe.get("Penerangan", "Tiada info")
                masa_penyediaan = recipe.get("masa_penyediaan", "tidak diketahui")
                masa_memasak = recipe.get("masa_memasak", "tidak diketahui")
                jumlah_masa = recipe.get("jumlah_masa", "tidak diketahui")
                hidangan = recipe.get("hidangan", "tidak diketahui")
                sumber = recipe.get("Sumber", "-")

                response = []
                response.append("Saya berjaya cari satu resepi untuk anda.\n")

                # Title
                response.append(f"#### 🍲 **{tajuk.title()}**\n")
            
                # Description
                response.append("##### 📝 **Penerangan**")
                response.append(f"{penerangan}\n")
            
                # Masa
                response.append("##### ⏱️ **Masa**")
                response.append(f"- **Masa penyediaan:** {masa_penyediaan}")
                response.append(f"- **Masa memasak:** {masa_memasak}")
                response.append(f"- **Jumlah masa:** {jumlah_masa}\n")
            
                # Hidangan
                response.append("##### 🍽️ **Hidangan**")
                response.append(f"**{hidangan}**\n")
            
                # Sumber
                response.append("##### 🔗 **Sumber**")
                response.append(f"[Link Resepi]({sumber})\n")
            
                # === Updated Bahan-Bahan section ===
                bahan = recipe.get("bahan_bahan", {})
                if bahan:
                    response.append("##### 📌 **Bahan-Bahan**\n")
                    for kategori, senarai in bahan.items():
                        response.append(f"**{kategori}:**")
                        for item in senarai:
                            response.append(f"- {item}")
                        response.append("")
            
                # Cara Memasak
                langkah = recipe.get("cara_memasak", {})
                if langkah:
                    response.append("##### 👩‍🍳 **Langkah-Langkah**")
                    for section, steps in langkah.items():
                        response.append(f"**{section}:**")
                        for step in steps:
                            response.append(f"- {step}")
                    response.append("")
            
                # Petua
                petua = recipe.get("petua_panduan", [])
                response.append("##### 💡 **Petua**")
                if petua:
                    for p in petua:
                        response.append(f"- {p}")
                else:
                    response.append("(Tiada petua disediakan)")
            
                return "\n".join(response)
            else:
                return "Maaf, saya tidak dapat menemui maklumat resepi penuh."
        elif 75 <= top_score < 85:
            suggestions = ", ".join(f"'{name}'" for name, _ in matches[:3])
            return f"Mungkin anda maksudkan: {suggestions}?"
        elif 70 <= top_score < 75:
            target_name = matches[0][0]
        else:
            return "Maaf, saya tidak pasti maksud anda. Boleh tanya dengan lebih jelas?"

    # === Final matching to get recipe object ===
    if not target_name:
        return "Maaf, saya tidak faham resepi yang dimaksudkan."

    recipe = next((r for r in recipes if r["Tajuk"].strip().lower() == target_name.strip().lower()), None)
    if not recipe:
        return f"Maaf, saya tidak jumpa resepi bernama '{target_name}'."

    tajuk = recipe.get("Tajuk", "Resepi")

    # === Handle recipe-based intents ===
    if intent == "penerangan":
        return f"Penerangan resepi '{tajuk}' ialah :\n\n{recipe.get('Penerangan', 'Tiada info')}"
    elif intent == "masa_penyediaan":
        return f"Masa penyediaan untuk '{tajuk}' ialah {recipe.get('masa_penyediaan', 'tidak diketahui')}."
    elif intent == "masa_memasak":
        return f"Masa memasak untuk '{tajuk}' ialah {recipe.get('masa_memasak', 'tidak diketahui')}."
    elif intent == "jumlah_masa":
        return f"Jumlah masa untuk '{tajuk}' ialah {recipe.get('jumlah_masa', 'tidak diketahui')}."
    elif intent == "bahan_bahan":
        bahan = recipe.get("bahan_bahan", {})
        lines = [f"Bahan-bahan untuk '{tajuk}' ialah:"]
        for kategori, senarai in bahan.items():
            lines.append(f"\n📌 {kategori}:")
            for item in senarai:
                lines.append(f"- {item}")
        return "\n".join(lines)
    elif intent == "cara_memasak":
        langkah = recipe.get("cara_memasak", {})
        response = [f"Langkah-langkah memasak '{tajuk}' ialah:"]
        for section, steps in langkah.items():
            response.append(f"\n📌 {section}:")
            for step in steps:
                response.append(f"- {step}")
        return "\n".join(response)
    elif intent == "petua_panduan":
        petua = recipe.get("petua_panduan", [])
        if not petua:
            return f"Petua memasak '{tajuk}':\n(Tiada petua disediakan)"
        return f"Petua memasak '{tajuk}' ialah:\n" + "\n".join(f"- {p}" for p in petua)
    else:
        return "Maaf, saya tidak pasti maksud anda. Boleh tanya dengan lebih jelas?"


# === Streamlit Chat UI ===
st.set_page_config(page_title="Chatbot Resepi", page_icon="🍲")
st.title("Chatbot Resepi")
st.markdown("Saya sedia menjawab soalan tentang resepi tradisional kaum Melayu! 🇲🇾")

# === Sidebar for sample questions and format info ===
st.sidebar.header("📍 Contoh Soalan")
sample_questions = [
    "Apa bahan untuk [rendang ayam]?",
    "Perlu berapa masa untuk masak [kari ayam istimewa]?",
    "Bagaimana cara memasak [nasi lemak bungkus]?",
    "Ada petua untuk [ayam masak merah]?",
    "ada nasi goreng kampung?",
    "Selamat sejahtera",
    "Terima kasih"
]
for q in sample_questions:
    if st.sidebar.button(q):
        st.session_state.messages.append({"role": "user", "content": q})
        response = generate_response(q)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ Format dan Panduan")
st.sidebar.markdown("""
- Gunakan tanda kurung [ ] untuk nyatakan nama resepi dengan jelas.
- Contoh: `Apa bahan untuk [rendang ayam]?`
- Jika tiada tanda kurung, sistem akan cuba cari padanan paling hampir dan jawapan mungkin kurang tepat.
- Tanya soalan dalam Bahasa Melayu sahaja.

#### ✅ Jenis Soalan Yang Disokong:
Berikut ialah jenis-jenis soalan (intent) yang boleh ditanya:
- **penerangan** – Contoh: `Apa itu [kari ayam istimewa]?`
- **masa_penyediaan** – Contoh: `Berapa lama masa penyediaan [rendang ayam]?`
- **masa_memasak** – Contoh: `Tempoh memasak [sambal belacan padu] berapa minit?`
- **jumlah_masa** – Contoh: `Berapa jumlah masa keseluruhan untuk masak [ayam masak hitam]?`
- **bahan_bahan** – Contoh: `Apa bahan diperlukan untuk [kangkung goreng belacan]?`
- **cara_memasak** – Contoh: `Bagaimana cara memasak [laksam sedap]?`
- **petua_panduan** – Contoh: `Ada petua untuk masakan [ayam masak merah]?`
- **bertegur-sapa** – Contoh: `Hai`, `Hello`, `Assalamualaikum`
- **selamat-tinggal** – Contoh: `Bye`, `Jumpa lagi`
- **terima-kasih** – Contoh: `Terima kasih`, `Thanks!`

""")

# === Initialize message history ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === Display message history ===
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# === Chat input (bottom of screen) ===
if prompt := st.chat_input("Cari resepi atau tanya soalan tentang resepi..."):
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate bot response
    response = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
