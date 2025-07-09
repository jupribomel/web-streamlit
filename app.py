import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import io
from sklearn.feature_extraction.text import TfidfVectorizer

# Download nltk data yang diperlukan
nltk.download('stopwords')
nltk.download('punkt')

# Fungsi memuat lexicon dari file CSV
def load_lexicon(file_path, sentiment_type):
    lexicon_df = pd.read_csv(file_path)
    # Mengasumsikan kolom 'word' ada dan kita akan menetapkan bobot secara manual
    if sentiment_type == 'positive':
        return {word: 5 for word in lexicon_df['word']}
    elif sentiment_type == 'negative':
        return {word: -5 for word in lexicon_df['word']}
    else:
        return {} # Mengembalikan dictionary kosong jika tipe tidak dikenal

# Fungsi menggabungkan lexicon positif dan negatif
def merge_lexicons(positive_lexicon, negative_lexicon):
    combined_lexicon = defaultdict(int)
    # Menetapkan bobot +5 untuk kata positif
    for word in positive_lexicon:
        combined_lexicon[word] = 5
    # Menetapkan bobot -5 untuk kata negatif
    for word in negative_lexicon:
        combined_lexicon[word] = -5
    return dict(combined_lexicon)

# Fungsi preprocessing teks
def preprocess_text(text, stop_words):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'@\w+|#\w+|http\S+|https\S+|[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# Fungsi pelabelan berdasarkan lexicon gabungan
def label_review(review, sentiment_lexicon):
    if not isinstance(review, str):
        return 1 # Default ke positif jika bukan string
    words = review.lower().split()
    score = 0
    for word in words:
        # Mengambil bobot dari lexicon, default 0 jika kata tidak ditemukan
        score += sentiment_lexicon.get(word, 0)
    return 1 if score >= 0 else 0 # 1 untuk positif/netral, 0 untuk negatif

# Fungsi menampilkan WordCloud
def display_wordcloud(text, title):
    if not text.strip():
        st.warning("Tidak ada data teks untuk WordCloud.")
        return
    wordcloud = WordCloud(width=800, height=400, background_color="#1e1e2f", colormap='plasma').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    st.pyplot(plt)

# Fungsi untuk menampilkan grafik frekuensi kata
def display_word_frequency(text, title, top_n=10):
    if not text.strip():
        st.warning("Tidak ada data teks untuk visualisasi frekuensi kata.")
        return
    words = text.split()
    word_counts = Counter(words)
    most_common = word_counts.most_common(top_n)
    if len(most_common) == 0:
        st.warning("Tidak ada data frekuensi kata yang cukup untuk ditampilkan.")
        return
    words, counts = zip(*most_common)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts), y=list(words), palette='plasma')
    plt.xlabel("Frekuensi")
    plt.ylabel("Kata")
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)

# Ganti fungsi perhitungan TF-IDF dengan sklearn TfidfVectorizer untuk lebih cepat
def compute_tfidf_weights_sklearn(docs):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return tfidf_df, vectorizer

# Definisi model RNN
class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
        x = x.unsqueeze(1)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Styling global
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #ffffff;
        color: #333333;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #FF6F61;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    .sub-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #FFA07A;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sidebar .sidebar-content h2 {
        color: #FF6F61 !important;
        font-weight: 700;
    }
    div.stDownloadButton > button:first-child {
        background-color: #FF6F61;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 8px 20px;
        transition: 0.3s ease background-color;
    }
    div.stDownloadButton > button:first-child:hover {
        background-color: #e6564b;
        color: white;
    }
    div[role="progressbar"] > div {
        background-color: #FF6F61 !important;
    }
    textarea {
        background-color: #222222 !important;
        color: #ffffff !important;
        border-radius: 5px;
    }
    section[data-testid="stSidebar"] > div.css-1d391kg {
        background-color: #1e1e2f;
        border-radius: 12px;
        padding: 1rem;
    }
    .box-container {
        background-color: #1e1e2f;
        padding: 15px 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 0 15px rgba(255,111,97,0.4);
    }
    .box-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 15px;
        color: #FF6F61;
        border-bottom: 3px solid #FF6F61;
        padding-bottom: 8px;
    }
    .stat-box {
        background-color: #f9f9f9;
        border-radius: 12px;
        padding: 15px 20px;
        margin: 8px;
        text-align: center;
        box-shadow: 0 0 10px rgba(255,111,97,0.3);
        color: #222222;
        font-weight: 600;
        flex: 1;
    }
    .stat-title {
        font-size: 1.2rem;
        margin-bottom: 8px;
        color: #FF6F61;
    }
    .stat-value {
        font-size: 1.5rem;
    }
    .green {
        color: #2ECC71 !important;
    }
    .red {
        color: #E74C3C !important;
    }
    .train-metrics {
        display: flex;
        justify-content: space-around;
        margin-top: 15px;
        gap: 15px;
    }
    .metric-box {
        background-color: #fff5f5;
        border-radius: 10px;
        padding: 10px 20px;
        box-shadow: 0 0 8px rgba(255,111,97,0.25);
        text-align: center;
        flex: 1;
    }
    .metric-title {
        font-weight: 700;
        color: #FF6F61;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
    }
    .training-complete {
        background-color: #fff0f0;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        text-align: center;
        box-shadow: 0 0 15px rgba(255,111,97,0.4);
        color: #d94242;
        font-weight: 700;
        font-size: 1.6rem;
    }
    .stats-row {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .stat-item {
        background-color: #f9f9f9;
        border-radius: 12px;
        padding: 15px 25px;
        box-shadow: 0 0 10px rgba(255,111,97,0.3);
        text-align: center;
        min-width: 140px;
        flex-grow: 1;
    }
    .stat-item .stat-label {
        font-weight: 700;
        color: #FF6F61;
        font-size: 1.2rem;
        margin-bottom: 8px;
    }
    .stat-item .stat-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #333333;
    }
    .stat-item.positive .stat-number {
        color: #2ECC71;
    }
    .stat-item.negative .stat-number {
        color: #E74C3C;
    }
    .download-buttons {
        display: flex;
        justify-content: center; /* Center the buttons */
        gap: 15px;
        margin-top: 20px;
        flex-wrap: wrap; /* Allow buttons to wrap on smaller screens */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("🔍 **Navigasi**")
menu = st.sidebar.radio("Pilih Halaman", ["🏠 Beranda", "📊 Analisis Sentimen"])

if menu == "🏠 Beranda":
    st.markdown(
        """
        <div style="background-color:#4682B4; border-radius:8px; padding: 10px 0px; margin-bottom:20px; text-align:center; color:white; font-weight:bold; font-size:22px;">
        🔍 <b>Analisis Sentimen</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        Selamat datang di aplikasi Analisis Sentimen berbasis RNN!
        Aplikasi ini memungkinkan Anda mengunggah file CSV berisi teks dan melakukan analisis sentimen secara otomatis.
        """
    )
    st.markdown("### 📌 Fitur Utama:")
    st.markdown(
        """
        - 🚀 Unggah file anda dengan format .CSV
        - 🐱 Analisis sentimen menggunakan model RNN
        - 📥 Unduh hasil model sentimen
        """
    )
    st.markdown(
        """
        Klik tab "📊 Analisis Sentimen" di sidebar untuk memulai analisis!
        """
    )
elif menu == "📊 Analisis Sentimen":
    st.sidebar.header("Pengaturan Model & File")
    hidden_size = st.sidebar.slider("Ukuran Hidden Layer", min_value=32, max_value=256, value=128, step=16)
    learning_rate = st.sidebar.number_input("Laju Pembelajaran", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f")
    epochs = st.sidebar.number_input("Jumlah Epoch", min_value=1, max_value=100, value=30, step=1)

    uploaded_file = st.sidebar.file_uploader("Unggah CSV: Kolom 'full_text'", type=["csv"], key="csvupload")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Kesalahan saat memuat file: {e}")
            st.stop()

        if 'full_text' not in df.columns:
            st.error("Kolom 'full_text' tidak ditemukan pada file yang diunggah!")
            st.stop()

        stop_words = set(stopwords.words('indonesian'))

        try:
            # Memuat lexicon dengan bobot yang ditentukan (+5 atau -5)
            positive_lexicon = load_lexicon('datapositif.csv', 'positive')
            negative_lexicon = load_lexicon('datanegatif.csv', 'negative')
        except Exception as e:
            st.error(f"Kesalahan memuat lexicon default: {e}")
            st.stop()

        @st.cache_data(show_spinner=False)
        def preprocess_and_label(dataframe, stop_words, pos_lex, neg_lex):
            df_proc = dataframe.copy()
            df_proc['preprocessed_text'] = df_proc['full_text'].apply(lambda x: preprocess_text(x, stop_words))
            # Menggabungkan lexicon yang sudah memiliki bobot +5 dan -5
            sentiment_lexicon = merge_lexicons(pos_lex, neg_lex)
            df_proc['label'] = df_proc['preprocessed_text'].apply(lambda x: label_review(x, sentiment_lexicon))

            total_data = len(df_proc)
            positive_count = (df_proc['label'] == 1).sum()
            negative_count = (df_proc['label'] == 0).sum()
            return df_proc, sentiment_lexicon, total_data, positive_count, negative_count

        df_proc, sentiment_lexicon, total_data, positive_count, negative_count = preprocess_and_label(df, stop_words, positive_lexicon, negative_lexicon)

        st.markdown('<div class="box-container">', unsafe_allow_html=True)
        st.subheader("📝 Hasil Preprocessing dan Pelabelan")
        
        # Membuat DataFrame untuk tampilan dengan label teks
        df_display = df_proc.copy()
        df_display['Sentimen'] = df_display['label'].map({1: 'Positif', 0: 'Negatif'})
        
        # Fungsi untuk styling sentimen
        def color_sentiment(val):
            color = '#2ECC71' if val == 'Positif' else '#E74C3C'
            return f'color: white; background-color: {color}; font-weight: bold'
        
        # Tampilkan DataFrame dengan styling
        st.dataframe(
            df_display[['preprocessed_text', 'Sentimen']].rename(columns={'preprocessed_text': 'Teks yang Diproses'})
            .style.applymap(color_sentiment, subset=['Sentimen']),
            height=400
        )

        # Statistik data
        st.markdown(
            f'''
            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-label">Total Data</div>
                    <div class="stat-number">{total_data}</div>
                </div>
                <div class="stat-item positive">
                    <div class="stat-label">Positif</div>
                    <div class="stat-number">{positive_count}</div>
                </div>
                <div class="stat-item negative">
                    <div class="stat-label">Negatif</div>
                    <div class="stat-number">{negative_count}</div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Visualization section
        positive_text = " ".join(df_proc[df_proc['label'] == 1]['preprocessed_text'])
        negative_text = " ".join(df_proc[df_proc['label'] == 0]['preprocessed_text'])

        # Positive word cloud
        st.markdown('<div class="box-container">', unsafe_allow_html=True)
        st.subheader("☁️ WordCloud Positif")
        display_wordcloud(positive_text, "Kata-Kata Positif")
        display_word_frequency(positive_text, "Frekuensi Kata Positif (Top 10)")
        st.markdown('</div>', unsafe_allow_html=True)

        # Negative word cloud
        st.markdown('<div class="box-container">', unsafe_allow_html=True)
        st.subheader("☁️ WordCloud Negatif")
        display_wordcloud(negative_text, "Kata-Kata Negatif")
        display_word_frequency(negative_text, "Frekuensi Kata Negatif (Top 10)")
        st.markdown('</div>', unsafe_allow_html=True)

        # TF-IDF and Model Training
        with st.spinner("Menghitung TF-IDF..."):
            tfidf_df, vectorizer = compute_tfidf_weights_sklearn(df_proc['preprocessed_text'].tolist())
            tfidf_df['label'] = df_proc['label'].values

        start_train = st.button("▶️ Mulai Training Model RNN")

        if start_train:
            X = tfidf_df.drop(columns=['label']).values
            y = tfidf_df['label'].values
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)

            X_train, X_test, y_train, y_test = train_test_split(
                X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor
            )

            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            input_size = X_train.shape[1]
            output_size = len(np.unique(y))
            model = CustomRNN(input_size, hidden_size, output_size)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            st.markdown('<div class="box-container">', unsafe_allow_html=True)
            st.subheader("🚀 Training Model RNN")

            progress_bar = st.progress(0)
            metric_container = st.container()

            accuracy = 0.0
            y_true, y_pred = [], []
            total_losses = []
            test_accs = []

            with st.spinner("Training model sedang berlangsung..."):
                for epoch in range(epochs):
                    model.train()
                    total_loss = 0
                    for X_batch, y_batch in train_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    total_losses.append(total_loss / len(train_loader))

                    model.eval()
                    correct, total = 0, 0
                    y_true.clear()
                    y_pred.clear()
                    with torch.no_grad():
                        for X_batch, y_batch in test_loader:
                            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                            outputs = model(X_batch)
                            predicted = torch.argmax(outputs, dim=1)
                            y_true.extend(y_batch.cpu().numpy())
                            y_pred.extend(predicted.cpu().numpy())
                            correct += torch.eq(predicted, y_batch).sum().item()
                            total += y_batch.size(0)

                    accuracy = correct / total
                    test_accs.append(accuracy)

                    progress_bar.progress((epoch + 1) / epochs)
                    metric_container.markdown(
                        f"""
                        <div class="train-metrics">
                            <div class="metric-box">
                                <div class="metric-title">Epoch</div>
                                <div class="metric-value">{epoch + 1} / {epochs}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-title">Loss</div>
                                <div class="metric-value">{total_losses[-1]:.6f}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-title">Akurasi</div>
                                <div class="metric-value green">{accuracy:.4f}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown(
                f"""
                <div class="training-complete">
                    🎉 Training selesai! Akurasi pada data test: {accuracy:.4f}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Plot training progress
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epochs + 1), total_losses, label='Training Loss', color='#FF6F61')
            plt.plot(range(1, epochs + 1), test_accs, label='Test Accuracy', color='#2ECC71')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Training Progress Metrics')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            buffer_png = io.BytesIO()
            plt.savefig(buffer_png, format='png', dpi=300, bbox_inches='tight')
            buffer_png.seek(0)

            buffer_model = io.BytesIO()
            torch.save(model.state_dict(), buffer_model)
            buffer_model.seek(0)

            st.markdown('<div class="download-buttons">', unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Download Training Metrics (PNG)",
                data=buffer_png,
                file_name="training_metrics.png",
                mime="image/png"
            )
            st.download_button(
                label="⬇️ Download Trained Model (PTH)",
                data=buffer_model,
                file_name="sentiment_model.pth",
                mime="application/octet-stream"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # Confusion Matrix and Classification Report
            st.markdown('<div class="box-container">', unsafe_allow_html=True)
            cm = confusion_matrix(y_true, y_pred)
            st.subheader("📉 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',
                        xticklabels=['Negatif', 'Positif'], 
                        yticklabels=['Negatif', 'Positif'],
                        ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title("Confusion Matrix")
            st.pyplot(fig)

            st.subheader("📃 Classification Report")
            report = classification_report(y_true, y_pred, target_names=['Negatif', 'Positif'])
            st.text(report)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Unggah file CSV untuk memulai analisis.")
