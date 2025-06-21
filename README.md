# 🧠 Sentiment Analysis Dashboard

An interactive Streamlit-based web application for performing multilingual sentiment analysis, keyword extraction, and visualizing results from text data. Designed for analyzing customer reviews, feedback, or any user-generated content in a simple, no-code environment.

---

## 🚀 Features

- ✅ Text input (typed or file upload: `.txt` or `.csv`)
- ✅ Multilingual sentiment analysis using Hugging Face Transformers
- ✅ Keyword extraction with RAKE (customizable stopwords)
- ✅ Visual summaries: Bar chart, Pie chart, and Word Cloud
- ✅ Download results in CSV and PDF format
- ✅ Smart session management and UI optimizations

---

## 🖼️ Demo



---

## 🔧 Tech Stack

- **Frontend & UI**: [Streamlit](https://streamlit.io/)
- **NLP Model**: [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
- **Keyword Extraction**: RAKE (via `rake_nltk`)
- **Visualization**: Matplotlib & WordCloud
- **PDF Export**: ReportLab
- **CSV Export**: Pandas

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-dashboard.git
   cd sentiment-analysis-dashboard
2. Install all dependencies
   Make sure you have Python installed. Then run:
   bash
    pip install -r requirements.txt

3.Run the app using Streamlit
      bash
       streamlit run app.py
    If your file is named app3.py, use:

    streamlit run app3.py
