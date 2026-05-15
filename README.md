# Emotion Detection in Music Lyrics

A Flask-based web application that uses machine learning and NLP to detect emotions and moods in music lyrics.

## 🎯 Project Overview

This project analyzes music lyrics to identify emotional content using a hybrid approach combining:
- **Machine Learning models** (Logistic Regression with TF-IDF + POS features)
- **Lexicon-based analysis** (NRC Emotion Lexicon)
- **Dual detection**: Separate models for emotions and moods

## ✨ Features

### Core Capabilities
- **Emotion Detection**: Identifies 10 emotions (joy, sadness, anger, fear, trust, disgust, surprise, anticipation, love, motivation)
- **Mood Detection**: Identifies 9 moods (happy, sad, angry, anxious, calm, relaxed, energetic, romantic, motivated)
- **Hybrid Analysis**: Combines ML predictions (70%) with lexicon-based scoring (30%)
- **Timeline Visualization**: Segments lyrics and shows emotion/mood changes across sections
- **Line-by-Line Analysis**: Annotates individual lines with matched emotion words
- **Music Recommendations**: Suggests playlist based on detected emotion
- **User Authentication**: Secure login/signup system with persistent history
- **Analysis History**: Save and review past analyses per user

### Technical Features
- **POS Tagging**: Part-of-speech distribution features for better context understanding
- **Data Augmentation**: Synonym replacement using WordNet for training data expansion
- **Sparse & Dense Features**: Supports both traditional ML and sentence transformer embeddings
- **Per-Class Metrics**: Detailed confusion matrix and classification reports saved for each model
- **Responsive UI**: Modern dark-themed interface with gradient accents and Chart.js visualizations

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │
│  (HTML/CSS/JS)  │
│   Chart.js UI   │
└────────┬────────┘
         │ REST API
┌────────▼────────┐
│   Flask App     │
│   (app.py)      │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼──────┐
│Emotion│  │  Mood   │
│ Model │  │  Model  │
└───┬──┘  └──┬──────┘
    │        │
    └────────┴─────┐
                   │
         ┌─────────▼──────────┐
         │  NRC Lexicon +     │
         │  POS Transformer   │
         └────────────────────┘
```

## 📁 Project Structure

```
emotion-project/
├── app.py                      # Main Flask application (1275 lines)
├── train_model.py              # Model training script with augmentation
├── features.py                 # POS tag transformer (reusable module)
├── requirements.txt            # Python dependencies (pinned versions)
├── ENVIRONMENT.md              # Environment configuration notes
├── users.db                    # SQLite database for user accounts & history
├── emotion_model.pkl           # Trained emotion classifier
├── mood_model.pkl              # Trained mood classifier
├── *.metrics.json              # Per-model metrics (accuracy, confusion matrix)
├── templates/
│   ├── index.html              # Main analysis UI (1158 lines)
│   ├── login.html              # Login form
│   └── signup.html             # Registration form
└── tests/
    ├── run_tests.py            # Test runner
    └── test_love_detection.py  # Specific love/motivation detection tests
```

## 🛠️ Technology Stack

### Backend
- **Flask** 2.3.3 - Web framework
- **scikit-learn** 1.3.2 - Machine learning pipeline
- **nltk** 3.8.1 - NLP processing (tokenization, POS tagging, WordNet)
- **numpy** 1.24.3 - Numerical operations
- **pandas** 2.0.3 - Data handling
- **joblib** 1.3.2 - Model serialization
- **python-dotenv** 1.0.0 - Environment variables
- **Werkzeug** 2.3.7 - Security utilities

### Frontend
- **HTML5/CSS3** - Custom styling with CSS variables
- **JavaScript (Vanilla)** - Async API calls, DOM manipulation
- **Chart.js** 3.9.1 - Emotion/mood distribution charts
- **Responsive Design** - Mobile-friendly grid layout

### ML/NLP Components
- **TF-IDF Vectorizer** - Text feature extraction (max 6000 features)
- **PosTagCountsTransformer** - Custom POS distribution features (9 buckets)
- **Logistic Regression** - Multi-class classifier (multinomial, LBFGS)
- **EmbeddingTransformer** - Optional Sentence-BERT integration (`all-MiniLM-L6-v2`)
- **GridSearchCV** - Hyperparameter tuning (C values: 0.1, 1.0, 10.0)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager
- Git (for cloning)

### Installation

1. **Clone or navigate to project directory**
   ```bash
   cd d:\emotion-project
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (automatic on first run)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('wordnet')
   ```

5. **Train models** (if not present)
   ```bash
   python train_model.py
   ```
   This generates `emotion_model.pkl` and `mood_model.pkl` with metrics files.

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the web interface**
   - Open browser: `http://localhost:5000`
   - Create account or login
   - Paste lyrics or provide URL for analysis

## 📊 Model Performance

### Training Details
- **Dataset Size**: 40+ samples per model (augmented 4x with synonym replacement)
- **Train/Test Split**: 80/20 stratified split
- **Cross-Validation**: 5-fold CV during training
- **Target Accuracy**: ≥95%

### Saved Metrics
Each model saves a `.metrics.json` file containing:
- Test accuracy
- Cross-validation mean ± std
- Per-class precision/recall/F1-score
- Confusion matrix
- Label ordering

### Feature Pipeline
```
Text → [TF-IDF (6000 feats) + POS Distribution (9 feats)] → Logistic Regression → Prediction
```

Optional: Replace TF-IDF+POS with **Sentence-BERT embeddings** if `sentence-transformers` is installed.

## 🔌 API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | ✅ | Main analysis page |
| `/login` | GET/POST | ❌ | User login |
| `/logout` | GET | ✅ | User logout |
| `/signup` | GET/POST | ❌ | User registration |
| `/api/analyze` | POST | ✅ | Analyze lyrics (emotion + mood) |
| `/api/analyze_mood` | POST | ✅ | Analyze mood only |
| `/api/health` | GET | ❌ | Health check |
| `/api/analyses` | GET | ✅ | List user's analysis history |
| `/api/analyses/<id>` | GET | ✅ | Get specific analysis |
| `/api/analyses/<id>` | DELETE | ✅ | Delete analysis |

### Example API Request
```json
POST /api/analyze
{
  "lyrics": "I love you so much, my heart is full of joy..."
}
```

### Example API Response
```json
{
  "primary_emotion": "love",
  "confidence": 0.87,
  "all_emotions": {"love": 0.87, "joy": 0.65, ...},
  "primary_mood": "romantic",
  "mood_confidence": 0.82,
  "matched_words": ["love", "heart", "joy"],
  "recommendations": [...],
  "timeline": [...],
  "line_analysis": [...]
}
```

## 🎨 UI Features

- **Dark Theme**: Radial gradient background with purple/indigo/cyan accents
- **Interactive Cards**: Hover effects with elevated shadows
- **Real-time Charts**: Emotion/mood distribution pie/bar charts
- **Timeline View**: Scrollable segments showing emotion progression
- **Keyword Highlighting**: Matched emotion/mood words color-coded
- **Playlist Display**: Song recommendations by detected emotion
- **Responsive Layout**: 2-column grid on desktop, stacked on mobile

## 🧪 Testing

Run the test suite:
```bash
python tests/run_tests.py
```

Specific tests:
```bash
python tests/test_love_detection.py
```

Tests verify:
- Love/motivation detection accuracy
- Model loading
- API endpoint responses
- Edge cases (short lyrics, empty input)

## 📈 Future Enhancements

- [ ] Expand training dataset with real song lyrics
- [ ] Add support for multiple languages
- [ ] Integrate Spotify/Genius API for real-time lyrics
- [ ] Deploy to cloud (Heroku, AWS, Vercel)
- [ ] Add user sharing/social features
- [ ] Implement deep learning models (BERT, RoBERTa)
- [ ] Batch analysis for multiple songs
- [ ] Export analysis reports (PDF, CSV)

## 🔒 Security Notes

- Passwords hashed with Werkzeug's `generate_password_hash`
- Flask sessions secured with random SECRET_KEY
- CSRF protection via Flask-WTF (if added)
- Input validation on all endpoints
- Max content length: 16MB
- SQLite parameterized queries prevent SQL injection

## 📝 Database Schema

### Users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
);
```

### Analyses Table
```sql
CREATE TABLE analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    lyrics TEXT NOT NULL,
    primary_emotion TEXT,
    emotion_confidence REAL,
    primary_mood TEXT,
    mood_confidence REAL,
    overall_tone TEXT,
    emotion_keywords TEXT,
    mood_keywords TEXT,
    key_phrases TEXT,
    created_at TEXT
);
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is open source and available for educational purposes.

## 👥 Authors

- AI Assistant - Initial development and architecture

## 🙏 Acknowledgments

- **NRC Emotion Lexicon** - Emotion word associations
- **NLTK Team** - NLP tools and corpora
- **scikit-learn Team** - ML framework
- **Flask Team** - Web framework
- **Chart.js Team** - Visualization library

---

**Built with ❤️ for music emotion analysis**
