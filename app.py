"""
Flask Backend for Emotion Detection in Music Lyrics
Author: AI Assistant
Description: REST API for emotion detection using ML + lexicon-based approach
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import json
from pathlib import Path
from html import unescape
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import warnings
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
warnings.filterwarnings('ignore')
import requests
import os
import json
from dotenv import load_dotenv

# --- START: FLASK-LOGIN IMPORTS ---
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user # type: ignore
# --- END: FLASK-LOGIN IMPORTS ---

# Download NLTK data (first run)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max file size
# --- START: FLASK-LOGIN CONFIGURATION ---
load_dotenv()
secret = os.environ.get('SECRET_KEY')
if not secret:
    secret = os.urandom(24).hex()
    print("Warning: SECRET_KEY not set; using a generated key (sessions reset on restart)")
app.config['SECRET_KEY'] = secret
# --- END: FLASK-LOGIN CONFIGURATION ---

# --- START: DATABASE SETUP ---
DB_PATH = Path('users.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
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
    """)
    conn.commit()
    cols = {row['name'] for row in conn.execute("PRAGMA table_info('analyses')").fetchall()}
    need = [
        ('lyrics', "ALTER TABLE analyses ADD COLUMN lyrics TEXT"),
        ('primary_emotion', "ALTER TABLE analyses ADD COLUMN primary_emotion TEXT"),
        ('emotion_confidence', "ALTER TABLE analyses ADD COLUMN emotion_confidence REAL"),
        ('primary_mood', "ALTER TABLE analyses ADD COLUMN primary_mood TEXT"),
        ('mood_confidence', "ALTER TABLE analyses ADD COLUMN mood_confidence REAL"),
        ('overall_tone', "ALTER TABLE analyses ADD COLUMN overall_tone TEXT"),
        ('emotion_keywords', "ALTER TABLE analyses ADD COLUMN emotion_keywords TEXT"),
        ('mood_keywords', "ALTER TABLE analyses ADD COLUMN mood_keywords TEXT"),
        ('key_phrases', "ALTER TABLE analyses ADD COLUMN key_phrases TEXT"),
        ('created_at', "ALTER TABLE analyses ADD COLUMN created_at TEXT")
    ]
    for name, stmt in need:
        if name not in cols:
            conn.execute(stmt)
    conn.commit()
    conn.close()

init_db()
# --- END: DATABASE SETUP ---


from features import PosTagCountsTransformer

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Sets the function name for the login view


class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

    def get_id(self):
        return str(self.id)


def get_user_by_username(username: str):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return user


def get_user_by_id(user_id: str):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return user


def create_user(username: str, password: str):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (username, generate_password_hash(password))
    )
    conn.commit()
    conn.close()

def save_analysis(user_id: int, lyrics: str, data: dict) -> int:
    conn = get_db_connection()
    cols = {row['name'] for row in conn.execute("PRAGMA table_info('analyses')").fetchall()}
    fields = ['user_id', 'lyrics']
    values = [user_id, lyrics]
    if 'primary_emotion' in cols:
        fields.append('primary_emotion')
        values.append(data.get('primary_emotion'))
    val_conf = float(data.get('confidence', 0) or 0)
    if 'emotion_confidence' in cols:
        fields.append('emotion_confidence')
        values.append(val_conf)
    if 'confidence' in cols:
        fields.append('confidence')
        values.append(val_conf)
    if 'primary_mood' in cols:
        fields.append('primary_mood')
        values.append(data.get('primary_mood'))
    if 'mood_confidence' in cols:
        fields.append('mood_confidence')
        values.append(float(data.get('mood_confidence', 0) or 0))
    if 'overall_tone' in cols:
        fields.append('overall_tone')
        values.append(data.get('overall_tone'))
    if 'emotion_keywords' in cols:
        fields.append('emotion_keywords')
        values.append(json.dumps(data.get('emotion_keywords') or []))
    if 'mood_keywords' in cols:
        fields.append('mood_keywords')
        values.append(json.dumps(data.get('mood_keywords') or []))
    if 'key_phrases' in cols:
        fields.append('key_phrases')
        values.append(json.dumps(data.get('key_phrases') or []))
    sql = f"INSERT INTO analyses ({', '.join(fields)}, created_at) VALUES ({', '.join(['?']*len(values))}, datetime('now'))"
    conn.execute(sql, tuple(values))
    conn.commit()
    row = conn.execute("SELECT last_insert_rowid() AS id").fetchone()
    conn.close()
    return int(row['id'])

def list_analyses(user_id: int, limit: int = 10):
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT id, primary_emotion, emotion_confidence, primary_mood, mood_confidence, overall_tone, created_at FROM analyses WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_analysis(user_id: int, analysis_id: int):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM analyses WHERE user_id = ? AND id = ?",
        (user_id, analysis_id)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        'id': row['id'],
        'lyrics': row['lyrics'],
        'primary_emotion': row['primary_emotion'],
        'emotion_confidence': row['emotion_confidence'],
        'primary_mood': row['primary_mood'],
        'mood_confidence': row['mood_confidence'],
        'overall_tone': row['overall_tone'],
        'emotion_keywords': json.loads(row['emotion_keywords'] or '[]'),
        'mood_keywords': json.loads(row['mood_keywords'] or '[]'),
        'key_phrases': json.loads(row['key_phrases'] or '[]'),
        'created_at': row['created_at']
    }

def delete_analysis(user_id: int, analysis_id: int) -> bool:
    conn = get_db_connection()
    cur = conn.execute(
        "DELETE FROM analyses WHERE user_id = ? AND id = ?",
        (user_id, analysis_id)
    )
    conn.commit()
    conn.close()
    return cur.rowcount > 0


# User loader function required by Flask-Login
@login_manager.user_loader
def load_user(user_id):
    user = get_user_by_id(user_id)
    if user:
        return User(user['id'], user['username'])
    return None


# --- START: POS transformer for loaded model ---
POS_BUCKETS = ["N", "V", "J", "R", "P", "M", "D", "W", "O"]  # Noun, Verb, Adj, Adv, Pron, Num, Det, Wh, Other


def bucket_pos(tag: str) -> str:
    if tag.startswith("N"):
        return "N"
    if tag.startswith("V"):
        return "V"
    if tag.startswith("J"):
        return "J"
    if tag.startswith("R"):
        return "R"
    if tag.startswith(("PRP", "WP")):
        return "P"
    if tag.startswith(("CD",)):
        return "M"
    if tag.startswith(("DT", "PDT")):
        return "D"
    if tag.startswith(("W",)):
        return "W"
    return "O"


class PosTagCountsTransformer:
    """Convert text to POS tag distribution (normalized counts) as sparse features."""

    def __init__(self):
        self.buckets = POS_BUCKETS
        self.bucket_index = {b: i for i, b in enumerate(self.buckets)}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            tokens = word_tokenize(text)
            tags = nltk.pos_tag(tokens)
            counts = Counter(bucket_pos(tag) for _, tag in tags)
            total = sum(counts.values())
            vec = np.zeros(len(self.buckets), dtype=float)
            if total > 0:
                for bucket, count in counts.items():
                    idx = self.bucket_index.get(bucket, None)
                    if idx is not None:
                        vec[idx] = count / total
            rows.append(vec)
        data = np.vstack(rows) if rows else np.zeros((0, len(self.buckets)))
        return csr_matrix(data)

# --- END: POS transformer for loaded model ---

# Load trained ML model
# Compatibility shim: some models were trained when PosTagCountsTransformer was defined
# in __main__, so make sure a reference exists for unpickling.
import sys
try:
    import features
    setattr(sys.modules.get('__main__'), 'PosTagCountsTransformer', features.PosTagCountsTransformer)
except Exception:
    pass

if os.environ.get('SKIP_MODEL_LOAD') == '1':
    model = None
else:
    try:
        model = joblib.load('emotion_model.pkl')
        print("Emotion detection model loaded successfully")
    except FileNotFoundError:
        print("✗ Error: emotion_model.pkl not found. Please train the model first.")
        model = None

if os.environ.get('SKIP_MODEL_LOAD') == '1':
    mood_model = None
else:
    try:
        mood_model = joblib.load('mood_model.pkl')
        print("Mood detection model loaded successfully")
    except FileNotFoundError:
        print("✗ Error: mood_model.pkl not found. Please train the model first.")
        mood_model = None

# NRC Emotion Lexicon (simplified version)
nrc_emotions = {
# extended to include distinct 'love' and 'motivation'
    'love': ['love'],
    'beloved': ['love'],
    'darling': ['love'],
    'affection': ['love'],
    'adore': ['love'],
    'romance': ['love'],
    'happy': ['joy'],
    'joy': ['joy'],
    'wonderful': ['joy'],
    'amazing': ['joy'],
    'beautiful': ['joy'],
    'fantastic': ['joy'],
    'excellent': ['joy'],
    'great': ['joy'],
    'good': ['joy'],
    'glad': ['joy'],
    'cheerful': ['joy'],
    'delighted': ['joy'],
    'bliss': ['joy'],
    'delight': ['joy'],
    'pleasure': ['joy'],
    'grateful': ['joy'],
    
    'sad': ['sadness'],
    'sadness': ['sadness'],
    'tears': ['sadness'],
    'crying': ['sadness'],
    'depressed': ['sadness'],
    'terrible': ['sadness'],
    'devastated': ['sadness'],
    'broken': ['sadness'],
    'lonely': ['sadness'],
    'abandoned': ['sadness'],
    'despair': ['sadness'],
    'sorrow': ['sadness'],
    'grief': ['sadness'],
    'regret': ['sadness'],
    'disappointed': ['sadness'],
    'empty': ['sadness'],
    'numb': ['sadness'],
    'anguish': ['sadness'],
    
    'angry': ['anger'],
    'anger': ['anger'],
    'hate': ['anger'],
    'mad': ['anger'],
    'furious': ['anger'],
    'rage': ['anger'],
    'livid': ['anger'],
    'enraged': ['anger'],
    'frustrated': ['anger'],
    'outrageous': ['anger'],
    'irritated': ['anger'],
    'annoyed': ['anger'],
    
    'fear': ['fear'],
    'scared': ['fear'],
    'afraid': ['fear'],
    'terrified': ['fear'],
    'frightened': ['fear'],
    'dread': ['fear'],
    'panic': ['fear'],
    'anxiety': ['fear'],
    'anxious': ['fear'],
    'worried': ['fear'],
    'terror': ['fear'],
    'uncertainty': ['fear'],
    
    'trust': ['trust'],
    'believe': ['trust'],
    'confidence': ['trust'],
    'faithful': ['trust'],
    'loyal': ['trust'],
    
    'disgust': ['disgust'],
    'disgusting': ['disgust'],
    'revolting': ['disgust'],
    'repulsive': ['disgust'],
    'contempt': ['disgust'],
    'vile': ['disgust'],
    
    'surprise': ['surprise'],
    'surprised': ['surprise'],
    'unexpected': ['surprise'],
    'amazed': ['surprise'],
    'astonished': ['surprise'],
    'amazement': ['surprise'],
    'wonder': ['surprise'],
    'wonderful': ['surprise', 'joy'],
    
    'anticipation': ['anticipation'],
    'anticipate': ['anticipation'],
    'optimism': ['anticipation'],
    'optimistic': ['anticipation'],
    'hope': ['anticipation'],
    'hopeful': ['anticipation'],
    'excited': ['anticipation', 'joy'],
    'excitement': ['anticipation', 'joy'],
    'enthusiasm': ['anticipation', 'joy'],
    'motivation': ['motivation'],
    'motivated': ['motivation'],
    'ambition': ['motivation'],
    'ambitious': ['motivation'],
    'drive': ['motivation'],
    'driven': ['motivation'],
    'inspire': ['motivation'],
    'inspired': ['motivation'],
    'aspire': ['motivation'],
    'goal': ['motivation'],
    'achieve': ['motivation'],
    'kiss': ['love'],
    'baby': ['love'],
    'real': ['love'],
}

mood_lexicon = {
    'happy': ['happy'],
    'joy': ['happy'],
    'smile': ['happy'],
    'laugh': ['happy'],
    'sad': ['sad'],
    'cry': ['sad'],
    'tears': ['sad'],
    'heartbroken': ['sad'],
    'angry': ['angry'],
    'rage': ['angry'],
    'furious': ['angry'],
    'irritated': ['angry'],
    'anxious': ['anxious'],
    'worried': ['anxious'],
    'nervous': ['anxious'],
    'tense': ['anxious'],
    'calm': ['calm'],
    'peaceful': ['calm'],
    'serene': ['calm'],
    'relaxed': ['relaxed'],
    'mellow': ['relaxed'],
    'soothed': ['relaxed'],
    'energetic': ['energetic'],
    'lively': ['energetic'],
    'active': ['energetic'],
    'romantic': ['romantic'],
    'love': ['romantic'],
    'affection': ['romantic'],
    'tender': ['romantic'],
    # motivation-related tokens mapped to the 'motivated' mood
    'motivation': ['motivated'],
    'motivated': ['motivated'],
    'drive': ['motivated'],
    'driven': ['motivated'],
    'ambition': ['motivated'],
    'ambitious': ['motivated'],
    'inspire': ['motivated'],
    'inspired': ['motivated'],
    'goal': ['motivated'],
    'achieve': ['motivated']
}

# Emotion color mapping for UI
emotion_colors = {
# ... (rest of emotion_colors dictionary is unchanged) ...
    'joy': '#FFD700',
    'sadness': '#4169E1',
    'anger': '#FF6347',
    'fear': '#8B008B',
    'trust': '#32CD32',
    'disgust': '#9370DB',
    'surprise': '#FF8C00',
    'anticipation': '#20B2AA',
    'love': '#FF1493',
    'motivation': '#00CED1'
}

# Playlist recommendations by emotion
playlists = {
# ... (rest of playlists dictionary is unchanged) ...
    'joy': [
        {'song': 'Walking on Sunshine', 'artist': 'Katrina & The Waves'},
        {'song': 'Don\'t Stop Me Now', 'artist': 'Queen'},
        {'song': 'Here Comes the Sun', 'artist': 'The Beatles'},
        {'song': 'Good as Hell', 'artist': 'Lizzo'},
    ],
    'sadness': [
        {'song': 'Someone Like You', 'artist': 'Adele'},
        {'song': 'The Scientist', 'artist': 'Coldplay'},
        {'song': 'Mad World', 'artist': 'Gary Jules'},
        {'song': 'Hurt', 'artist': 'Johnny Cash'},
    ],
    'anger': [
        {'song': 'Break Stuff', 'artist': 'Limp Bizkit'},
        {'song': 'My Heart Will Go On', 'artist': 'Rage Against the Machine'},
        {'song': 'Before I Forget', 'artist': 'Slipknot'},
        {'song': 'Killing in the Name', 'artist': 'Rage Against the Machine'},
    ],
    'fear': [
        {'song': 'Thriller', 'artist': 'Michael Jackson'},
        {'song': 'Monster', 'artist': 'Skillet'},
        {'song': 'Paranoia', 'artist': 'The Black Eyed Peas'},
        {'song': 'Haunted', 'artist': 'Evanescence'},
    ],
    'trust': [
        {'song': 'I Will Follow You into the Dark', 'artist': 'Death Cab for Cutie'},
        {'song': 'Stand by Me', 'artist': 'Ben E. King'},
        {'song': 'You\'re My Best Friend', 'artist': 'Queen'},
        {'song': 'I Got You Babe', 'artist': 'Sonny & Cher'},
    ],
    'disgust': [
        {'song': 'Zombie', 'artist': 'The Cranberries'},
        {'song': 'Dumpweed', 'artist': 'blink-182'},
        {'song': 'Sick of It All', 'artist': 'Gwar'},
        {'song': 'Filthy', 'artist': 'Justin Timberlake'},
    ],
    'surprise': [
        {'song': 'Uptown Funk', 'artist': 'Bruno Mars & Mark Ronson'},
        {'song': 'Happy', 'artist': 'Pharrell Williams'},
        {'song': 'Walking on Sunshine', 'artist': 'Katrina & The Waves'},
        {'song': 'Shut Up and Dance', 'artist': 'Walk the Moon'},
    ],
    'anticipation': [
        {'song': 'Eye of the Tiger', 'artist': 'Survivor'},
        {'song': 'Waiting for the Sun', 'artist': 'The Doors'},
        {'song': 'The Wait', 'artist': 'Vinilo vs Eastcolors'},
        {'song': 'Anticipation', 'artist': 'Carly Simon'},
    ],
    'love': [
        {'song': 'All of Me', 'artist': 'John Legend'},
        {'song': 'Make You Feel My Love', 'artist': 'Adele'},
        {'song': 'Thinking Out Loud', 'artist': 'Ed Sheeran'},
        {'song': 'Unchained Melody', 'artist': 'The Righteous Brothers'},
    ],
    'motivation': [
        {'song': 'Stronger', 'artist': 'Kanye West'},
        {'song': 'Lose Yourself', 'artist': 'Eminem'},
        {'song': 'Don’t Stop Believin’', 'artist': 'Journey'},
        {'song': 'Hall of Fame', 'artist': 'The Script'},
    ]
}


def preprocess_lyrics(lyrics):
# ... (rest of preprocess_lyrics function is unchanged) ...
    """Preprocess lyrics: clean, tokenize, remove stopwords"""
    # Convert to lowercase
    lyrics = lyrics.lower()
    
    # Remove special characters and digits
    lyrics = re.sub(r'[^a-zA-Z\s]', '', lyrics)
    
    # Tokenize
    tokens = word_tokenize(lyrics)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return tokens


def lexicon_based_emotion_scoring(lyrics):
# ... (rest of lexicon_based_emotion_scoring function is unchanged) ...
    """Score emotions using NRC Emotion Lexicon"""
    tokens = preprocess_lyrics(lyrics)
    emotion_scores = {
        'joy': 0, 'sadness': 0, 'anger': 0, 'fear': 0,
        'trust': 0, 'disgust': 0, 'surprise': 0, 'anticipation': 0,
        'love': 0, 'motivation': 0
    }
    
    matched_words = []
    for token in tokens:
        if token in nrc_emotions:
            # add tokens in order, but keep unique list to avoid duplicates in results
            if token not in matched_words:
                matched_words.append(token)
            for emotion in nrc_emotions[token]:
                emotion_scores[emotion] += 1
    
    # Normalize scores
    total = sum(emotion_scores.values())
    if total > 0:
        emotion_scores = {e: s / total for e, s in emotion_scores.items()}
    
    return emotion_scores, matched_words

def lexicon_based_mood_scoring(lyrics):
    tokens = preprocess_lyrics(lyrics)
    mood_scores = {
        'happy': 0, 'sad': 0, 'angry': 0, 'anxious': 0,
        'calm': 0, 'relaxed': 0, 'energetic': 0, 'romantic': 0,
        'motivated': 0
    }
    matched_words = []
    for token in tokens:
        if token in mood_lexicon:
            # keep unique tokens for mood as well
            if token not in matched_words:
                matched_words.append(token)
            for mood in mood_lexicon[token]:
                mood_scores[mood] += 1
    total = sum(mood_scores.values())
    if total > 0:
        mood_scores = {e: s / total for e, s in mood_scores.items()}
    return mood_scores, matched_words


def annotate_lyrics_lines(raw_lyrics: str):
    """
    Return per-line annotations with matched words and emotion meanings.
    Each entry: {
        line_index, text,
        matched_words: [{word, emotions, meaning}],
        all_emotions: {emotion: score},
        dominant_emotion
    }
    """
    lines = [ln.rstrip('\n') for ln in raw_lyrics.splitlines()]
    annotations = []
    meanings = {
        'joy': 'Warmth, positivity, uplift, often tied to love.',
        'sadness': 'Somber, reflective, sense of loss or longing.',
        'anger': 'Tension, frustration, conflict and pushback.',
        'fear': 'Anxious or uneasy, watching for what could go wrong.',
        'trust': 'Steady, loyal, reassuring and supportive.',
        'disgust': 'Aversion or rejection of what feels wrong.',
        'surprise': 'Unexpected shifts, twists, or jolts.',
        'anticipation': 'Forward-looking, excited for what is next.',
        'love': 'Affection, intimacy, and heartfelt connection.',
        'motivation': 'Drive, ambition, and momentum toward goals.'
    }

    for idx, line in enumerate(lines):
        clean = re.sub(r'[^a-zA-Z\s]', ' ', line.lower())
        tokens = clean.split()
        word_hits = []
        per_emotion_counts = {e: 0 for e in meanings.keys()}

        for token in tokens:
            if token in nrc_emotions:
                emos = nrc_emotions[token]
                for emo in emos:
                    per_emotion_counts[emo] = per_emotion_counts.get(emo, 0) + 1
                word_hits.append({
                    'word': token,
                    'emotions': emos,
                    'meaning': "; ".join([meanings.get(e, '') for e in emos if meanings.get(e, '')])
                })

        total = sum(per_emotion_counts.values())
        dominant = None
        all_emotions = {}
        if total > 0:
            all_emotions = {e: c / total for e, c in per_emotion_counts.items()}
            dominant = max(per_emotion_counts.items(), key=lambda x: x[1])[0]
        else:
            all_emotions = {e: 0 for e in meanings.keys()}

        # Keep non-empty lines to give per-line context, even if no matches
        if line.strip():
            annotations.append({
                'line_index': idx,
                'text': line,
                'matched_words': word_hits,
                'dominant_emotion': dominant,
                'all_emotions': all_emotions
            })

    return annotations


def detect_emotion(lyrics):
# ... (rest of detect_emotion function is unchanged) ...
    """Detect emotion using ML model + lexicon-based approach"""
    
    if not model:
        return {
            'error': 'Model not loaded',
            'status': 'error'
        }
    
    # Get ML model predictions
    ml_prediction = model.predict([lyrics])[0]
    ml_probabilities = model.predict_proba([lyrics])[0]
    ml_classes = model.classes_
    
    # Create ML probability dict
    ml_scores = {emotion: float(prob) for emotion, prob in zip(ml_classes, ml_probabilities)}
    
    # Get lexicon-based scores
    lexicon_scores, matched_words = lexicon_based_emotion_scoring(lyrics)
    
    # Combine scores (70% ML, 30% Lexicon)
    combined_scores = {}
    for emotion in ml_scores.keys():
        ml_weight = ml_scores.get(emotion, 0) * 0.7
        lex_weight = lexicon_scores.get(emotion, 0) * 0.3
        combined_scores[emotion] = ml_weight + lex_weight
    
    # Normalize combined scores
    total = sum(combined_scores.values())
    if total > 0:
        combined_scores = {e: s / total for e, s in combined_scores.items()}
    
    # Get primary emotion
    primary_emotion = max(combined_scores.items(), key=lambda x: x[1])[0]
    
    # Sort by score
    sorted_emotions = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get recommendations
    recommendations = playlists.get(primary_emotion, [])
    
    return {
        'primary_emotion': primary_emotion,
        'confidence': combined_scores[primary_emotion],
        'all_emotions': {e: float(s) for e, s in sorted_emotions},
        'ml_scores': ml_scores,
        'lexicon_scores': lexicon_scores,
        'matched_words': matched_words,
        'recommendations': recommendations,
        'emotion_color': emotion_colors.get(primary_emotion, '#808080'),
        'status': 'success'
    }

def detect_mood(lyrics):
    if not mood_model:
        return {
            'error': 'Model not loaded',
            'status': 'error'
        }
    ml_prediction = mood_model.predict([lyrics])[0]
    ml_probabilities = mood_model.predict_proba([lyrics])[0]
    ml_classes = mood_model.classes_
    ml_scores = {m: float(p) for m, p in zip(ml_classes, ml_probabilities)}
    lexicon_scores, matched_words = lexicon_based_mood_scoring(lyrics)
    combined = {}
    for mood in ml_scores.keys():
        combined[mood] = ml_scores.get(mood, 0) * 0.7 + lexicon_scores.get(mood, 0) * 0.3
    total = sum(combined.values())
    if total > 0:
        combined = {e: s / total for e, s in combined.items()}
    primary = max(combined.items(), key=lambda x: x[1])[0]
    sorted_moods = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    mood_descriptions = {
        'happy': 'Upbeat and cheerful, a bright listening state.',
        'sad': 'Tender and reflective, leaning into melancholy.',
        'angry': 'Edgy and intense, cathartic energy.',
        'anxious': 'Restless and tense, heightened alertness.',
        'calm': 'Soft and peaceful, steady breathing.',
        'relaxed': 'Loose and easy, unwinding gently.',
        'energetic': 'Lively and driving, forward momentum.',
        'romantic': 'Warm and intimate, closeness and affection.',
        'motivated': 'Driven and focused, ready to take action toward goals.'
    }
    return {
        'primary_mood': primary,
        'confidence': combined[primary],
        'all_moods': {e: float(s) for e, s in sorted_moods},
        'ml_scores': ml_scores,
        'lexicon_scores': lexicon_scores,
        'matched_words': matched_words,
        'description': mood_descriptions.get(primary, 'Distinct listening mood.'),
        'status': 'success'
    }


def fetch_lyrics_from_url(source_url: str) -> str | None:
    """Best-effort fetch of lyrics from a provided URL (YouTube/Spotify pages)."""
    try:
        resp = requests.get(source_url, timeout=6)
        if resp.status_code != 200 or not resp.text:
            return None

        html_text = resp.text

        # Replace common line-break tags with newlines
        html_text = re.sub(r'(?i)<br\s*/?>', '\n', html_text)
        html_text = re.sub(r'(?i)</p>', '\n', html_text)

        # Drop scripts/styles to avoid noise
        html_text = re.sub(r'(?is)<(script|style)[^>]*>.*?</\1>', '', html_text)

        # Strip remaining tags
        text = re.sub(r'<[^>]+>', '\n', html_text)
        text = unescape(text)

        # Collapse whitespace and keep reasonable newlines
        lines = [line.strip() for line in text.splitlines()]
        cleaned = "\n".join([ln for ln in lines if ln])

        # Heuristic: require minimal length to consider it lyrics
        if len(cleaned) < 120:
            return None

        return cleaned
    except Exception:
        return None


def segment_lyrics_auto(lyrics):
# ... (rest of helper functions are unchanged) ...
    """Auto-segment lyrics by paragraph, falling back to line chunks."""
    paragraphs = [block.strip() for block in re.split(r'\n\s*\n', lyrics) if block.strip()]
    if paragraphs:
        return paragraphs
    
    # Fallback: chunk by lines (4 lines per segment)
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    chunk_size = 4
    segments = [
        "\n".join(lines[i:i + chunk_size])
        for i in range(0, len(lines), chunk_size)
    ]
    
    return segments if segments else [lyrics.strip()]


def describe_emotion(primary, confidence, matched_words):
    """Create a richer natural-language description for a segment."""
    descriptions = {
        'joy': 'Radiates warmth and positivity, hinting at affection or uplift.',
        'sadness': 'Carries a somber, reflective tone with a sense of loss or longing.',
        'anger': 'Feels tense and forceful, expressing frustration or conflict.',
        'fear': 'Conveys anxiety or unease, watching for what might go wrong.',
        'trust': 'Sounds reassuring and steady, leaning on loyalty and support.',
        'disgust': 'Shows aversion or rejection, pushing back against what feels wrong.',
        'surprise': 'Highlights unexpected turns or shifts in feeling.',
        'anticipation': 'Builds forward-looking energy, expecting what comes next.',
        'love': 'Centers on affection, closeness, and heartfelt connection.',
        'motivation': 'Signals drive and forward motion toward goals.'
    }

    base = descriptions.get(primary, 'Carries a distinct emotional tone.')

    # Emphasize love themes when present in joy-related text.
    if primary == 'joy' and any(word in matched_words for word in ['love', 'loving', 'beloved']):
        base = 'Celebrates love and warmth, leaning into affectionate, hopeful energy.'

    unique_kw = []
    for w in matched_words:
        if w not in unique_kw:
            unique_kw.append(w)
    keywords = f" Keywords: {', '.join(unique_kw[:5])}." if unique_kw else ""
    return f"{base} Confidence {(confidence * 100):.0f}%." + keywords


# --- START: LOGIN/LOGOUT ROUTES ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'on'
        
        db_user = get_user_by_username(username or "")
        if db_user and check_password_hash(db_user['password_hash'], password or ""):
            user_obj = User(db_user['id'], db_user['username'])
            login_user(user_obj, remember=remember)
            # Redirect to the page the user was trying to access (Flask-Login handles 'next' argument)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'error')
            
    # Renders the login form
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
# --- END: LOGIN/LOGOUT ROUTES ---


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = (request.form.get('password') or '').strip()
        confirm = (request.form.get('confirm_password') or '').strip()

        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('signup.html')

        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')

        existing = get_user_by_username(username)
        if existing:
            flash('Username already exists. Please choose another.', 'error')
            return render_template('signup.html')

        try:
            create_user(username, password)
            flash('Account created. Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception:
            flash('Could not create account. Please try again.', 'error')
            return render_template('signup.html')

    return render_template('signup.html')


@app.route('/')
@login_required # <-- SECURES THE HOME PAGE
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
@login_required # <-- SECURES THE API ENDPOINT
def analyze_lyrics():
# ... (rest of analyze_lyrics function is unchanged) ...
    """API endpoint for emotion analysis"""
    try:
        data = request.get_json()
        lyrics = data.get('lyrics', '').strip()
        source_url = (data.get('source_url') or '').strip()
        fetched_from_link = False
        
        if not lyrics and source_url:
            fetched = fetch_lyrics_from_url(source_url)
            if fetched:
                lyrics = fetched
                fetched_from_link = True
            else:
                return jsonify({
                    'error': 'Could not fetch lyrics from the link. Please paste lyrics manually.',
                    'status': 'error'
                }), 400
        
        if not lyrics:
            return jsonify({'error': 'Please provide lyrics', 'status': 'error'}), 400
        
        if len(lyrics) < 100:
            return jsonify({'error': 'Please provide at least 100 characters', 'status': 'error'}), 400
        
        # Detect emotion
        result = detect_emotion(lyrics)
        result['source_url'] = source_url
        result['fetched_from_link'] = fetched_from_link
        # Make the top-level description based on the primary emotion
        result['description'] = describe_emotion(
            result.get('primary_emotion', ''),
            result.get('confidence', 0),
            result.get('matched_words', [])
        )
        result['line_analysis'] = annotate_lyrics_lines(lyrics)

        # Detect mood separately
        mood_res = detect_mood(lyrics)
        if mood_res.get('status') == 'success':
            result['primary_mood'] = mood_res.get('primary_mood')
            result['mood_confidence'] = mood_res.get('confidence')
            result['all_moods'] = mood_res.get('all_moods')
            result['mood_description'] = mood_res.get('description')
            mood_matched = mood_res.get('matched_words') or []
        else:
            mood_matched = []

        # Build tone map for high-level descriptions (used for summaries)
        tone_map = {
            'love': ['romantic', 'affectionate'],
            'motivation': ['motivational', 'driven'],
            'joy': ['uplifting', 'positive'],
            'sadness': ['somber', 'reflective'],
            'anger': ['intense', 'forceful'],
            'fear': ['uneasy', 'anxious'],
            'trust': ['reassuring', 'steady'],
            'disgust': ['averse', 'rejecting'],
            'surprise': ['unexpected', 'dynamic'],
            'anticipation': ['forward-looking', 'excited']
        }

        emo = result.get('primary_emotion') or ''
        emots_sorted = list((result.get('all_emotions') or {}).items())
        second_emo = emots_sorted[1][0] if len(emots_sorted) > 1 else None
        primary_tones = tone_map.get(emo, ['distinct'])
        undertone_emo = second_emo or (list(result.get('all_emotions', {}).keys())[0] if result.get('all_emotions') else None)
        undertones = tone_map.get(undertone_emo or '', [])
        overall_tone = f"Predominantly {primary_tones[0]} and {primary_tones[1] if len(primary_tones)>1 else primary_tones[0]}, with undertones of {undertones[0] if undertones else 'subtle variation'}."
        result['overall_tone'] = overall_tone
        result['tone_summary'] = overall_tone

        # Build emotion-specific keywords (unique and relevant to the primary emotion)
        emo_words = []
        for w in result.get('matched_words') or []:
            if emo in (nrc_emotions.get(w) or []):
                if w not in emo_words:
                    emo_words.append(w)
        # Replace top-level matched_words with emotion-specific keywords to avoid showing unrelated words in the UI
        result['matched_words'] = emo_words

        # Build mood-specific keywords, remove any words already used by emotion keywords
        mood_words = []
        primary_mood = result.get('primary_mood') or ''
        for w in mood_matched:
            if primary_mood in (mood_lexicon.get(w) or []):
                if w not in mood_words:
                    mood_words.append(w)
        # Filter out any overlap with emotion keywords
        dedup = set(emo_words)
        mood_words = [w for w in mood_words if w not in dedup]

        # If mood label overlaps with emotion label or mapped equivalents, try to select a distinct mood
        moods_sorted = list((result.get('all_moods') or {}).items())
        mood_descriptions = {
            'happy': 'Upbeat and cheerful, a bright listening state.',
            'sad': 'Tender and reflective, leaning into melancholy.',
            'angry': 'Edgy and intense, cathartic energy.',
            'anxious': 'Restless and tense, heightened alertness.',
            'calm': 'Soft and peaceful, steady breathing.',
            'relaxed': 'Loose and easy, unwinding gently.',
            'energetic': 'Lively and driving, forward momentum.',
            'romantic': 'Warm and intimate, closeness and affection.'
        }

        current_mood = result.get('primary_mood')
        if current_mood and (current_mood == emo or current_mood in tone_map.get(emo, [])):
            # pick next best mood that doesn't overlap semantically
            for m, _ in moods_sorted:
                if m != current_mood and m not in tone_map.get(emo, []) and m != emo:
                    result['primary_mood'] = m
                    result['mood_confidence'] = float(result.get('all_moods', {}).get(m, 0))
                    result['mood_description'] = mood_descriptions.get(m, result.get('mood_description'))
                    # rebuild mood_words for the newly chosen mood
                    new_mood_words = [w for w in mood_matched if m in (mood_lexicon.get(w) or [])]
                    new_mood_words = [w for w in new_mood_words if w not in dedup]
                    mood_words = new_mood_words
                    break

        result['emotion_keywords'] = emo_words[:6]
        result['mood_keywords'] = mood_words[:6]
        # Provide convenient top-level keys used by the frontend
        result['mood'] = result.get('primary_mood')
        result['matched_words'] = result['matched_words'][:6]

        phrases = []
        annotations = result['line_analysis']
        for a in annotations:
            mw = [m['word'] for m in a.get('matched_words') or []]
            if any(x in mw for x in emo_words):
                txt = a.get('text') or ''
                if txt and txt not in phrases:
                    phrases.append(txt)
            if len(phrases) >= 5:
                break
        if len(phrases) < 5:
            for a in annotations:
                txt = a.get('text') or ''
                if txt and txt not in phrases:
                    phrases.append(txt)
                if len(phrases) >= 5:
                    break
        result['key_phrases'] = phrases[:5]

        breakdown = []
        for i, (em_name, em_score) in enumerate(emots_sorted[:3]):
            level = 'Strong' if i == 0 else ('Moderate' if i == 1 else 'Subtle')
            kwords = []
            for w in (result.get('matched_words') or []):
                if em_name in (nrc_emotions.get(w) or []):
                    if w not in kwords:
                        kwords.append(w)
            breakdown.append({
                'emotion': em_name,
                'score': float(em_score),
                'level': level,
                'keywords': kwords[:4]
            })
        result['emotion_breakdown'] = breakdown

        save_data = {
            'primary_emotion': result.get('primary_emotion'),
            'confidence': result.get('confidence'),
            'primary_mood': result.get('primary_mood'),
            'mood_confidence': result.get('mood_confidence'),
            'overall_tone': result.get('overall_tone'),
            'emotion_keywords': result.get('emotion_keywords'),
            'mood_keywords': result.get('mood_keywords'),
            'key_phrases': result.get('key_phrases')
        }
        if current_user and getattr(current_user, 'id', None):
            result['analysis_id'] = save_analysis(int(current_user.id), lyrics, save_data)

        # Build timeline by auto-segmenting lyrics
        timeline = []
        segments = segment_lyrics_auto(lyrics)
        for idx, segment in enumerate(segments):
            segment_result = detect_emotion(segment)
            if segment_result.get('status') != 'success':
                continue

            # Also detect mood for this segment to provide distinct mood vs emotion results
            seg_mood = detect_mood(segment)
            seg_mood_matched = seg_mood.get('matched_words') or []

            # build per-segment emotion keywords (relevant to the primary emotion)
            seg_emo = segment_result.get('primary_emotion')
            seg_emo_keywords = []
            for w in segment_result.get('matched_words') or []:
                if seg_emo in (nrc_emotions.get(w) or []):
                    if w not in seg_emo_keywords:
                        seg_emo_keywords.append(w)

            # build per-segment mood keywords and remove overlap with emotion keywords
            seg_mood_keywords = []
            seg_primary_mood = seg_mood.get('primary_mood')
            for w in seg_mood_matched:
                if seg_primary_mood in (mood_lexicon.get(w) or []):
                    if w not in seg_mood_keywords:
                        seg_mood_keywords.append(w)
            seg_mood_keywords = [w for w in seg_mood_keywords if w not in set(seg_emo_keywords)]

            timeline.append({
                'segment_index': idx,
                'text': segment,
                'primary_emotion': segment_result['primary_emotion'],
                'confidence': segment_result['confidence'],
                'emotion_keywords': seg_emo_keywords,
                'mood': seg_primary_mood,
                'mood_confidence': seg_mood.get('confidence'),
                'mood_keywords': seg_mood_keywords,
                'all_emotions': segment_result['all_emotions'],
                'description': describe_emotion(
                    segment_result['primary_emotion'],
                    segment_result['confidence'],
                    seg_emo_keywords
                ),
                'mood_description': seg_mood.get('description')
            })
        
        result['timeline'] = timeline
        result['segment_count'] = len(timeline)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/analyze_mood', methods=['POST'])
@login_required
def analyze_mood():
    try:
        data = request.get_json()
        lyrics = (data.get('lyrics') or '').strip()
        if not lyrics:
            return jsonify({'error': 'Please provide lyrics', 'status': 'error'}), 400
        if len(lyrics) < 100:
            return jsonify({'error': 'Please provide at least 100 characters', 'status': 'error'}), 400
        result = detect_mood(lyrics)
        return jsonify(result), 200 if result.get('status') == 'success' else 500
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
# ... (rest of health_check function is unchanged) ...
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200

@app.route('/api/analyses', methods=['GET'])
@login_required
def analyses_list():
    try:
        rows = list_analyses(int(current_user.id), limit=10)
        return jsonify({'items': rows, 'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/analyses/<int:analysis_id>', methods=['GET'])
@login_required
def analyses_get(analysis_id: int):
    try:
        row = get_analysis(int(current_user.id), analysis_id)
        if not row:
            return jsonify({'error': 'Not found', 'status': 'error'}), 404
        return jsonify({'item': row, 'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/analyses/<int:analysis_id>', methods=['DELETE'])
@login_required
def analyses_delete(analysis_id: int):
    try:
        ok = delete_analysis(int(current_user.id), analysis_id)
        if not ok:
            return jsonify({'error': 'Not found', 'status': 'error'}), 404
        return jsonify({'deleted': True, 'status': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
# ... (rest of error handlers are unchanged) ...
    """Handle file too large"""
    return jsonify({'error': 'File too large', 'status': 'error'}), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found', 'status': 'error'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


if __name__ == '__main__':
# ... (rest of __main__ block is unchanged) ...
    print("=" * 60)
    print("Emotion Detection in Music Lyrics - Flask Application")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Open browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
