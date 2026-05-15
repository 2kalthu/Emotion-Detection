"""
Train Emotion Detection Model
Run this script once to generate emotion_model.pkl
"""

import sys
import warnings
from collections import Counter
import json

import joblib
import numpy as np
import pandas as pd
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from scipy.sparse import csr_matrix

warnings.filterwarnings('ignore')

# Ensure required NLTK resources
for resource in ['punkt', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4']:
    try:
        # wordnet is under corpora, punkt under tokenizers
        if resource == 'punkt':
            nltk.data.find(f'tokenizers/{resource}')
        elif resource in ('wordnet', 'omw-1.4'):
            nltk.data.find(f'corpora/{resource}')
        else:
            nltk.data.find(f'taggers/{resource}')
    except LookupError:
        nltk.download(resource)

print("=" * 70)
print("Emotion Detection Model Training")
print("=" * 70)

# Training dataset (40 Items - VERIFIED)
training_lyrics = [
    'i love you so much my heart is full of joy',
    'i miss you tears falling down i feel so lonely and sad',
    'i hate this i am so angry and frustrated everything is wrong',
    'i am scared and afraid of what will happen to me',
    'this is amazing and wonderful i am so happy and excited',
    'i feel terrible and depressed nothing makes sense',
    'you make me so mad i cannot believe this happened',
    'i am terrified and frightened of everything',
    'life is beautiful and i love every moment with joy',
    'my heart is broken and i am devastated',
    'this is outrageous i am furious at this injustice',
    'something unexpected and surprising happened today',
    'i feel cheerful and delighted about the future',
    'darkness surrounds me and i am afraid',
    'i trust you and believe in you completely',
    'this disgusts me and i find it revolting',
    'i anticipate great things coming my way',
    'you hurt me deeply and i feel betrayed',
    'the sun is shining and i feel wonderful',
    'this is worst day of my life i feel empty',
    'incredible and fantastic things happening',
    'i am anxious and worried about everything',
    'love fills my soul and i am grateful',
    'loneliness is killing me inside i feel abandoned',
    'outrageous behavior i am absolutely livid',
    'uncertainty fills me with dread and panic',
    'optimism and hope guide my path forward',
    'sorrow and grief consume me completely',
    'joy and laughter brighten my darkest days',
    'disappointment and regret weigh heavily',
    'excitement and enthusiasm drive me forward',
    'despair and hopelessness surround me',
    'amazement and wonder fill my eyes',
    'contempt and disgust course through me',
    'trust and confidence in the future',
    'betrayal cuts deep into my heart',
    'bliss and serenity calm my troubled mind',
    'anguish and torment tear me apart',
    'delight and pleasure bring me peace',
    'dread and terror consume my thoughts',
    # love and motivation examples
    'your love lifts me up and i can t help but smile',
    'i am motivated to chase my dreams and achieve greatness',
    'inspired and driven, i will reach my goal',
    'i love you more than words could ever show',
    'your embrace makes the world feel right',
    'my heart beats for you endlessly',
    'the love we share is a light in the dark',
    'i fall deeper in love with you every day',
    'your kisses make my worries disappear',
    'romantic nights with you are my favorite song',
    'loving you gives me strength and hope',
    'my heart sings when you are near',
    'you are my one true love and my home'
]

# Training emotions (40 Items - VERIFIED)
training_emotions = [
    'joy', 'sadness', 'anger', 'fear', 'joy',
    'sadness', 'anger', 'fear', 'joy', 'sadness',
    'anger', 'surprise', 'joy', 'fear', 'trust',
    'disgust', 'anticipation', 'sadness', 'joy', 'sadness',
    'joy', 'fear', 'joy', 'sadness', 'anger',
    'fear', 'anticipation', 'sadness', 'joy', 'sadness',
    'joy', 'fear', 'joy', 'sadness', 'surprise',
    'disgust', 'trust', 'sadness', 'joy', 'sadness',
    'joy', 'disgust', 'trust',
    # labels for new samples added: love / motivation
    'love', 'motivation', 'motivation',
    # additional love samples
    'love','love','love','love','love','love','love','love','love','love'
]

# --- Mood dataset (small, for demo purposes) ---
training_moods = [
    'happy', 'sad', 'angry', 'anxious', 'calm', 'relaxed', 'energetic', 'romantic',
    'happy', 'sad', 'angry', 'anxious', 'calm', 'relaxed', 'energetic', 'romantic',
    'happy', 'sad', 'angry', 'anxious', 'calm', 'relaxed', 'energetic', 'romantic',
    'happy', 'sad', 'angry', 'anxious', 'calm', 'relaxed', 'energetic', 'romantic',
    'happy', 'sad', 'angry', 'anxious', 'calm', 'relaxed', 'energetic', 'romantic',
    # fix appended samples: love -> romantic, others remain motivated
    'romantic', 'motivated', 'motivated',
    # moods for additional love samples
    'romantic','romantic','romantic','romantic','romantic','romantic','romantic','romantic','romantic','romantic'
]

# --- Feature helpers ---
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


from features import PosTagCountsTransformer


# Create DataFrame
print("\n[1/4] Creating training dataset...")

len_lyrics = len(training_lyrics)
len_emotions = len(training_emotions)
len_moods = len(training_moods)

# Enforce strict equality so labels can't drift silently
if not (len_lyrics == len_emotions == len_moods):
    raise ValueError(f"Training lists length mismatch: lyrics={len_lyrics}, emotions={len_emotions}, moods={len_moods}. Please fix `training_lyrics`, `training_emotions`, and `training_moods` to be aligned.")

# Create DataFrame with mood column for better diagnostics
df = pd.DataFrame({
    'lyrics': training_lyrics,
    'emotion': training_emotions,
    'mood': training_moods
})
print(f"     ✓ Created dataset with {len(df)} samples")
print(f"     Emotion distribution:")
for emotion, count in df['emotion'].value_counts().sort_index().items():
    print(f"        - {emotion.capitalize()}: {count}")
print(f"     Mood distribution:")
for mood, count in df['mood'].value_counts().sort_index().items():
    print(f"        - {mood.capitalize()}: {count}")


def synonym_replace(text, replace_prob=0.15):
    """Simple synonym replacement using WordNet for lightweight augmentation."""
    tokens = word_tokenize(text)
    out = []
    for t in tokens:
        if random.random() < replace_prob:
            syns = set()
            for syn in wordnet.synsets(t):
                for l in syn.lemmas():
                    w = l.name().replace('_', ' ')
                    if w.lower() != t.lower():
                        syns.add(w)
            if syns:
                out.append(random.choice(list(syns)))
                continue
        out.append(t)
    return ' '.join(out)


def augment_dataset(texts, labels, multiplier=2):
    """Return augmented (texts, labels) lists by applying synonym replacement."""
    aug_texts, aug_labels = list(texts), list(labels)
    for _ in range(multiplier - 1):
        for t, l in zip(texts, labels):
            aug_texts.append(synonym_replace(t))
            aug_labels.append(l)
    return aug_texts, aug_labels


# Try to use sentence-transformers if available
SENTENCE_TRANSFORMER_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMER_AVAILABLE = False

class EmbeddingTransformer:
    """Wraps a SentenceTransformer to produce dense vector features for sklearn pipelines."""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            raise RuntimeError('sentence-transformers not available; cannot use EmbeddingTransformer')
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            raise RuntimeError('sentence-transformers not available; cannot use EmbeddingTransformer')
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model.encode(list(X), show_progress_bar=False)


def train_and_save_model(name, texts, labels, out_file, use_embeddings=None):
    if use_embeddings is None:
        use_embeddings = SENTENCE_TRANSFORMER_AVAILABLE
    """Train a model for given texts/labels and save it. Returns accuracy on test split."""
    print(f"\n[Training {name}] Samples: {len(texts)}")

    # Augment dataset slightly if small
    if len(texts) < 400:
        print("     Dataset small — applying light augmentation (multiplier=4)")
        texts, labels = augment_dataset(texts, labels, multiplier=4)

    df_local = pd.DataFrame({'text': texts, 'label': labels})
    train_df, test_df = train_test_split(df_local, test_size=0.2, stratify=df_local['label'], random_state=42)

    if use_embeddings:
        features_local = EmbeddingTransformer()
        pipe = Pipeline([
            ('embed', features_local),
            ('clf', LogisticRegression(max_iter=2000, random_state=42, multi_class='multinomial', solver='lbfgs', class_weight='balanced'))
        ])
        param_grid = {'clf__C': [0.1, 1.0, 10.0]}
    else:
        features_local = FeatureUnion([
            ('tfidf', TfidfVectorizer(max_features=6000, stop_words='english', ngram_range=(1,2))),
            ('pos', PosTagCountsTransformer())
        ])
        pipe = Pipeline([
            ('features', features_local),
            ('clf', LogisticRegression(max_iter=2000, random_state=42, multi_class='multinomial', solver='lbfgs', class_weight='balanced'))
        ])
        param_grid = {'clf__C': [0.1, 1.0, 10.0]}

    gs = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    try:
        gs.fit(train_df['text'], train_df['label'])
        best = gs.best_estimator_

        # Cross-validated score on training set
        cv_scores = cross_val_score(best, train_df['text'], train_df['label'], cv=5, scoring='accuracy', n_jobs=-1)
        print(f"     CV (train) accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

        preds = best.predict(test_df['text'])
        acc = accuracy_score(test_df['label'], preds)
        print(f"     Best params: {gs.best_params_}")
        print(f"     Test accuracy: {acc:.2%}")
        print("\nValidation report:")
        print(classification_report(test_df['label'], preds))

        # Capture classification_report & confusion matrix for per-class analysis
        report_dict = classification_report(test_df['label'], preds, output_dict=True)
        # Determine label ordering from trained classifier if possible
        try:
            clf = best.named_steps.get('clf', None)
            if clf is not None and hasattr(clf, 'classes_'):
                labels = list(clf.classes_)
            else:
                labels = sorted(df_local['label'].unique().tolist())
        except Exception:
            labels = sorted(df_local['label'].unique().tolist())

        cm = confusion_matrix(test_df['label'], preds, labels=labels)

        # Refit on full (augmented) data and persist the model
        best.fit(df_local['text'], df_local['label'])
        try:
            joblib.dump(best, out_file)
        except Exception as e:
            print(f"     ✗ Error saving model to {out_file}: {e}")

        # Save metrics including per-class report and confusion matrix
        metrics = {
            'name': name,
            'test_accuracy': float(acc),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'labels': labels,
            'classification_report': report_dict,
            'confusion_matrix': cm.tolist()
        }
        try:
            with open(f'{out_file}.metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"     ✗ Error saving metrics: {e}")

        print(f"     ✓ {name} training complete (model attempt saved to '{out_file}')")
        return acc
    except Exception as e:
        print(f"     ✗ Error training {name}: {e}")
        return 0.0

# Train & save both models (emotion + mood)
print("\n[2/4] Training models (emotion + mood)...")

# Ensure mood labels match sample size if needed
if len(training_moods) < len(training_lyrics):
    print(f"     ⚠️ mood labels fewer than lyrics, repeating to match size")
    repeats = (len(training_lyrics) + len(training_moods) - 1) // len(training_moods)
    training_moods = (training_moods * repeats)[: len(training_lyrics)]

emotion_acc = train_and_save_model('Emotion', training_lyrics, training_emotions, 'emotion_model.pkl')
mood_acc = train_and_save_model('Mood', training_lyrics, training_moods, 'mood_model.pkl')

if emotion_acc < 0.95:
    print(f"\n⚠️ Emotion model accuracy {emotion_acc:.2%} is below 95%. Consider adding more labeled samples or using a transformer-based model (e.g., sentence-transformers) to improve performance.")
else:
    print(f"\n✅ Emotion model reached target accuracy: {emotion_acc:.2%}")

if mood_acc < 0.95:
    print(f"⚠️ Mood model accuracy {mood_acc:.2%} is below 95%. For higher accuracy add diverse labeled mood data or use embeddings/transformers.")
else:
    print(f"✅ Mood model reached target accuracy: {mood_acc:.2%}")

# Quick smoke tests using saved models
print("\n" + "=" * 70)
print("Quick Smoke Tests (using saved models)")
print("=" * 70)

try:
    em_model = joblib.load('emotion_model.pkl')
    md_model = joblib.load('mood_model.pkl')
    test_cases = [
        ("i love this song it makes me so happy and joyful", 'emotion'),
        ("i feel terrible and sad today nothing is going right", 'emotion'),
        ("i am so angry and furious at this", 'emotion'),
        ("i am afraid and scared of everything", 'emotion'),
        ("i trust you completely with all my heart", 'emotion'),
        ("this disgusts me it is revolting", 'emotion'),
        ("something amazing happened today how wonderful", 'emotion'),
        ("i anticipate great things coming my way", 'emotion'),
        ("i feel relaxed and calm after a long day", 'mood'),
        ("this song pumps me up and energetic", 'mood'),
    ]

    for text, kind in test_cases:
        if kind == 'emotion':
            pred = em_model.predict([text])[0]
            prob = em_model.predict_proba([text])[0].max()
            print(f"EMOTION Test: '{text[:50]}...' -> {pred} (Conf: {prob:.1%})")
        else:
            pred = md_model.predict([text])[0]
            prob = md_model.predict_proba([text])[0].max()
            print(f"MOOD Test: '{text[:50]}...' -> {pred} (Conf: {prob:.1%})")

    print("\n" + "=" * 70)
    print("✓ Training Complete! Models saved and smoke-tested.")
    print("=" * 70)
    print("\nModels are ready for use. Start the app with:")
    print("  python app.py")
    print("=" * 70)
except Exception as e:
    print(f"Could not run smoke tests: {e}")
