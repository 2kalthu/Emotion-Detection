from app import lexicon_based_emotion_scoring, annotate_lyrics_lines


def test_lexicon_detects_love():
    text = "I love you so much, my heart sings and I adore every moment with you."
    scores, matched = lexicon_based_emotion_scoring(text)
    # Lexicon should match 'love' tokens and produce a non-zero love score
    assert 'love' in matched or scores.get('love', 0) > 0.0
    assert scores.get('love', 0) > 0.0


def test_annotate_lines_includes_love_meaning():
    lyrics = "My darling, you are everything I need\nI love you more each day"
    annotations = annotate_lyrics_lines(lyrics)

    found = False
    for a in annotations:
        for w in a['matched_words']:
            if w['word'] in ('love', 'darling'):
                found = True
                # Ensure the meaning includes an affectionate hint
                assert 'Affection' in w['meaning'] or 'Affection' in w['meaning'].capitalize() or 'love' in w['emotions']
    assert found
