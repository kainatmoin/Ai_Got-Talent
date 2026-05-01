#  Emotion Classifier — AI Got Talent

This project was built for **Developers Day '26** hosted by FAST NUCES Karachi. 
The competition was called **AI Got Talent** and the challenge was simple — 
build a machine learning model that can detect emotions in tweets. 
We had 2 hours. No pretrained models. Just raw Python and a lot of coffee ☕

---

## What does it do?

You give it a tweet. It tells you what emotions are in it.

Not just one emotion — a tweet can have multiple emotions at the same time.
For example:

> *"I'm so proud of what we achieved, this feels amazing!"*
> → `pride`, `joy`, `admiration`

> *"This situation is frustrating and honestly makes me really upset."*
> → `anger`, `disgust`, `sadness`

The model detects 9 emotions:
`admiration` `anger` `disgust` `fear` `hope` `joy` `love` `pride` `sadness`

---

## How we built it

We kept things simple but smart.

**Text Cleaning**
First we clean the tweet — lowercase everything, remove links, 
mentions, hashtags, and fix weird characters. 
We also expand contractions like "I'm" → "I am" so the model 
understands better.

**Feature Extraction**
We used TF-IDF vectorization in three ways:
- Word level (unigrams, bigrams, trigrams)
- Character level (catches spelling variations)
- Bigram focused (phrase patterns)

All three are combined into one big feature matrix.

**The Models**
We didn't rely on just one model. We trained four:
- Logistic Regression (C=5)
- Logistic Regression (C=15) 
- LinearSVC
- SGD Classifier

Then we combined their predictions using a weighted ensemble — 
basically letting all four vote together for better accuracy.

**Threshold Tuning**
Instead of using a fixed 0.5 cutoff for every emotion, 
we tuned a separate threshold for each emotion. 
Some emotions like `pride` needed a lower threshold (0.25) 
while others like `fear` needed higher (0.50). 
This alone boosted our F1 score significantly.

---

## Results

| Metric | Score |
|--------|-------|
| F1 Micro | 0.893 |
| F1 Macro | 0.893 |
| Precision | 0.886 |
| Recall | 0.901 |
| Hamming Loss | 0.049 |
| Exact Match | 66.7% |

Per emotion breakdown:

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|----|
| admiration | 0.943 | 0.983 | 0.962 |
| anger | 0.879 | 0.860 | 0.870 |
| disgust | 0.962 | 0.927 | 0.944 |
| fear | 0.944 | 0.919 | 0.932 |
| hope | 0.962 | 0.856 | 0.906 |
| joy | 0.759 | 0.837 | 0.796 |
| love | 0.871 | 0.908 | 0.889 |
| pride | 0.941 | 0.969 | 0.955 |
| sadness | 0.738 | 0.845 | 0.788 |

---

## How to run it yourself

**1. Clone the repo**
```bash
