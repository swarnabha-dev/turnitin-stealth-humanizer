#!/usr/bin/env python3
"""
detect.py - Next-generation Turnitin-grade AI detection engine.

Multi-Tier Architecture:
  1. Turnitin Sliding-Window Neural Engine (Cloud 355M RoBERTa Large, 0% CPU):
     - Evaluates sentences within local 3-sentence sliding context windows.
     - Computes Turnitin-exact AI %: (word_count in AI sentences / total_words) * 100.
     - Highlights flagged sentences in terminal with Turnitin-style cyan/blue markers.
     - Turnitin cluster gating: suppresses isolated short false-positives unless correlated.
  2. Native StealthHumanizer 12-Metric Engine (Pure Python):
     - Perplexity, Burstiness, Vocab Diversity, Sentence Variation, Transitions,
       Passive Voice, AI Phrases, Sentence Starts, Pronouns, Hedging, Quantifiers.
  3. 6-Point Linguistic & Heuristic Baseline:
     - Burstiness 2.0 (CV + consecutive delta, decimal & abbreviation safe)
     - Lexical Richness / Perplexity proxy (TTR + Hapax Legomena)
     - Specificity (numbers, metrics, currencies, proper entities per 100 words)
     - Slop Density (2026 AI buzzwords and clichés from slop.json)
     - Fingerprint (zero-width chars, format marks, typography tells)
     - Voice (natural contractions, personal pronouns, structural tropes)
  4. Local Fallback: transformers pipeline (if local torch/transformers present)

Usage:
  python detect.py draft.txt
  python detect.py draft.txt --turnitin          # Run detailed Turnitin sliding window & highlight
  python detect.py draft.txt --no-neural        # Pure offline heuristics
  python detect.py draft.txt --json
  python detect.py before.txt after.txt         # Compare two drafts
"""

import argparse
import json
import math
import os
import re
import statistics
import sys
import unicodedata
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
LEX = os.path.join(HERE, "slop.json")
ROUTER_ENDPOINT = "https://router.huggingface.co/hf-inference/models/"

DEFAULT_LARGE_MODEL = "openai-community/roberta-large-openai-detector"
FALLBACK_MODELS = [
    "openai-community/roberta-large-openai-detector",
    "openai-community/roberta-base-openai-detector",
    "Hello-SimpleAI/chatgpt-detector-roberta"
]

ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc", "eg", "ie",
    "sep", "oct", "nov", "dec", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "vol", "dept", "est", "approx", "al", "fig", "no", "phd", "msc", "bsc", "btech", "mtech"
}

WORD_RE = re.compile(r"[A-Za-z']+")
CONTRACTIONS = re.compile(r"\b\w+'(?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
PRONOUNS = re.compile(r"\b(i|me|my|mine|we|us|our|you|your)\b", re.IGNORECASE)
NUMBERS = re.compile(r"\b\d[\d,.]*%?\b|\$\d|€\d|£\d|₹\d")
PROPER = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", re.MULTILINE)

# ANSI Colors for Turnitin Highlighting
CYAN_BG = "\033[46m\033[30m"
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"

# ==================== STEALTHHUMANIZER NATIVE PATTERNS ====================
STEALTH_AI_PHRASES = [
    'it is important to note', 'it is worth mentioning', 'it is worth noting',
    'in conclusion', 'in summary', 'to summarize', 'to conclude',
    'furthermore', 'moreover', 'additionally', 'in addition',
    'it is crucial', 'it is essential', 'it is imperative',
    'plays a crucial role', 'plays an important role', 'plays a pivotal role',
    'has the potential to', 'it is evident that', 'it is clear that',
    'demonstrates the', 'illustrates the', 'showcases the',
    'underscores the', 'highlights the', 'emphasizes the',
    'on the other hand', 'in terms of', 'when it comes to',
    'as previously mentioned', 'as discussed earlier', 'as noted above',
    'it should be noted', 'it must be noted', 'needless to say',
    'last but not least', 'first and foremost', 'at the end of the day',
    'in today\'s world', 'in this day and age', 'in the modern era',
    'in the contemporary landscape', 'in the current landscape',
    'a myriad of', 'delve into', 'delves into',
    'tapestry of', 'navigating the landscape',
    'multifaceted', 'robust', 'seamless', 'streamline',
    'synergy', 'paradigm', 'paradigm shift', 'holistic',
    'innovative', 'cutting-edge', 'state-of-the-art', 'groundbreaking',
    'transformative', 'comprehensive', 'unprecedented',
    'utilize', 'facilitate', 'optimize', 'leverage',
    'implement', 'foster', 'cultivate', 'empower',
    'embark on a journey', 'sheds light on', 'brings to the forefront'
]

STEALTH_AI_STARTERS = [
    'in this article', 'this paper', 'this study', 'this research',
    'the results', 'the findings', 'the analysis', 'the data',
    'it is widely', 'it is commonly', 'there is a',
    'one of the', 'another important', 'a key aspect',
    'the importance of', 'the significance of', 'the role of',
    'research has shown', 'studies have shown', 'evidence suggests'
]

STEALTH_HEDGING_PHRASES = [
    'it could be argued', 'one might consider', 'it is possible that',
    'it would seem', 'this suggests that', 'this may indicate',
    'it appears that', 'this could potentially', 'one could argue'
]

STEALTH_QUANTIFIERS = [
    'numerous', 'various', 'multiple', 'several', 'a variety of',
    'a multitude of', 'a range of', 'a number of', 'countless',
    'a vast array of', 'a wide range of', 'a significant number of'
]

STEALTH_TRANSITION_WORDS = [
    'however', 'therefore', 'moreover', 'furthermore', 'additionally',
    'consequently', 'nevertheless', 'meanwhile', 'subsequently', 'thus',
    'hence', 'accordingly', 'similarly', 'likewise', 'conversely',
    'otherwise', 'instead', 'rather', 'yet', 'still'
]

STEALTH_HUMAN_INDICATORS = [
    'basically', 'actually', 'literally', 'honestly', 'like',
    'you know', 'i mean', 'kind of', 'sort of', 'pretty much',
    'i think', 'i feel like', 'i guess', "i'd say", 'to be honest',
    'weirdly', 'interestingly', 'funnily enough', 'surprisingly',
    'anyway', 'so yeah', 'i dunno', 'tbh', 'imo'
]


def load_env():
    """Load environment variables from .env files if present."""
    search_paths = [
        os.path.join(HERE, ".env"),
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(HERE), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(HERE)), ".env"),
    ]
    for p in search_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass


load_env()


def clamp(n):
    return max(0.0, min(100.0, float(n)))


def scale(value, human, machine):
    if human == machine:
        return 50.0
    return clamp((value - machine) / (human - machine) * 100.0)


def split_sentences(text):
    """Split text into sentences while protecting decimals, abbreviations, and 1-word lines."""
    if not text or not text.strip():
        return []

    urls = []
    def stash_url(m):
        urls.append(m.group(0))
        return f"__URL_{len(urls)-1}__"
    processed = re.sub(r"https?://\S+|www\.\S+", stash_url, text)
    processed = re.sub(r"(\d+)\.(\d+)", r"\1__DECIMAL__\2", processed)

    def protect_abbr(m):
        word = m.group(1).lower().rstrip(".")
        if word in ABBREVIATIONS:
            return m.group(0).replace(".", "__DOT__")
        return m.group(0)

    processed = re.sub(r"\b([A-Za-z]{1,6})\.(?=\s+[A-Za-z0-9])", protect_abbr, processed)
    processed = re.sub(r"\b([A-Z])\.(?=\s+[A-Z])", r"\1__DOT__", processed)

    raw_chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(\[])|(?<=[.!?])\n+|\n\s*\n", processed)

    sentences = []
    for chunk in raw_chunks:
        chunk = chunk.replace("__DECIMAL__", ".").replace("__DOT__", ".")
        for i, u in enumerate(urls):
            chunk = chunk.replace(f"__URL_{i}__", u)
        clean_s = re.sub(r"^\s*[-*•\d\.\)]\s*", "", chunk).strip()
        if clean_s and any(c.isalnum() for c in clean_s):
            sentences.append(clean_s)

    return sentences


def words(text):
    return WORD_RE.findall(text)


# ==================== CLOUD NEURAL (ROBERTA LARGE) ====================

def query_hf_router(text, model, token):
    """Query Hugging Face Router endpoint."""
    api_url = f"{ROUTER_ENDPOINT}{model}"
    # Truncate to ~150 words or 900 chars to stay safely within RoBERTa's 512 token embedding limit
    safe_text = " ".join(text.split()[:160]) if len(text.split()) > 160 else text[:900]
    payload = json.dumps({"inputs": safe_text}).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "Antigravity-AIDetector/2.0"
        }
    )
    with urllib.request.urlopen(req, timeout=12) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if isinstance(data, list) and len(data) > 0:
        entries = data[0] if isinstance(data[0], list) else data
        label_map = {item["label"].upper(): item["score"] for item in entries if "label" in item and "score" in item}

        real_score = label_map.get("LABEL_1") or label_map.get("REAL") or label_map.get("HUMAN")
        fake_score = label_map.get("LABEL_0") or label_map.get("FAKE") or label_map.get("CHATGPT")

        if real_score is not None:
            human_val = real_score * 100.0
        elif fake_score is not None:
            human_val = (1.0 - fake_score) * 100.0
        else:
            human_val = 50.0

        return clamp(human_val), model
    raise ValueError("Unexpected HF response format")


def check_neural(text, token=None, model=DEFAULT_LARGE_MODEL, allow_local_fallback=True):
    """Query Hugging Face Router with automatic model cascading and local fallback."""
    if not text or len(text.strip()) < 20:
        return None, "text too short for neural classification", None

    hf_token = token or os.environ.get("HF_TOKEN")

    if hf_token:
        models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
        last_error = ""
        for m in models_to_try:
            try:
                score, used_model = query_hf_router(text, m, hf_token)
                model_name = used_model.split("/")[-1]
                detail = f"HF Cloud Large ({model_name}): {score:.1f}% human confidence (0% CPU)"
                return score, detail, "hf_cloud"
            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
            except Exception as e:
                last_error = str(e)

        err_msg = f"HF Cloud exhausted ({last_error})"
    else:
        err_msg = "No HF_TOKEN in env/.env"

    if allow_local_fallback:
        try:
            from transformers import pipeline
            classifier = pipeline("text-classification", model="openai-community/roberta-base-openai-detector",
                                  truncation=True, max_length=512)
            res = classifier(text[:1500])[0]
            label = res["label"].upper()
            conf = res["score"]
            if "REAL" in label or "HUMAN" in label or "1" in label:
                human_val = conf * 100.0
            else:
                human_val = (1.0 - conf) * 100.0
            detail = f"Local CPU Transformer (roberta-base): {label} ({conf:.1%})"
            return clamp(human_val), detail, "local_pipeline"
        except ImportError:
            pass
        except Exception as e:
            err_msg += f" | Local CPU fallback error: {str(e)}"

    return None, err_msg, None


# ==================== TURNITIN SLIDING-WINDOW ENGINE ====================

def run_turnitin_sliding_window(text, token=None, model=DEFAULT_LARGE_MODEL):
    """
    Evaluates text using Turnitin's sliding-window architecture:
    1. Splits document into sentences.
    2. Builds local 3-sentence sliding context window for each sentence.
    3. Evaluates AI probability via RoBERTa Large 355M.
    4. Gating: Flags sentences only when AI confidence >= 75% in context.
    5. Computes Turnitin AI Index: (flagged_words / total_words) * 100.
    """
    sentences = split_sentences(text)
    total_doc_words = len(words(text))
    if not sentences or total_doc_words < 10:
        return {
            "turnitin_ai_index": 0.0,
            "turnitin_human_index": 100.0,
            "total_words": total_doc_words,
            "flagged_words": 0,
            "flagged_sentences": [],
            "sentences_analysis": [],
            "advisory": "Text too short for Turnitin analysis"
        }

    hf_token = token or os.environ.get("HF_TOKEN")
    sentence_analyses = [None] * len(sentences)
    flagged_indices = set()

    # Pre-calculate local window text, words, and base stealth scores
    windows = []
    base_scores = []
    for i, sent in enumerate(sentences):
        sent_words = len(sent.split())
        start_idx = max(0, i - 1)
        end_idx = min(len(sentences), i + 2)
        window_text = " ".join(sentences[start_idx:end_idx])
        s_res = stealth_analyze_sentence(sent)
        base_ai_prob = (100.0 - s_res["score"]) / 100.0
        windows.append((i, sent, sent_words, window_text, base_ai_prob))
        base_scores.append(base_ai_prob)

    # Determine which windows to query via Cloud RoBERTa Large
    # To prevent rate-limiting/timeouts on large documents, cap neural calls to top 15 candidates
    query_indices = set()
    if hf_token:
        if len(sentences) <= 15:
            query_indices = set(range(len(sentences)))
        else:
            candidates = sorted(
                [i for i, prob in enumerate(base_scores) if prob >= 0.30],
                key=lambda idx: base_scores[idx],
                reverse=True
            )
            query_indices = set(candidates[:15])

    def evaluate_window(item):
        i, sent, sent_words, window_text, base_ai_prob = item
        ai_prob = base_ai_prob

        if i in query_indices and hf_token:
            try:
                score, _ = query_hf_router(window_text, model, hf_token)
                ai_prob = (100.0 - score) / 100.0
            except Exception:
                ai_prob = base_ai_prob

        return i, {
            "index": i,
            "text": sent,
            "word_count": sent_words,
            "ai_probability": round(ai_prob * 100.0, 1),
            "flagged": False
        }

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as executor:
        for idx, res in executor.map(evaluate_window, windows):
            sentence_analyses[idx] = res

    # Turnitin Cluster Gating:
    # Turnitin flags sentence if AI prob >= 78% OR if consecutive sentences >= 65%
    for i, sa in enumerate(sentence_analyses):
        if sa["ai_probability"] >= 78.0 and sa["word_count"] >= 6:
            flagged_indices.add(i)
        elif sa["ai_probability"] >= 65.0:
            # Check if neighbor is also elevated
            has_prev = (i > 0 and sentence_analyses[i-1]["ai_probability"] >= 65.0)
            has_next = (i < len(sentence_analyses) - 1 and sentence_analyses[i+1]["ai_probability"] >= 65.0)
            if has_prev or has_next:
                flagged_indices.add(i)

    flagged_words = 0
    flagged_sentences_list = []
    for i, sa in enumerate(sentence_analyses):
        if i in flagged_indices:
            sa["flagged"] = True
            flagged_words += sa["word_count"]
            flagged_sentences_list.append(sa)

    turnitin_ai_index = (flagged_words / total_doc_words) * 100.0 if total_doc_words > 0 else 0.0
    turnitin_human_index = 100.0 - turnitin_ai_index

    advisory = ""
    if total_doc_words < 300:
        advisory = f"Document is {total_doc_words} words (Turnitin requires >= 300 words for official submissions)."

    return {
        "turnitin_ai_index": round(turnitin_ai_index, 1),
        "turnitin_human_index": round(turnitin_human_index, 1),
        "total_words": total_doc_words,
        "flagged_words": flagged_words,
        "flagged_sentences": flagged_sentences_list,
        "sentences_analysis": sentence_analyses,
        "advisory": advisory
    }


# ==================== NATIVE STEALTHHUMANIZER 12-METRIC ENGINE ====================

def stealth_calculate_perplexity(text):
    w = text.lower().split()
    if len(w) < 5:
        return 50.0
    freq = {}
    for word in w:
        freq[word] = freq.get(word, 0) + 1
    values = list(freq.values())
    max_freq = max(values)
    avg_freq = len(w) / len(values)
    uniformity = max_freq / avg_freq if avg_freq else 1.0

    bigrams = [f"{w[i]} {w[i+1]}" for i in range(len(w) - 1)]
    bigram_freq = {}
    for b in bigrams:
        bigram_freq[b] = bigram_freq.get(b, 0) + 1
    unique_bigrams = len(bigram_freq)
    bigram_diversity = unique_bigrams / len(bigrams) if bigrams else 0.5
    score = (bigram_diversity * 60.0) + ((100.0 - uniformity * 15.0) * 0.4)
    return clamp(score)


def stealth_calculate_burstiness(sentences):
    if len(sentences) < 3:
        return 50.0
    lengths = [len(s.split()) for s in sentences]
    avg = sum(lengths) / len(lengths) if lengths else 1.0
    variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    burstiness = (std_dev / avg) * 100.0 if avg else 0.0
    return clamp(burstiness * 2.5)


def stealth_calculate_vocab_diversity(text):
    cleaned = re.sub(r'[^\w\s]', '', text.lower()).split()
    w = [word for word in cleaned if len(word) > 2]
    if len(w) < 10:
        return 50.0
    return clamp((len(set(w)) / len(w)) * 100.0)


def stealth_calculate_sentence_length_variation(sentences):
    if len(sentences) < 3:
        return 50.0
    lengths = [len(s.split()) for s in sentences]
    max_len = max(lengths)
    min_len = min(lengths)
    avg = sum(lengths) / len(lengths) if lengths else 1.0
    return clamp(((max_len - min_len) / avg) * 60.0)


def stealth_calculate_transition_frequency(text):
    w = text.lower().split()
    if len(w) < 10:
        return 50.0
    lower = text.lower()
    count = 0
    for tw in STEALTH_TRANSITION_WORDS:
        count += len(re.findall(r'\b' + re.escape(tw) + r'\b', lower))
    return clamp((count / len(w)) * 1000.0)


def stealth_calculate_passive_voice_ratio(sentences):
    if len(sentences) < 2:
        return 50.0
    patterns = [
        re.compile(r'\b(is|are|was|were|been|being)\s+\w+ed\b', re.IGNORECASE),
        re.compile(r'\b(is|are|was|were|been|being)\s+\w+en\b', re.IGNORECASE),
    ]
    passive_count = 0
    for s in sentences:
        for p in patterns:
            m = p.findall(s)
            if m:
                passive_count += len(m)
    return clamp((passive_count / len(sentences)) * 100.0)


def stealth_calculate_ai_phrase_density(text, sentences):
    lower = text.lower()
    count = 0
    for phrase in STEALTH_AI_PHRASES:
        if phrase in lower:
            count += 1
    return clamp((count / max(len(sentences), 1)) * 20.0)


def stealth_calculate_sentence_start_diversity(sentences):
    if len(sentences) < 4:
        return 50.0
    starts = []
    for s in sentences:
        sp = s.split()
        if sp:
            cleaned = re.sub(r'[^a-z]', '', sp[0].lower())
            if cleaned:
                starts.append(cleaned)
    if not starts:
        return 50.0
    return clamp((len(set(starts)) / len(starts)) * 100.0)


def stealth_calculate_pronoun_usage(text):
    personal_pronouns = {'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your'}
    words_list = text.lower().split()
    if not words_list:
        return 50.0
    count = sum(1 for w in words_list if w in personal_pronouns)
    ratio = (count / len(words_list)) * 500.0
    return clamp(ratio)


def stealth_calculate_hedging_frequency(text):
    lower = text.lower()
    count = sum(1 for phrase in STEALTH_HEDGING_PHRASES if phrase in lower)
    return clamp(count * 15.0)


def stealth_calculate_quantifier_overuse(text):
    lower = text.lower()
    count = 0
    for q in STEALTH_QUANTIFIERS:
        count += len(re.findall(r'\b' + re.escape(q) + r'\b', lower))
    return clamp(count * 10.0)


def stealth_analyze_sentence(sentence):
    issues = []
    score = 50.0
    lower = sentence.lower()

    for phrase in STEALTH_AI_PHRASES:
        if phrase in lower:
            issues.append(f'AI phrase: "{phrase}"')
            score -= 10.0

    for starter in STEALTH_AI_STARTERS:
        if lower.strip().startswith(starter):
            issues.append('AI-like sentence opener')
            score -= 6.0
            break

    human_signals = sum(1 for h in STEALTH_HUMAN_INDICATORS if h in lower)
    score += human_signals * 3.0

    contractions = re.findall(r"\b\w+'(?:s|t|re|ve|ll|d|m)\b", sentence)
    if contractions:
        score += len(contractions) * 3.0

    if re.search(r'\b(i|me|my|we|us|our)\b', lower):
        score += 3.0
    if re.search(r'\byou\b', lower):
        score += 2.0
    if sentence.strip().endswith('?'):
        score += 4.0
    if sentence.strip().endswith('!'):
        score += 3.0
    if '(' in sentence and ')' in sentence:
        score += 3.0
    if re.match(r'^(and|but|so|because|also|plus|or|well|ok|hey)\b', lower.strip()):
        score += 4.0

    words_in_s = sentence.split()
    if (12 <= len(words_in_s) <= 22
            and sentence.strip().endswith('.')
            and not contractions
            and not re.search(r'[()\[\]—]', sentence)
            and human_signals == 0):
        issues.append('Uniform structure')
        score -= 6.0

    score = clamp(score)
    classification = 'human' if score >= 55.0 else ('maybe' if score >= 35.0 else 'ai')
    return {"text": sentence, "score": score, "classification": classification, "issues": issues}


def run_stealth_detector(text):
    sentences = split_sentences(text)
    if not sentences:
        return 50.0, "text too short", {}

    sentence_results = [stealth_analyze_sentence(s) for s in sentences]
    sentence_avg = sum(r["score"] for r in sentence_results) / len(sentence_results) if sentence_results else 50.0

    perplexity = stealth_calculate_perplexity(text)
    burstiness = stealth_calculate_burstiness(sentences)
    vocab = stealth_calculate_vocab_diversity(text)
    sentence_var = stealth_calculate_sentence_length_variation(sentences)
    transitions = stealth_calculate_transition_frequency(text)
    passive = stealth_calculate_passive_voice_ratio(sentences)
    ai_phrases = stealth_calculate_ai_phrase_density(text, sentences)
    sentence_start = stealth_calculate_sentence_start_diversity(sentences)
    pronoun = stealth_calculate_pronoun_usage(text)
    hedging = stealth_calculate_hedging_frequency(text)
    quantifier = stealth_calculate_quantifier_overuse(text)

    weights = {
        'sentenceAvg': 0.25,
        'perplexity': 0.15,
        'burstiness': 0.15,
        'vocabulary': 0.05,
        'sentenceVariation': 0.08,
        'transitions': 0.08,
        'passive': 0.05,
        'aiPhrases': 0.12,
        'sentenceStart': 0.05,
        'pronoun': 0.03,
        'hedging': 0.03,
        'quantifier': 0.02,
    }

    overall_score = (
        sentence_avg * weights['sentenceAvg'] +
        perplexity * weights['perplexity'] +
        burstiness * weights['burstiness'] +
        vocab * weights['vocabulary'] +
        sentence_var * weights['sentenceVariation'] +
        (100.0 - transitions) * weights['transitions'] +
        (100.0 - passive) * weights['passive'] +
        (100.0 - ai_phrases) * weights['aiPhrases'] +
        sentence_start * weights['sentenceStart'] +
        pronoun * weights['pronoun'] +
        (100.0 - hedging) * weights['hedging'] +
        (100.0 - quantifier) * weights['quantifier']
    )
    overall_score = clamp(overall_score)
    verdict = "human" if overall_score >= 55.0 else ("mixed" if overall_score >= 35.0 else "ai")

    detail = (f"StealthHumanizer 12-Metric: {overall_score:.1f}% ({verdict}), "
              f"PPL: {int(perplexity)}, Burst: {int(burstiness)}, Vocab: {int(vocab)}, Slop: {int(ai_phrases)}")

    return overall_score, detail, {
        "score": round(overall_score, 1),
        "verdict": verdict,
        "metrics": {
            "perplexity": round(perplexity, 1),
            "burstiness": round(burstiness, 1),
            "vocabularyDiversity": round(vocab, 1),
            "sentenceLengthVariation": round(sentence_var, 1),
            "transitionFrequency": round(transitions, 1),
            "passiveVoiceRatio": round(passive, 1),
            "aiPhraseDensity": round(ai_phrases, 1),
            "sentenceStartDiversity": round(sentence_start, 1),
            "pronounUsage": round(pronoun, 1),
            "hedgingFrequency": round(hedging, 1),
            "quantifierOveruse": round(quantifier, 1),
        },
        "sentences": sentence_results
    }


# ==================== 6-POINT LINGUISTIC PANEL ====================

def check_burstiness(text):
    sents = split_sentences(text)
    lens = [len(s.split()) for s in sents if len(s.split()) > 0]
    if len(lens) < 4:
        return 50.0, "too short to judge burstiness (need 4+ sentences)"

    mean = statistics.mean(lens)
    stdev = statistics.pstdev(lens)
    cv = stdev / mean if mean > 0 else 0.0

    consec_deltas = [abs(lens[i] - lens[i-1]) for i in range(1, len(lens))]
    mean_delta = statistics.mean(consec_deltas) if consec_deltas else 0.0
    delta_ratio = mean_delta / mean if mean > 0 else 0.0

    combined_metric = cv * 0.65 + delta_ratio * 0.35
    score = scale(combined_metric, human=0.65, machine=0.22)
    detail = f"CV {cv:.2f}, cadence delta {delta_ratio:.2f} across {len(lens)} sentences (want CV 0.55+)"
    return score, detail


def check_lexical_diversity(text):
    w = [word.lower() for word in words(text)]
    if len(w) < 30:
        return 50.0, "too short to judge lexical diversity"

    unique = set(w)
    ttr = len(unique) / len(w)

    counts = {}
    for word in w:
        counts[word] = counts.get(word, 0) + 1
    hapax = sum(1 for c in counts.values() if c == 1)
    hapax_ratio = hapax / len(w)

    lex_richness = ttr * 0.6 + hapax_ratio * 0.4
    score = scale(lex_richness, human=0.60, machine=0.35)
    detail = f"TTR {ttr:.2f}, unique ratio {hapax_ratio:.2f} ({len(unique)}/{len(w)} words)"
    return score, detail


def check_specificity(text):
    w = words(text)
    if len(w) < 25:
        return 50.0, "too short to judge specificity"

    per100 = 100.0 / len(w)
    num_hits = len(NUMBERS.findall(text))
    proper_hits = len(set(PROPER.findall(text)))
    total_hits = num_hits + proper_hits
    density = total_hits * per100
    score = scale(density, human=5.5, machine=0.8)
    detail = f"{total_hits} concrete markers ({num_hits} numbers, {proper_hits} entities), {density:.1f}/100 words"
    return score, detail


def check_slop(text, lex):
    w = words(text)
    if not w:
        return 50.0, "empty text"

    hits = 0
    found = []
    for entry in lex.get("words", []) + lex.get("phrases", []):
        find = entry["find"]
        pattern = re.compile(r"\b" + re.escape(find).replace(r"\ ", r"\s+") + r"\b", re.IGNORECASE)
        n = len(pattern.findall(text))
        if n:
            hits += n
            found.append(find)

    density = hits * 100.0 / len(w)
    score = scale(density, human=0.0, machine=3.5)
    detail = f"{hits} stock terms, {density:.1f}/100 words"
    if found:
        unique_found = sorted(set(found))
        detail += " (" + ", ".join(unique_found[:4]) + (", ..." if len(unique_found) > 4 else "") + ")"
    return score, detail


def check_fingerprint(text):
    invisible = sum(1 for c in text if unicodedata.category(c) == "Cf")
    em = text.count("—")
    curly = sum(text.count(c) for c in "‘’“”")
    ellip = text.count("…")
    nbsp = sum(text.count(c) for c in "\u00a0\u202f\u2009\u2007\u2003\u2002")
    total = invisible * 5 + em * 2 + curly + ellip + nbsp
    per1k = total * 1000.0 / max(len(text), 1)
    score = scale(per1k, human=0.0, machine=10.0)
    detail = (f"{invisible} invisible, {em} em dash, {curly} curly quote, "
              f"{ellip} ellipsis, {nbsp} hard space")
    return score, detail


def check_voice(text, lex):
    w = words(text)
    if len(w) < 25:
        return 50.0, "too short to judge voice"

    per100 = 100.0 / len(w)
    contractions = len(CONTRACTIONS.findall(text)) * per100
    person = len(PRONOUNS.findall(text)) * per100

    tells = 0
    names = []
    for s in lex.get("structures", []):
        try:
            n = len(re.compile(s["regex"], re.MULTILINE).findall(text))
        except re.error:
            continue
        if n:
            tells += n
            names.append(s["id"])

    bullets = [len(b.split()) for b in re.findall(r"(?m)^\s*[-*•]\s+(.+)$", text)]
    uniform = (len(bullets) >= 3 and statistics.pstdev(bullets) < 1.6)

    score = (scale(contractions, human=3.0, machine=0.0) * 0.35
             + scale(person, human=7.5, machine=1.0) * 0.35
             + clamp(100.0 - tells * 20.0) * 0.30)
    if uniform:
        score -= 12.0
        names.append("uniform-bullets")

    detail = (f"{contractions:.1f} contractions, {person:.1f} pronouns per 100 words, "
              f"{tells} structural tell(s)")
    if names:
        detail += " [" + ", ".join(sorted(set(names))[:4]) + "]"
    return clamp(score), detail


CHECKS = ["BURSTINESS", "LEXICAL RICHNESS", "SPECIFICITY", "SLOP DENSITY", "FINGERPRINT", "VOICE"]


def run(text, lex, neural_mode=True, stealth_mode=True, turnitin_mode=True, token=None, model=DEFAULT_LARGE_MODEL):
    results = {}
    results["BURSTINESS"] = check_burstiness(text)
    results["LEXICAL RICHNESS"] = check_lexical_diversity(text)
    results["SPECIFICITY"] = check_specificity(text)
    results["SLOP DENSITY"] = check_slop(text, lex)
    results["FINGERPRINT"] = check_fingerprint(text)
    results["VOICE"] = check_voice(text, lex)

    ling_scores = [results[c][0] for c in CHECKS]
    ling_mean = statistics.mean(ling_scores)
    ling_min = min(ling_scores)

    # 1. Neural Transformer Check (Document Level)
    neural_score = None
    if neural_mode:
        n_score, n_detail, n_engine = check_neural(text, token=token, model=model)
        if n_score is not None:
            results["NEURAL (ROBERTA LARGE)"] = (n_score, n_detail)
            neural_score = n_score
        else:
            results["NEURAL (ROBERTA LARGE)"] = (None, f"Unavailable ({n_detail})")

    # 2. Native StealthHumanizer 12-Metric Check
    stealth_score = None
    stealth_data = None
    if stealth_mode:
        s_score, s_detail, stealth_data = run_stealth_detector(text)
        results["STEALTHHUMANIZER (12-METRIC)"] = (s_score, s_detail)
        stealth_score = s_score

    # 3. Turnitin Sliding-Window Engine
    turnitin_data = None
    if turnitin_mode:
        turnitin_data = run_turnitin_sliding_window(text, token=token, model=model)
        t_human = turnitin_data["turnitin_human_index"]
        t_ai = turnitin_data["turnitin_ai_index"]
        t_detail = f"Turnitin Sliding Window: {t_human:.1f}% Human ({t_ai:.1f}% AI), {len(turnitin_data['flagged_sentences'])} flagged sents"
        results["TURNITIN AI INDEX"] = (t_human, t_detail)

    # Overall calculation:
    # Blend: 35% Turnitin Sliding Window + 30% Neural Large + 20% Stealth 12-Metric + 15% Linguistic
    if turnitin_data and neural_score is not None and stealth_score is not None:
        t_score = turnitin_data["turnitin_human_index"]
        overall = t_score * 0.35 + neural_score * 0.30 + stealth_score * 0.20 + (ling_min * 0.08 + ling_mean * 0.07)
        verdict = "PASS" if overall >= 75.0 and t_score >= 80.0 and ling_min >= 50.0 else (
            "REVIEW" if overall >= 50.0 else "FLAGGED"
        )
    elif neural_score is not None and stealth_score is not None:
        overall = neural_score * 0.40 + stealth_score * 0.30 + (ling_min * 0.15 + ling_mean * 0.15)
        verdict = "PASS" if overall >= 72.0 and ling_min >= 50.0 else (
            "REVIEW" if overall >= 50.0 else "FLAGGED"
        )
    else:
        overall = ling_mean * 0.60 + ling_min * 0.40
        verdict = "PASS" if overall >= 70.0 and ling_min >= 52.0 else (
            "REVIEW" if overall >= 50.0 else "FLAGGED"
        )

    return results, clamp(overall), verdict, stealth_data, turnitin_data


def bar(score, width=24):
    if score is None:
        return " " * width
    filled = round(score / 100.0 * width)
    return "#" * filled + "." * (width - filled)


def render(results, overall, verdict, label=None, stealth_data=None, turnitin_data=None, out=sys.stdout):
    title = "TURNITIN-GRADE SOTA AI DETECTION PANEL" + (f"  -  {label}" if label else "")
    print("\n" + title, file=out)
    print("=" * max(len(title), 74), file=out)

    order = CHECKS.copy()
    if "STEALTHHUMANIZER (12-METRIC)" in results:
        order.append("STEALTHHUMANIZER (12-METRIC)")
    if "NEURAL (ROBERTA LARGE)" in results:
        order.append("NEURAL (ROBERTA LARGE)")
    if "TURNITIN AI INDEX" in results:
        order.append("TURNITIN AI INDEX")

    for name in order:
        score, detail = results[name]
        if score is not None:
            print(f"  {name:<28} {bar(score)} {score:5.1f}", file=out)
        else:
            print(f"  {name:<28} {'[OFFLINE/SKIPPED]':<24}   ---", file=out)
        print(f"  {'':<28} {detail}", file=out)

    print("-" * 74, file=out)
    print(f"  {'OVERALL HUMAN SCORE':<28} {bar(overall)} {overall:5.1f}   [{verdict}]", file=out)

    # Turnitin Detailed Section
    if turnitin_data:
        ai_idx = turnitin_data["turnitin_ai_index"]
        flagged_w = turnitin_data["flagged_words"]
        total_w = turnitin_data["total_words"]
        flagged_s = turnitin_data["flagged_sentences"]

        print("\n  " + "=" * 70, file=out)
        print(f"  TURNITIN REPORT SUMMARY: {ai_idx:.1f}% AI  ({100.0 - ai_idx:.1f}% Human)", file=out)
        print(f"  Word Count: {total_w} | Flagged Words: {flagged_w} | Flagged Sentences: {len(flagged_s)}", file=out)
        if turnitin_data.get("advisory"):
            print(f"  Notice: {turnitin_data['advisory']}", file=out)

        if flagged_s:
            print(f"\n  {RED}Turnitin Highlighted Sentences (Simulated Cyan Highlighting):{RESET}", file=out)
            for s in flagged_s:
                print(f"    {CYAN_BG}[Turnitin AI Flag: {s['ai_probability']:.1f}% | {s['word_count']} words]{RESET}", file=out)
                print(f"    \"{s['text']}\"\n", file=out)
        else:
            print(f"  {GREEN}No sentences flagged by Turnitin sliding window model! (0% AI Highlight){RESET}", file=out)
        print("  " + "=" * 70, file=out)

    if verdict != "PASS":
        valid_checks = [c for c in CHECKS if results[c][0] is not None]
        weakest = min(valid_checks, key=lambda c: results[c][0])
        print(f"\n  Weakest linguistic signal: {weakest} ({results[weakest][0]:.1f}). Fix that first.", file=out)

    print("", file=out)


def main():
    ap = argparse.ArgumentParser(description="Turnitin-grade SOTA AI detector.")
    ap.add_argument("input", nargs="?", default="-", help="file, or - for stdin")
    ap.add_argument("compare", nargs="?", help="second file, to show before/after comparison")
    ap.add_argument("--json", action="store_true", help="emit output as JSON")
    ap.add_argument("--neural", dest="neural", action="store_true", default=True, help="enable neural classification (default: on)")
    ap.add_argument("--no-neural", dest="neural", action="store_false", help="disable neural classification")
    ap.add_argument("--stealth", dest="stealth", action="store_true", default=True, help="enable StealthHumanizer 12-metric engine (default: on)")
    ap.add_argument("--no-stealth", dest="stealth", action="store_false", help="disable StealthHumanizer 12-metric engine")
    ap.add_argument("--turnitin", dest="turnitin", action="store_true", default=True, help="enable Turnitin sliding window simulation (default: on)")
    ap.add_argument("--no-turnitin", dest="turnitin", action="store_false", help="disable Turnitin sliding window")
    ap.add_argument("--token", default=None, help="Hugging Face API token (overrides env / .env)")
    ap.add_argument("--model", default=DEFAULT_LARGE_MODEL, help=f"Hugging Face model ID (default: {DEFAULT_LARGE_MODEL})")
    ap.add_argument("--lexicon", default=LEX, help="path to slop.json")
    args = ap.parse_args()

    lex = json.load(open(args.lexicon, encoding="utf-8"))
    read = lambda p: sys.stdin.read() if p == "-" else open(p, encoding="utf-8").read()

    targets = [(args.input, read(args.input))]
    if args.compare:
        targets.append((args.compare, read(args.compare)))

    payload = []
    for name, text in targets:
        results, overall, verdict, s_data, t_data = run(
            text, lex,
            neural_mode=args.neural,
            stealth_mode=args.stealth,
            turnitin_mode=args.turnitin,
            token=args.token,
            model=args.model
        )
        payload.append({
            "source": name,
            "checks": {
                k: {"score": round(v[0], 1) if v[0] is not None else None, "detail": v[1]}
                for k, v in results.items()
            },
            "turnitin": t_data,
            "stealth_details": s_data,
            "human_score": round(overall, 1),
            "verdict": verdict,
        })

    if args.json:
        print(json.dumps(payload if args.compare else payload[0], indent=2, ensure_ascii=False))
        return

    for (name, text), p in zip(targets, payload):
        results, overall, verdict, s_data, t_data = run(
            text, lex,
            neural_mode=args.neural,
            stealth_mode=args.stealth,
            turnitin_mode=args.turnitin,
            token=args.token,
            model=args.model
        )
        render(results, overall, verdict, label=os.path.basename(name) if args.compare else None,
               stealth_data=s_data, turnitin_data=t_data)

    if args.compare:
        a, b = payload
        delta = b["human_score"] - a["human_score"]
        print(f"  Comparison: {a['human_score']:.1f} [{a['verdict']}]  ->  "
              f"{b['human_score']:.1f} [{b['verdict']}]   ({delta:+.1f})\n")

    sys.exit(0 if payload[-1]["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
