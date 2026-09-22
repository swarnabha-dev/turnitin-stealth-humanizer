---
name: li-human
description: >-
  Next-generation AI text detector and multi-layer humanizer. Evaluates drafts
  using a dual-layer architecture: Hugging Face RoBERTa Large (355M) cloud neural
  inference, native Python StealthHumanizer 12-metric engine, and an advanced 6-point
  linguistic/statistical panel. Strips machine watermarks, eliminates comma splices,
  balances rhythm, and produces authentically human prose with zero local npm/Node footprint.
---

# li-human: Next-Gen AI Detector & Stealth Humanizer

Integrated tools powered by **Cloud RoBERTa Large (355M)** and a **Native 12-Metric StealthHumanizer Engine** in pure Python (no Node server or npm dependencies required):

```bash
# Clean draft, preview changes, and view rhythm + StealthHumanizer diagnostics
python humanize.py draft.txt --report

# Neural rewrite via Qwen-72B and immediately verify Turnitin AI score
python humanize.py draft.txt --neural --verify

# Tone specific neural rewrite (e.g. story, builder, contrarian, casual, analytical, linkedin)
python humanize.py draft.txt --tone builder --verify

# Force local heuristic pipeline only, verify Turnitin score
python humanize.py draft.txt --local --verify

# Score draft using Turnitin Sliding-Window & Visual Cyan Highlighting
python detect.py draft.txt --turnitin

# Standard Multi-Engine Panel (RoBERTa Large 355M + Native Stealth 12-Metric + Linguistic)
python detect.py draft.txt

# Run purely offline (zero network/API)
python detect.py draft.txt --no-neural

# Compare before/after delta
python detect.py before.txt after.txt
```

### `humanize.py` CLI Arguments
- `--local`: force local heuristic pipeline only (no LLM call)
- `--neural`: use Qwen-72B neural rewrite (default: auto if HF_TOKEN present)
- `--tone {story,builder,contrarian,casual,analytical,linkedin}`: Voice archetype (default: story)
- `--verify`: automatically run detect.py on output to show Turnitin score
- `--report`: print what changed, to stderr
- `--out OUT`: write cleaned text here instead of stdout
- `--temp TEMP`: Sampling temperature for neural rewrite (default: 0.82)
- `--json`: emit {text, report} as JSON

### `detect.py` CLI Arguments
- `--json`: emit output as JSON
- `--no-neural`: disable neural classification (useful for offline mode)
- `--no-stealth`: disable StealthHumanizer 12-metric engine
- `--no-turnitin`: disable Turnitin sliding window
- `--turnitin`: explicitly enable Turnitin sliding window simulation (default: on)
- `--model MODEL`: Hugging Face model ID (default: openai-community/roberta-large-openai-detector)

---

## 1. Multi-Engine AI Detection Panel (`detect.py`)

`detect.py` scores drafts across three complementary layers:

### A. Turnitin-Grade Sliding-Window Engine (`--turnitin`)
- **3-Sentence Sliding Window**: Evaluates sentences within localized context windows, mimicking Turnitin's contextual evaluation rather than evaluating sentences in vacuum.
- **Turnitin AI % Formula**: Exact calculation based on $\frac{\text{Flagged Words}}{\text{Total Words}} \times 100\%$.
- **Terminal Highlighting**: Highlights flagged AI sentences with cyan/blue ANSI markers.
- **Cluster Gating**: Suppresses isolated short-sentence false positives unless neighboring context exhibits elevated AI confidence ($\ge 65\%$).

### B. Primary Cloud Neural: RoBERTa Large 355M (0% Local CPU Load)
- **Model**: `openai-community/roberta-large-openai-detector` (355M parameters, large transformer architecture).
- **Endpoint**: Hugging Face Inference Router (`https://router.huggingface.co/hf-inference/models/`).
- **Resilient Cloud Cascade**: Automatically cascades to `roberta-base` (125M) and `chatgpt-detector-roberta` if needed.
- **Zero Local Footprint**: Queries execute remotely via `urllib` using `HF_TOKEN` from `.env`.
- **Local Fallback**: Automatically falls back to local transformers pipeline (if installed) or offline engines when internet/quota is exhausted.

### C. Native StealthHumanizer 12-Metric Engine (Pure Python)
- **Zero Dependencies**: Full Python port of StealthHumanizer's 12 detection heuristics without needing Node.js or `node_modules`.
- **Metrics**: Perplexity, Burstiness, Vocabulary Diversity, Sentence Length Variation, Transition Frequency, Passive Voice Ratio, AI Phrase Density, Sentence Start Diversity, Pronoun Usage, Hedging Frequency, Quantifier Overuse, and Sentence-Level Risk Analysis.

### D. 6-Point Linguistic & Statistical Panel (Pure Python)
- **BURSTINESS 2.0**: Sentence CV + consecutive length delta (decimal, title, and abbreviation safe).
- **LEXICAL RICHNESS**: Type-Token Ratio (TTR) and Hapax Legomena ratio.
- **SPECIFICITY**: Concrete entities, metrics, currencies, and numbers per 100 words.
- **SLOP DENSITY**: Deep scan against curated 2026 AI buzzword and cliché lexicon (`slop.json`).
- **FINGERPRINT**: Invisible Unicode characters (category `Cf`), zero-width marks, em dashes, curly typography.
- **VOICE**: Contractions, natural pronoun usage, and structural tropes.

---

## 2. Integrated StealthHumanizer Writing Workflow

Antigravity operates directly as your humanizer engine using the complete StealthHumanizer methodology:
1. **Purity Pass**: Normalizes Unicode, eliminates invisible watermarks, straightens quotes, and resolves em dashes context-awarely without creating comma splices.
2. **Stealth Style Pass**: Applies StealthHumanizer's Anti-Detection Core (deliberate sentence length asymmetry, high token perplexity, unexpected conversational domain phrasing, zero AI collocations).
3. **Verification Loop**: Scans the output against `detect.py` (RoBERTa Large + Native Stealth Engine + Linguistic Panel) and refines flagged lines until achieving an **85–100% human score**.
