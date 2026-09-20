# Turnitin-Grade SOTA AI Detector & Stealth Humanizer

A SOTA **Turnitin-grade AI detector and multi-archetype neural humanizer** engine, combined with specialized agentic skills for LinkedIn content strategy, profile optimization, and thought leadership.

Created and enhanced by **Swarnabha Halder** ([@swarnabha-dev](https://github.com/swarnabha-dev)).

---

## Key Features & Architecture

### 1. Turnitin-Grade Sliding-Window AI Detection (`detect.py`)
- **Sliding-Window Neural Analysis (`--turnitin`)**: Evaluates 3-sentence sliding context windows using parallel cloud inference with **RoBERTa Large 355M** (`openai-community/roberta-large-openai-detector`) at **0% local CPU load**.
- **Turnitin AI % Formula**: Calculates exact Turnitin index: $\text{Turnitin AI \%} = \left(\frac{\text{Flagged Words}}{\text{Total Words}}\right) \times 100\%$.
- **Cluster Gating & Visual Highlighting**: Suppresses isolated short false positives unless neighboring context shows elevated AI probability ($\ge 65\%$). Highlights flagged AI sentences in terminal ANSI cyan/blue markers.
- **Native 12-Metric Stealth Humanizer**: Pure Python port of 12 heuristics (Perplexity, Burstiness CV, Vocabulary TTR, Passive Voice, AI Phrase Density, Quantifier Overuse, Sentence Start Diversity). Zero Node.js or npm footprint.

### 2. Multi-Archetype Neural Humanizer (`humanize.py`)
- **SOTA Cloud Neural Engine (`--neural`)**: Leverages `Qwen/Qwen2.5-72B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct` via Hugging Face Router API.
- **5 Dynamic Voice Archetypes (`--tone`)**:
  - `linkedin`: Personalized LinkedIn thought leadership (optimized for high dwell time, mobile truncation, and punchy whitespace).
  - `builder`: Pragmatic systems engineering, production realities, and architecture in the trenches.
  - `casual`: Conversational, "thinking out loud", unpretentious.
  - `contrarian`: Intellectual challenger, sharp contrasts, puncturing industry dogma.
  - `analytical`: Deep-dive theoretical clarity.
- **Closed-Loop Turnitin Verification (`--verify`)**: Automatically runs `detect.py` after humanization to confirm a verified 0-5% Turnitin AI score.
- **Smart Punctuation & Typographic Purity**: Removes invisible Unicode watermarks (`Cf` category) and resolves em dashes context-awarely without creating comma splices.

---

## Quick Start & Installation

### 1. Clone Repository
```bash
git clone https://github.com/swarnabha-dev/turnitin-stealth-humanizer.git
cd turnitin-stealth-humanizer
```

### 2. Configure Environment & Voice
```bash
# Copy template files
cp .env.template .env
cp templates/voice.md.template templates/voice.md

# Add your Hugging Face API Token into .env for 0% CPU cloud neural inference:
# HF_TOKEN=hf_...
```

---

## Humanizer Usage & Commands

```bash
# Score a draft using Turnitin Sliding-Window & Visual Cyan Highlighting
python skills/li-human/detect.py draft.txt --turnitin

# Humanize using Qwen-72B Neural Engine + LinkedIn Voice & verify Turnitin score
python skills/li-human/humanize.py draft.txt --tone linkedin --neural --verify

# Humanize using Pragmatic Builder tone
python skills/li-human/humanize.py draft.txt --tone builder --neural --verify

# Pure local offline heuristic execution (zero API/network)
python skills/li-human/detect.py draft.txt --no-neural
```

---

## The Eleven Agent Skills

| Command | Description |
| --- | --- |
| `/li-human` | Next-gen Turnitin-grade AI detector & SOTA neural humanizer engine. |
| `/li-post` | One idea into a post using 21 proven hook formulas and personalized voice. |
| `/li-comment` | Write authentic, non-bot comments on other people's posts across 9 types. |
| `/li-reply` | Triage and draft replies for your post's comment section. |
| `/li-profile` | Score your profile out of 100 against a 12-part rubric and rewrite weak areas. |
| `/li-plan` | Build your weekly content schedule and engagement hit-list. |
| `/li-carousel` | Generate slide-by-slide copy, cover hooks, and document post structure. |
| `/li-repurpose` | Turn long-form videos, podcasts, or transcripts into a week of standalone posts. |
| `/li-dm` | High-conversion connection requests, first messages, and follow-up sequences. |
| `/li-inbox` | Sort inbox messages into leads, recruiters, peers, and spam. |
| `/li-audit` | Post-mortem analysis on published posts to measure reach multiples and engagement. |

---

## Credits & License

- **Enhanced & Maintained by**: Swarnabha Halder ([swarnabha-dev](https://github.com/swarnabha-dev)) - Turnitin sliding window engine, 355M cloud RoBERTa Large integration, pure Python Stealth Humanizer engine, and multi-archetype neural rewrite pipeline.
- **Original Skill Concept**: Credit to Jake Schincariol ([opusjake.ai](https://opusjake.ai)).
- **License**: MIT License.
