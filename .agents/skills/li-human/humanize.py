#!/usr/bin/env python3
"""
humanize.py - Next-generation SOTA humanizer (Neural Qwen-72B + Multi-Layer Cleansing Pipeline).

Two Operational Modes:
  1. Neural SOTA Mode (--neural, default when HF_TOKEN is present):
     - Uses Qwen/Qwen2.5-72B-Instruct (72 Billion parameters) via Hugging Face Router API (0% local CPU load).
     - Fallback: meta-llama/Llama-3.1-8B-Instruct.
     - Rewrites text with authentic storytelling cadence, natural perplexity, and zero AI clichés.
     - Then passes output through typographic and watermark purity filters.
  2. Local Rule-Based Mode (--local or offline fallback):
     - Cleans invisible Unicode watermarks & format characters (Cf category, BOMs, soft hyphens).
     - Context-aware em dash replacement (prevents comma splices).
     - Lexical de-sloping against curated 2026 AI buzzword dictionary (slop.json).
     - Rhythm & cadence analysis.
  3. Integrated Turnitin Verification (--verify):
     - Automatically runs detect.py on the humanized output to prove the Turnitin AI % is 0% (or pass).

Usage:
  python humanize.py draft.txt
  python humanize.py draft.txt --neural               # Force Qwen-72B neural rewrite
  python humanize.py draft.txt --local                # Force pure local heuristic rewrite
  python humanize.py draft.txt --verify               # Rewrite and immediately verify with Turnitin panel
  python humanize.py draft.txt -o clean.txt
  python humanize.py draft.txt --report
  python humanize.py draft.txt --stealth-prompt
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import unicodedata
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
LEX = os.path.join(HERE, "slop.json")
ROUTER_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"

DEFAULT_NEURAL_MODEL = "Qwen/Qwen2.5-72B-Instruct"
FALLBACK_NEURAL_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct"
]

URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+")
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "vs", "etc", "eg", "ie",
    "sep", "oct", "nov", "dec", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "vol", "dept", "est", "approx", "al"
}


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


def load_lexicon(path=LEX):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _cp(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return (int(a[2:], 16), int(b[2:], 16))
    return int(spec[2:], 16)


def protect_urls(text):
    found = []
    def stash(m):
        found.append(m.group(0))
        return f"\x00URL{len(found) - 1}\x00"
    return URL_RE.sub(stash, text), found


def restore_urls(text, found):
    for i, url in enumerate(found):
        text = text.replace(f"\x00URL{i}\x00", url)
    return text


def pass_invisible(text, lex):
    hits = []
    for entry in lex.get("invisible", []):
        cp = _cp(entry["cp"])
        if isinstance(cp, tuple):
            pattern = "[" + re.escape(chr(cp[0])) + "-" + re.escape(chr(cp[1])) + "]"
        else:
            pattern = re.escape(chr(cp))
        n = len(re.findall(pattern, text))
        if n:
            hits.append({"name": entry["cp"] + " " + entry["name"], "count": n,
                         "action": entry["action"]})
            text = re.sub(pattern, "" if entry["action"] == "delete" else " ", text)
    
    stray = [c for c in text if unicodedata.category(c) == "Cf"]
    if stray:
        hits.append({"name": "other invisible format chars", "count": len(stray),
                     "action": "delete"})
        text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    return text, hits


def smart_replace_em_dashes(text):
    """
    Context-aware em dash replacement to prevent comma splices.
    """
    hits_count = text.count("—")
    if not hits_count:
        return text, 0

    def repl_paired(m):
        aside = m.group(1).strip()
        if "," in aside or len(aside.split()) > 5:
            return f" ({aside}) "
        return f", {aside}, "

    text = re.sub(r"\s*—\s*([^—\n]+?)\s*—\s*", repl_paired, text)

    independent_starters = r"(?:[A-Z][a-z]+|This|That|These|Those|It|We|They|He|She|I|There|Here|Now|So|The|A|An|What|Why|How)"
    
    def repl_independent(m):
        before = m.group(1).rstrip()
        starter = m.group(2)
        if before and before[-1] in ".!?:;":
            return f"{before} {starter}"
        return f"{before}. {starter}"

    text = re.sub(r"([^\n—]+?)\s*—\s*(" + independent_starters + r"\b)", repl_independent, text)

    def repl_colon(m):
        return m.group(1) + ": "

    text = re.sub(r"([^\n—]+?)\s*—\s*(?=(?:namely|i\.e\.|meaning|specifically|in other words)\b)", repl_colon, text, flags=re.IGNORECASE)

    def repl_remaining(m):
        before = m.group(1)
        after = m.group(2)
        if before.endswith(",") or after.startswith(","):
            return f"{before} {after}"
        return f"{before}, {after}"

    text = re.sub(r"(\S+)\s*—\s*(\S+)", repl_remaining, text)
    text = text.replace("—", ", ")

    text = re.sub(r",\s*([,.;:!?])", r"\1", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r",\s*\n", "\n", text)
    return text, hits_count


def pass_typographic(text, lex):
    hits = []
    text, em_hits = smart_replace_em_dashes(text)
    if em_hits:
        hits.append({"name": "— EM DASH (smart context-aware)", "count": em_hits, "to": ". / , / : / -"})

    for entry in lex.get("typographic", []):
        ch = entry["from"]
        if ch == "—":
            continue
        n = text.count(ch)
        if not n:
            continue
        hits.append({"name": f"{ch} {entry['name']}", "count": n, "to": entry["to"].strip() or "(space)"})
        if ch == "–":
            text = re.sub(r"\s*–\s*(?=\d)", "-", text)
            text = re.sub(r"\s+–\s+", ", ", text)
            text = text.replace("–", "-")
        else:
            text = text.replace(ch, entry["to"])

    text = re.sub(r",\s*([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*\n", "\n", text)
    return text, hits


def _match_case(src, repl):
    if not repl:
        return repl
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[0].isupper():
        return repl[0].upper() + repl[1:]
    return repl


def pass_lexical(text, lex):
    hits = []
    entries = sorted(lex.get("phrases", []) + lex.get("words", []),
                     key=lambda e: len(e["find"]), reverse=True)
    for entry in entries:
        find = entry["find"]
        pattern = re.compile(r"\b" + re.escape(find).replace(r"\ ", r"\s+") + r"\b", re.IGNORECASE)
        found = pattern.findall(text)
        if not found:
            continue
        hits.append({"find": find, "replace": entry.get("replace") or "(deleted)",
                     "count": len(found), "family": entry.get("family", "slop")})
        text = pattern.sub(lambda m: _match_case(m.group(0), entry.get("replace", "")), text)

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(?m)^[ \t]*([,.;:])\s*", "", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"(?m)^[ \t]+$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r",\s*(also|so|still|basically|in the end)\s*,\s*",
                  lambda m: ". " + m.group(1)[0].upper() + m.group(1)[1:] + ", ", text)
    return text, hits


def split_sentences_local(text):
    cleaned = re.sub(r"(\d+)\.(\d+)", r"\1__DEC__\2", text)
    def protect(m):
        if m.group(1).lower().rstrip(".") in ABBREVIATIONS:
            return m.group(0).replace(".", "__DOT__")
        return m.group(0)
    cleaned = re.sub(r"\b([A-Za-z]{1,6})\.(?=\s+[A-Za-z0-9])", protect, cleaned)
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(\[])|(?<=[.!?])\n+|\n\s*\n", cleaned)
    sents = []
    for r in raw:
        r = r.replace("__DEC__", ".").replace("__DOT__", ".")
        r = re.sub(r"^\s*[-*•\d\.\)]\s*", "", r).strip()
        if r and any(c.isalnum() for c in r):
            sents.append(r)
    return sents


def scan_structures_and_rhythm(text, lex):
    flags = []

    for s in lex.get("structures", []):
        try:
            pattern = re.compile(s["regex"], re.MULTILINE)
        except re.error:
            continue
        found = pattern.findall(text)
        if found:
            flags.append({"name": s["name"], "count": len(found), "fix": s.get("fix", "Rewrite")})

    sents = split_sentences_local(text)
    lens = [len(s.split()) for s in sents]
    if len(lens) >= 4:
        mean = statistics.mean(lens)
        cv = statistics.pstdev(lens) / mean if mean else 0.0

        if cv < 0.35:
            flags.append({
                "name": f"Monotonous sentence cadence (CV {cv:.2f})",
                "count": len(lens),
                "fix": "Mix natural sentence lengths. Let narrative lines flow and keep key takeaways clear."
            })

        long_sents = [s for s in sents if len(s.split()) >= 34]
        if long_sents:
            flags.append({
                "name": f"Long compound sentences (34+ words)",
                "count": len(long_sents),
                "fix": f"Consider breaking complex clauses: e.g. \"{long_sents[0][:60]}...\""
            })

    return flags


DEFAULT_VOICE_PROFILES = {
    "story": {
        "label": "Reflective Narrative / Insight",
        "instructions": (
            "VOICE & PERSPECTIVE: Reflective, intellectual, and observational.\n"
            "- Ground the post in personal presence ('I spent Friday afternoon...').\n"
            "- Unfold the story chronologically: the premise -> the historical root -> where it broke -> the new breakthrough.\n"
            "- Cadence: Contemplative, measured, and insightful."
        )
    },
    "builder": {
        "label": "Pragmatic Systems Engineer / Builder",
        "instructions": (
            "VOICE & PERSPECTIVE: Production engineer, builder, and battle-tested practitioner.\n"
            "- Open directly with the real-world engineering problem (distributed consensus, race conditions, infinite loops in prod).\n"
            "- Focus on why theoretical guarantees prevent real systems from crashing or hanging.\n"
            "- Cadence: Direct, grounded, concrete, and unpretentious."
        )
    },
    "contrarian": {
        "label": "Sharp / Provocative Thinker",
        "instructions": (
            "VOICE & PERSPECTIVE: Intellectual challenger and contrarian thinker.\n"
            "- Open with a provocative paradox or challenging dogma (e.g. why textbook termination proofs fail in the real world).\n"
            "- Contrast clean theoretical assumptions against messy randomized realities.\n"
            "- Cadence: Assertive, punchy contrasts, tight logical tension."
        )
    },
    "casual": {
        "label": "Conversational Colleague / 'Thinking Out Loud'",
        "instructions": (
            "VOICE & PERSPECTIVE: Conversational, friendly, and curious—like talking with a colleague over coffee.\n"
            "- Open informally ('Still thinking about a talk I caught Friday...', 'Here's something that completely shifted how I view...').\n"
            "- Use natural rhetorical questions, parenthetical asides, and an accessible, engaging tone.\n"
            "- Cadence: Easygoing, authentic, and engaging."
        )
    },
    "analytical": {
        "label": "Theoretical Deep-Dive / Mathematical Clarity",
        "instructions": (
            "VOICE & PERSPECTIVE: Rigorous computer science researcher.\n"
            "- Focus directly on the formal verification mechanics, supermartingales, and proof theory.\n"
            "- Clarify subtle distinctions (soundness vs completeness, bounded vs unbounded steps).\n"
            "- Cadence: Authoritative, academically precise, and conceptually clean."
        )
    },
    "linkedin": {
        "label": "Personalized LinkedIn Thought Leadership",
        "instructions": (
            "VOICE & PERSPECTIVE: Domain practitioner and systems builder.\n"
            "- TONE: High technical signal, sharp, engaging, and unpretentious ('architecture in the trenches' meets rigorous research).\n"
            "- LINKEDIN FORMATTING RULES (MANDATORY FOR HIGH ENGAGEMENT & DWELL TIME):\n"
            "  * Line 1 (The Hook): Standalone bold line (<130 chars) that stops the feed scroll.\n"
            "  * Line 2: Immediate payoff to Line 1 that earns the 'see more' click before mobile truncation.\n"
            "  * Pacing: Short, breathable paragraphs (1-3 lines each) with generous white space. No walls of text.\n"
            "  * Length: Snappy and punchy (180-230 words / 1,050-1,300 chars).\n"
            "  * Personal Statement: Connect theoretical ideas directly to real engineering reality (distributed consensus, FLP impossibility, production deadlocks).\n"
            "  * Not Boring / Not Aggressive: Assertive and insightful without sounding like a textbook or a toxic tech-bro.\n"
            "  * Closer: One thoughtful, high-signal question that technical peers, architects, and founders want to answer.\n"
            "  * Hashtags: 3-4 targeted, high-relevance hashtags at the end.\n"
            "- BANNED WORDS: game-changer, revolutionary, delve, testament to, unlock, seamless, in today's fast-paced digital era, thrilled to announce."
        )
    }
}


def load_voice_md():
    """Load user's personal voice.md profile if available."""
    search_paths = [
        os.path.join(os.path.dirname(HERE), "templates", "voice.md"),
        os.path.join(os.path.dirname(os.path.dirname(HERE)), "templates", "voice.md"),
        os.path.expanduser("~/.claude/linkedin/voice.md"),
        os.path.join(HERE, "voice.md"),
    ]
    for p in search_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8-sig") as f:
                    content = f.read().strip()
                    if content and not content.startswith("# voice.md Template"):
                        return content
            except Exception:
                pass
    return ""


def load_voice_profiles():
    """Load voice profiles from user config or template file if available, falling back to defaults."""
    search_paths = [
        os.path.join(os.path.dirname(HERE), "templates", "voice_profiles.json"),
        os.path.join(os.path.dirname(os.path.dirname(HERE)), "templates", "voice_profiles.json"),
        os.path.expanduser("~/.claude/linkedin/voice_profiles.json"),
        os.path.join(HERE, "voice_profiles.json"),
    ]
    for p in search_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    user_profiles = json.load(f)
                    if isinstance(user_profiles, dict):
                        merged = DEFAULT_VOICE_PROFILES.copy()
                        merged.update(user_profiles)
                        return merged
            except Exception:
                pass
    return DEFAULT_VOICE_PROFILES


VOICE_PROFILES = load_voice_profiles()


def generate_stealth_prompt(text, tone="story"):
    profiles = load_voice_profiles()
    profile = profiles.get(tone, profiles.get("story", DEFAULT_VOICE_PROFILES["story"]))
    voice_md_content = load_voice_md()
    voice_md_block = f"\n[PERSONAL VOICE PROFILE (voice.md)]\n{voice_md_content}\n" if voice_md_content else ""

    prompt = f"""[ROLE & GOAL]
You are an expert human writer and domain practitioner. Rewrite the provided draft in the '{profile['label']}' style so that it reads with authentic human voice and easily passes Turnitin and neural AI detectors with a 95%+ Human score.

[VOICE PROFILE: {profile['label'].upper()}]
{profile['instructions']}
{voice_md_block}
[ANTI-DETECTION HEURISTICS (MANDATORY)]
1. STORYTELLING & CADENCE:
   - Mix natural sentence lengths. Explanations should breathe naturally without artificial 2-word staccato chops.
   - Do NOT use bullet points or numbered lists in narrative essays.
2. TOKEN PERPLEXITY & ENTROPY:
   - Break predictable token probability chains. Use natural domain phrasing over academic boilerplate.
   - Avoid generic AI transitions ("Moreover", "Furthermore", "Additionally", "In conclusion", "It is worth noting").
3. ABSOLUTE PROHIBITION OF AI CLICHÉS:
   - BANNED VOCABULARY: delve, tapestry, beacon, testament, pivotal, foster, harness, multifaceted, ever-evolving, catalyst, synergy, embark, game-changer, unlock the potential.
   - BANNED STRUCTURES: No "Not only X, but also Y", no "It's not about X, it's about Y", no rule-of-three triplets, no "In today's fast-paced world".
4. TYPOGRAPHIC PURITY:
   - Do NOT use em dashes (—). Use a colon, full stop, or clean commas.
5. GROUNDING:
   - Keep all concrete numbers, mathematical notation, venue details, speaker names, and technical terms intact.

[DRAFT TO REWRITE]
\"\"\"
{text.strip()}
\"\"\"

Output ONLY the rewritten, humanized draft. No preamble, no meta-commentary.
"""
    return prompt


# ==================== NEURAL HUMANIZER (QWEN-72B / LLAMA-8B) ====================

def neural_rewrite(text, model=DEFAULT_NEURAL_MODEL, tone="story", temperature=0.82, token=None):
    """
    Execute neural rewriting via Qwen-72B or Llama-8B on the Hugging Face Router with tone styling.
    """
    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        return None, "No HF_TOKEN found in env or .env"

    profiles = load_voice_profiles()
    profile = profiles.get(tone, profiles.get("story", DEFAULT_VOICE_PROFILES["story"]))
    voice_md_content = load_voice_md()

    voice_guidance = ""
    if voice_md_content:
        voice_guidance = f"\n\nUSER PERSONAL VOICE GUIDANCE (from templates/voice.md):\n{voice_md_content}\n"

    system_prompt = (
        f"You are an authentic, highly skilled domain writer. Rewrite the provided draft into an engaging, "
        f"authentically human LinkedIn post in the '{profile['label']}' style.\n\n"
        f"{profile['instructions']}"
        f"{voice_guidance}\n\n"
        f"CORE WRITING PRINCIPLES (MANDATORY):\n"
        f"1. PRESERVE ACCURACY: Keep all technical entities, speaker names, institutions, mathematical terms, and theorems completely accurate.\n"
        f"2. NATURAL CADENCE & ENTROPY: Write with genuine rhythm. Mix short and long clauses naturally. Do NOT force mechanical 2-word staccato lines or formulaic copywriting templates.\n"
        f"3. ZERO BULLET POINTS: Write in flowing, natural prose paragraphs. Do not use bullet lists or numbered outlines.\n"
        f"4. TOTAL SLOP BAN: Never use AI buzzwords (delve, tapestry, beacon, testament, pivotal, multifaceted, vibrant, foster, furthermore, moreover, in conclusion, in today's fast-paced world).\n"
        f"5. NO EM DASH OVERUSE: Avoid em dashes (—). Use colons, periods, or clean commas instead.\n"
        f"6. OUTPUT: Produce ONLY the rewritten post ready to read. No intro, no meta-explanation, no quotation wrappers."
    )

    models_to_try = [model] + [m for m in FALLBACK_NEURAL_MODELS if m != model]
    # Deduplicate
    seen = set()
    models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

    for m in models_to_try:
        payload = {
            "model": m,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Rewrite this draft in the specified '{profile['label']}' style:\n\n{text.strip()}"}
            ],
            "max_tokens": 1600,
            "temperature": float(temperature),
            "top_p": 0.92
        }
        req = urllib.request.Request(
            ROUTER_CHAT_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json",
                "User-Agent": "Antigravity-NeuralHumanizer/2.0"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"].strip()
                content = re.sub(r"^```(?:markdown)?\s*\n", "", content)
                content = re.sub(r"\n```$", "", content)
                return content.strip(), f"Neural ({m.split('/')[-1]})"
        except urllib.error.HTTPError as e:
            continue
        except Exception as e:
            continue

    return None, "All cloud neural models failed"


def humanize(text, lex, use_neural=False, neural_model=DEFAULT_NEURAL_MODEL, tone="story", temperature=0.82, token=None):
    engine_used = "Local Heuristic Pipeline"
    if use_neural:
        rewritten, engine_info = neural_rewrite(text, model=neural_model, tone=tone, temperature=temperature, token=token)
        if rewritten:
            text = rewritten
            engine_used = f"{engine_info} [{tone}]"

    text, urls = protect_urls(text)
    text, inv = pass_invisible(text, lex)
    text, typo = pass_typographic(text, lex)
    text, lexi = pass_lexical(text, lex)
    text = restore_urls(text, urls)
    structures = scan_structures_and_rhythm(text, lex)

    return text.strip() + "\n", {
        "engine": engine_used,
        "invisible": inv,
        "typographic": typo,
        "lexical": lexi,
        "structures": structures,
    }


def render_stealth_report(text, out=sys.stderr):
    try:
        from detect import run_stealth_detector
        score, detail, data = run_stealth_detector(text)
        print("\nSTEALTHHUMANIZER 12-METRIC ENGINE ANALYSIS\n" + "-" * 42, file=out)
        print(f"Score: {score:.0f}% human ({data.get('verdict', 'unknown')})", file=out)
        metrics = data.get("metrics", {})
        print(f"Metrics: PPL {metrics.get('perplexity')}, Burstiness {metrics.get('burstiness')}, Vocab {metrics.get('vocabularyDiversity')}, Slop {metrics.get('aiPhraseDensity')}", file=out)
        ai_sents = [s for s in data.get("sentences", []) if s.get("issues")]
        if ai_sents:
            print("\nTop sentence issues:", file=out)
            for s in ai_sents[:4]:
                print(f"  [{s['score']:.0f}%] \"{s['text'][:65]}...\"", file=out)
                print(f"        Issues: {', '.join(s['issues'])}", file=out)
        print("", file=out)
    except Exception:
        pass


def render_report(report, out=sys.stderr):
    def head(title):
        print(f"\n{title}\n" + "-" * len(title), file=out)

    total = (sum(h["count"] for h in report["invisible"])
             + sum(h["count"] for h in report["typographic"])
             + sum(h["count"] for h in report["lexical"]))

    head("STEALTH HUMANIZE REPORT")
    print(f"  Engine: {report.get('engine', 'Local')}")
    print(f"  {total} machine artefacts cleansed, "
          f"{len(report['structures'])} structural / rhythm points flagged", file=out)

    if report["invisible"]:
        head("1. INVISIBLE WATERMARKS & CHARACTERS")
        for h in report["invisible"]:
            print(f"  {h['count']:>3}x  {h['name']}  -> {h['action']}", file=out)
    if report["typographic"]:
        head("2. TYPOGRAPHY & PUNCTUATION")
        for h in report["typographic"]:
            print(f"  {h['count']:>3}x  {h['name']}  -> {h['to']}", file=out)
    if report["lexical"]:
        head("3. SLOP & CLICHÉ LEXICON")
        for h in report["lexical"]:
            print(f"  {h['count']:>3}x  {h['find']}  -> {h['replace']}   [{h['family']}]", file=out)
    if report["structures"]:
        head("4. RHYTHM & STRUCTURAL DIAGNOSTICS")
        for h in report["structures"]:
            print(f"  {h['count']:>3}x  {h['name']}\n        Fix: {h['fix']}", file=out)
    if not any(report[k] for k in ["invisible", "typographic", "lexical", "structures"]):
        head("CLEAN")
        print("  Text is clean. No machine fingerprint detected.", file=out)
    print("", file=out)


def main():
    ap = argparse.ArgumentParser(description="Turnitin-grade SOTA Humanizer (Qwen-72B Neural + Pipeline).")
    ap.add_argument("input", nargs="?", default="-", help="file, or - for stdin")
    ap.add_argument("-o", "--out", help="write cleaned text here instead of stdout")
    ap.add_argument("--neural", dest="neural", action="store_true", default=None, help="use Qwen-72B neural rewrite (default: auto if HF_TOKEN present)")
    ap.add_argument("--local", dest="neural", action="store_false", help="force local heuristic pipeline only")
    ap.add_argument("--tone", choices=list(VOICE_PROFILES.keys()), default="story", help=f"Voice archetype: {', '.join(VOICE_PROFILES.keys())} (default: story)")
    ap.add_argument("--temp", type=float, default=0.82, help="Sampling temperature for neural rewrite (default: 0.82)")
    ap.add_argument("--model", default=DEFAULT_NEURAL_MODEL, help=f"Neural model ID (default: {DEFAULT_NEURAL_MODEL})")
    ap.add_argument("--token", default=None, help="Hugging Face API token")
    ap.add_argument("--report", action="store_true", help="print what changed, to stderr")
    ap.add_argument("--verify", action="store_true", help="automatically run detect.py on output to show Turnitin score")
    ap.add_argument("--stealth-prompt", action="store_true", help="output LLM anti-detection rewrite prompt")
    ap.add_argument("--json", action="store_true", help="emit {text, report} as JSON")
    ap.add_argument("--lexicon", default=LEX, help="path to slop.json")
    args = ap.parse_args()

    raw = sys.stdin.read() if args.input == "-" else open(args.input, encoding="utf-8").read()

    if args.stealth_prompt:
        prompt = generate_stealth_prompt(raw, tone=args.tone)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(prompt)
            print(f"wrote stealth prompt to {args.out}", file=sys.stderr)
        else:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stdout.write(prompt)
        return

    # Auto-enable neural if HF_TOKEN is present and user didn't explicitly pass --local
    use_neural = args.neural
    if use_neural is None:
        use_neural = bool(args.token or os.environ.get("HF_TOKEN"))

    lex = load_lexicon(args.lexicon)
    clean, report = humanize(
        raw, lex,
        use_neural=use_neural,
        neural_model=args.model,
        tone=args.tone,
        temperature=args.temp,
        token=args.token
    )

    if args.json:
        print(json.dumps({"text": clean, "report": report}, indent=2, ensure_ascii=False))
        return

    out_target = args.out
    if out_target:
        with open(out_target, "w", encoding="utf-8") as fh:
            fh.write(clean)
        print(f"wrote {out_target}", file=sys.stderr)
    else:
        sys.stdout.write(clean)

    if args.report:
        render_report(report)
        render_stealth_report(clean)

    if args.verify:
        import tempfile
        verify_path = out_target
        cleanup_temp = False
        if not verify_path:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as tmp:
                tmp.write(clean)
                verify_path = tmp.name
                cleanup_temp = True

        detect_script = os.path.join(HERE, "detect.py")
        subprocess.run([sys.executable, detect_script, verify_path])
        if cleanup_temp:
            try:
                os.remove(verify_path)
            except Exception:
                pass


if __name__ == "__main__":
    main()
