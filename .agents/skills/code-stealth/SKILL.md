---
name: code-stealth
description: >-
  Dedicated methodology and workflow for producing AI-undetectable and 
  plagiarism-free (MOSS-bypassed) code. Uses Abstract Syntax Tree (AST) 
  manipulation, Halstead metric shifting, and semantic restructuring.
---

# code-stealth: AI & Plagiarism Code Evader

This skill equips the agent with the exact methodology to bypass institutional code plagiarism checkers (like Stanford MOSS) and AI code detectors (like Turnitin Code, Codeleaks).

## 1. How AI & Plagiarism Detectors Analyze Code

Detectors do **not** just read the text of the code. They strip away whitespace, comments, and variable names, and compile the code into an **Abstract Syntax Tree (AST)**. 

### Plagiarism (MOSS) Mechanics:
- **K-Gram Hashing:** MOSS breaks the AST into overlapping chunks (k-grams) and hashes them (Winnowing algorithm). 
- **Variable Renaming is Useless:** Simply changing `index` to `i` or adding empty lines does not change the AST. MOSS will still flag it as a 100% match.
- **Vulnerability:** MOSS is sensitive to control-flow alterations. Changing a `for` loop to a `while` loop, or swapping the order of independent operations, generates entirely different AST hashes.

### AI Detection Mechanics:
- **Canonical Structures:** AI models heavily favor "textbook" implementations (e.g., standard `zip()` iterations, list comprehensions, `dp` array setups).
- **Halstead Complexity:** AI code usually minimizes operators and maximizes operands efficiently. 
- **Lexical Fingerprints:** AI uses highly predictable variable aliases (`y_hat`, `grad_w`, `result`, `temp`, `curr`) and writes comments that strictly narrate the next line of code rather than the developer's architectural intent.

---

## 2. The Code-Stealth Methodology

Whenever you are asked to write or humanize code using this skill, you must apply the **AST Transformation Protocol**. Do not just rename variables—you must fundamentally alter the execution graph.

### Technique A: Control-Flow Inversion (Breaks MOSS)
- **Swap Loop Types:** Convert standard `for` loops into `while` loops, or vice versa.
- **Reverse Iteration:** Instead of counting up `0` to `n`, count down `n-1` to `0` where mathematically safe.
- **Flatten Comprehensions:** Unroll standard list/dictionary comprehensions into procedural loops with manual `.append()` or index assignment.
- **Branch Restructuring:** Swap `if / else` orderings by negating the boolean condition. Extract nested conditions into separate early-return checks.

### Technique B: Semantic Aliasing (Breaks AI Detectors)
- **Non-Textbook Naming:** Never use `n = len(x)`. Use context-specific names like `row_count`, `seq_len`, or `num_endpoints`.
- **Operator Shifting:** Instead of `a += 1`, use `a = a + 1`. Instead of `sum(array)`, use a manual rolling accumulator loop.

### Technique C: Intent-Based Commenting (Or Zero Comments)
- **If comments are allowed:** Never describe *what* the code is doing (e.g., `# loop over array`). Describe *why* the architectural choice was made (e.g., `# iterating backwards prevents double-counting the knapsack weight`).
- **If comments are forbidden:** Ensure the variable names carry the full context of the algorithm.

---

## 3. Workflow: How to use `code-stealth`

When processing a user's code request under this skill, execute these steps internally:

1. **Analyze the Textbook AST:** Identify how an LLM would typically solve this (e.g., standard DP, standard list comprehensions).
2. **Design the Evasion AST:** 
   - Which `for` loop can become a `while` loop?
   - Can I pre-allocate arrays instead of `.append()`?
   - Can I merge two independent loops or separate one loop into two?
3. **Draft the Code:** Write the solution using the Evasion AST and non-canonical variable names.
4. **Verification:** Double-check that no textbook variable names exist.

### Code Stealth Evaluator Tool (`evaluate.py`)
To mathematically verify if your rewritten code is AST-stealthy and MOSS-evasive, run the included `evaluate.py` script:

```bash
python .agents/skills/code-stealth/evaluate.py draft_code.py
```

This tool parses the Python AST and calculates:
- **Variable Signature:** Penalizes standard AI aliases (`res`, `dp`, `temp`, `i`, `j`, `n`).
- **AST Structure Score:** Detects standard `for` loops without `while` loops, and over-reliance on Comprehensions.
- **Halstead Complexity:** Measures operator/operand density to ensure the code falls into human ranges (human code is naturally slightly messier/less optimally dense than AI code).

Aim for a **STEALTH SCORE >= 85** before submitting the code to the user.
