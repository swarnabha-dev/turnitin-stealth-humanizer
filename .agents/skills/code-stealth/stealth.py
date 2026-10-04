import ast
import sys
import math
import hashlib

sys.stdout.reconfigure(encoding="utf-8")

AI_TELL_VARS = {"res", "result", "temp", "curr", "dp", "ans", "arr", "lst", "val", "idx", "i", "j", "n", "m", "length"}

class CodeEvaluator(ast.NodeVisitor):
    def __init__(self):
        self.var_names = []
        self.list_comps = 0
        self.dict_comps = 0
        self.for_loops = 0
        self.while_loops = 0
        self.operators = 0
        self.operands = 0
        self.unique_operators = set()
        self.unique_operands = set()
        self.structural_tokens = [] # For MOSS Winnowing

    def _record_structure(self, node_type):
        self.structural_tokens.append(node_type)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.var_names.append(node.id)
        self.operands += 1
        self.unique_operands.add(node.id)
        self._record_structure("VAR")
        self.generic_visit(node)
        
    def visit_Constant(self, node):
        self.operands += 1
        self.unique_operands.add(str(node.value))
        self._record_structure("CONST")
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self.list_comps += 1
        self._record_structure("LISTCOMP")
        self.generic_visit(node)

    def visit_For(self, node):
        self.for_loops += 1
        self._record_structure("FOR")
        self.generic_visit(node)

    def visit_While(self, node):
        self.while_loops += 1
        self._record_structure("WHILE")
        self.generic_visit(node)
        
    def visit_If(self, node):
        self._record_structure("IF")
        self.generic_visit(node)
        
    def visit_BinOp(self, node):
        self.operators += 1
        self.unique_operators.add(type(node.op).__name__)
        self._record_structure(f"BINOP_{type(node.op).__name__}")
        self.generic_visit(node)
        
    def visit_AugAssign(self, node):
        self.operators += 1
        self.unique_operators.add(type(node.op).__name__)
        self._record_structure(f"AUG_{type(node.op).__name__}")
        self.generic_visit(node)

def get_moss_winnowing_fingerprint(tokens, k=5, window_size=4):
    """
    SOTA MOSS-style Winnowing Algorithm Simulator.
    1. Generates k-grams of structural tokens.
    2. Hashes k-grams.
    3. Selects minimum hash in sliding windows to create a robust structural fingerprint.
    """
    if len(tokens) < k:
        return []
    
    # 1. Generate k-grams and hash them
    hashes = []
    for i in range(len(tokens) - k + 1):
        k_gram = "".join(tokens[i:i+k])
        h = int(hashlib.md5(k_gram.encode('utf-8')).hexdigest()[:8], 16)
        hashes.append(h)
        
    if len(hashes) < window_size:
        return hashes
        
    # 2. Winnowing (sliding window min hash)
    fingerprint = []
    for i in range(len(hashes) - window_size + 1):
        window = hashes[i:i+window_size]
        min_hash = min(window)
        if not fingerprint or fingerprint[-1] != min_hash:
            fingerprint.append(min_hash)
            
    return fingerprint

def evaluate_code(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Syntax Error: Cannot parse AST. {e}", file=sys.stderr)
        sys.exit(1)

    evaluator = CodeEvaluator()
    evaluator.visit(tree)

    # 1. Variable Analysis
    ai_vars_found = [v for v in evaluator.var_names if v.lower() in AI_TELL_VARS]
    var_score = max(0, 100 - (len(ai_vars_found) * 15))

    # 2. Structural/AST Analysis
    struct_score = 100
    if evaluator.list_comps > 0:
        struct_score -= (evaluator.list_comps * 10)
    if evaluator.for_loops > 0 and evaluator.while_loops == 0:
        struct_score -= 20

    struct_score = max(0, struct_score)

    # 3. Halstead Complexity
    N1 = evaluator.operators
    N2 = evaluator.operands
    n1 = len(evaluator.unique_operators)
    n2 = len(evaluator.unique_operands)
    
    vocabulary = n1 + n2
    length = N1 + N2
    volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
    difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
    effort = difficulty * volume

    halstead_score = min(100, max(0, int((difficulty / 15.0) * 100)))

    # 4. MOSS Fingerprint
    fingerprint = get_moss_winnowing_fingerprint(evaluator.structural_tokens)
    fingerprint_hex = [hex(h) for h in fingerprint[:5]] # show top 5

    overall_score = int((var_score * 0.4) + (struct_score * 0.4) + (halstead_score * 0.2))

    print("=" * 60)
    print(" 🛡️ NEXT-GEN CODE STEALTH ENGINE (MOSS + HALSTEAD) 🛡️")
    print("=" * 60)
    print(f"AI Variable Tells Found : {len(ai_vars_found)} {set(ai_vars_found)}")
    print(f"List/Dict Comprehensions: {evaluator.list_comps + evaluator.dict_comps}")
    print(f"For Loops               : {evaluator.for_loops}")
    print(f"While Loops             : {evaluator.while_loops}")
    print("-" * 60)
    print(f"Halstead Difficulty     : {difficulty:.2f}")
    print(f"Halstead Effort         : {effort:.2f}")
    print(f"MOSS Structural Hash    : {' '.join(fingerprint_hex)}...")
    print("-" * 60)
    print(f"Variables Score         : {var_score}/100")
    print(f"Structure Score         : {struct_score}/100")
    print(f"Halstead Score          : {halstead_score}/100")
    print("=" * 60)
    print(f"OVERALL STEALTH SCORE   : {overall_score}/100")
    
    if overall_score >= 85:
        print("Verdict: HIGHLY STEALTHY (AST Undetectable / MOSS Evaded)")
    elif overall_score >= 60:
        print("Verdict: MODERATE (May trigger structural flags)")
    else:
        print("Verdict: HIGH RISK (Canonical AI Structure Detected)")
        print("\n[!] SUGGESTED AST EVASION STRATEGY:")
        if evaluator.for_loops > 0:
            print("  -> Convert standard 'for' loops to 'while' loops.")
        if evaluator.list_comps > 0:
            print("  -> Flatten list comprehensions into manual append loops.")
        if ai_vars_found:
            print(f"  -> Rename variables: {set(ai_vars_found)} to domain-specific aliases.")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stealth.py <source_file.py>")
        sys.exit(1)
    evaluate_code(sys.argv[1])
