import re
from pathlib import Path

def audit_latex(path):
    print(f"Auditing {path.name}...")
    text = path.read_text(encoding="utf-8")
    
    # Stack check
    stack = []
    errors = []
    tokens = re.finditer(r"\\(begin|end)\{([a-zA-Z0-9_*]+)\}", text)
    for m in tokens:
        kind, env = m.group(1), m.group(2)
        line_no = text[:m.start()].count("\n") + 1
        if kind == "begin":
            stack.append((env, line_no))
        else:
            if not stack:
                errors.append(f"Unmatched \\end{{{env}}} at line {line_no}")
            else:
                last_env, start_line = stack.pop()
                if last_env != env:
                    errors.append(f"Mismatched env: \\begin{{{last_env}}} at line {start_line} closed by \\end{{{env}}} at line {line_no}")
    if stack:
        for env, start_line in stack:
            errors.append(f"Unclosed \\begin{{{env}}} from line {start_line}")
            
    # Check braces count (ignoring escaped \{ and \}, comments)
    cleaned = re.sub(r"\\\{|\\\}|%.*", "", text)
    open_b = cleaned.count("{")
    close_b = cleaned.count("}")
    print(f"  Environments check: {'FAILED' if errors else 'PASSED'}")
    print(f"  Brace balance: {open_b} open vs {close_b} close ({'BALANCED' if open_b == close_b else 'UNBALANCED'})")
    if errors:
        print("  ERRORS found:")
        for e in errors:
            print("   ", e)
    else:
        print("  All LaTeX environments properly closed and nested!")
    print()

if __name__ == "__main__":
    audit_latex(Path("research/manuscript.tex"))
    audit_latex(Path("research/supplementary_materials.tex"))
