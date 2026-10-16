import ast
import io
import os
import re
import tokenize
from pathlib import Path


def strip_python_comments(source: str) -> str:
    tokens = list(tokenize.tokenize(io.BytesIO(source.encode("utf-8")).readline))
    lines = source.splitlines(keepends=True)
    
    comment_spans_by_line = {}
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            srow, scol = tok.start
            erow, ecol = tok.end
            comment_spans_by_line.setdefault(srow, []).append((scol, ecol))
            
    new_lines = []
    for line_idx, line in enumerate(lines, start=1):
        if line_idx in comment_spans_by_line:
            spans = sorted(comment_spans_by_line[line_idx], key=lambda x: x[0])
            first_comment_col = spans[0][0]
            code_part = line[:first_comment_col].rstrip()
            if code_part.strip():
                new_lines.append(code_part + "\n")
        else:
            new_lines.append(line)
            
    cleaned = []
    blank_count = 0
    for line in new_lines:
        if not line.strip():
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)
            
    return "".join(cleaned)


def condense_docstrings(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    doc_spans = []

    def get_first_line(doc: str) -> str:
        lines = [l.strip() for l in doc.strip().splitlines() if l.strip()]
        if not lines:
            return ""
        first = lines[0]
        first = re.sub(r"\s+", " ", first).strip()
        first = first.replace('"""', "'")
        return first

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if len(node.body) > 0 and isinstance(node.body[0], ast.Expr):
                val = node.body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    srow = node.body[0].lineno
                    erow = node.body[0].end_lineno
                    first_line = get_first_line(val.value)
                    if first_line:
                        doc_spans.append((srow, erow, first_line))

    if not doc_spans:
        return source

    doc_spans.sort(key=lambda x: x[0], reverse=True)
    lines = source.splitlines()

    for srow, erow, summary in doc_spans:
        idx_start = srow - 1
        idx_end = erow
        orig_line = lines[idx_start] if idx_start < len(lines) else ""
        m = re.match(r"^(\s*)", orig_line)
        orig_indent = m.group(1) if m else ""
        new_doc = f'{orig_indent}"""{summary}"""'
        lines[idx_start:idx_end] = [new_doc]

    return "\n".join(lines) + "\n"


def process_python_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False

    condensed = condense_docstrings(text)
    stripped = strip_python_comments(condensed)

    try:
        ast.parse(stripped)
    except SyntaxError as e:
        print(f"[ERR] Syntax error after comment stripping {path}: {e}")
        return False

    path.write_text(stripped, encoding="utf-8")
    return True


def strip_js_comments(text: str) -> str:
    result = []
    i = 0
    n = len(text)
    
    while i < n:
        if text[i] == '"' or text[i] == "'":
            quote = text[i]
            result.append(quote)
            i += 1
            while i < n:
                ch = text[i]
                result.append(ch)
                if ch == '\\':
                    i += 1
                    if i < n:
                        result.append(text[i])
                elif ch == quote:
                    break
                i += 1
            i += 1
        elif text[i] == '`':
            result.append('`')
            i += 1
            while i < n:
                ch = text[i]
                result.append(ch)
                if ch == '\\':
                    i += 1
                    if i < n:
                        result.append(text[i])
                elif ch == '`':
                    break
                i += 1
            i += 1
        elif text[i:i+3] == '{/*':
            i += 3
            while i < n and text[i:i+3] != '*/}':
                i += 1
            i += 3
        elif text[i:i+2] == '/*':
            i += 2
            while i < n and text[i:i+2] != '*/':
                i += 1
            i += 2
        elif text[i:i+2] == '//':
            i += 2
            while i < n and text[i] != '\n':
                i += 1
        else:
            result.append(text[i])
            i += 1
            
    out = "".join(result)
    lines = [re.sub(r"[ \t]+$", "", line) for line in out.splitlines()]
    
    cleaned = []
    blank_count = 0
    for l in lines:
        if not l.strip():
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(l)
        else:
            blank_count = 0
            cleaned.append(l)
            
    return "\n".join(cleaned) + "\n"


def process_js_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False

    stripped = strip_js_comments(text)
    path.write_text(stripped, encoding="utf-8")
    return True


def main():
    root = Path(__file__).resolve().parent.parent

    backend_dir = root / "backend"
    python_count = 0
    for py_file in backend_dir.rglob("*.py"):
        if process_python_file(py_file):
            python_count += 1

    print(f"[OK] Processed {python_count} Python files in backend/")

    frontend_src = root / "frontend" / "src"
    js_count = 0
    for ext in ("*.js", "*.jsx", "*.ts", "*.tsx", "*.css"):
        for js_file in frontend_src.rglob(ext):
            if process_js_file(js_file):
                js_count += 1

    print(f"[OK] Processed {js_count} frontend files in frontend/src/")


if __name__ == "__main__":
    main()
