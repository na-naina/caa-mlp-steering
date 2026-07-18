#!/usr/bin/env python3
"""Strip LaTeX markup from a paper .tex file to produce plain-prose output.

Keeps: prose, inline percentages, rough numbers.
Drops: preamble, all \\commands{}, equations, tables, figures,
       bibliography, comments, citations (replaced with [CITE]).
"""
import re
import sys
from pathlib import Path


def strip_latex(text: str) -> str:
    # Drop everything before \begin{document}
    m = re.search(r"\\begin\{document\}", text)
    if m:
        text = text[m.end():]
    # Drop everything after \end{document}
    m = re.search(r"\\end\{document\}", text)
    if m:
        text = text[:m.start()]

    # Drop comments (% to end of line)
    text = re.sub(r"(?m)^%.*$", "", text)
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)

    # Drop figure, table, table*, figure* environments entirely (including caption text)
    for env in ["figure", "figure*", "table", "table*", "tabular", "equation",
                "align", "align*", "itemize", "enumerate"]:
        text = re.sub(
            rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}",
            "",
            text,
            flags=re.DOTALL,
        )

    # Strip environments that should be unwrapped (keep content): quote, center, abstract-like
    for env in ["quote", "center"]:
        text = re.sub(rf"\\begin\{{{env}\}}", "", text)
        text = re.sub(rf"\\end\{{{env}\}}", "", text)

    # Replace citations with [CITE]
    text = re.sub(r"\\cite[a-z]*\{[^}]+\}", "[CITE]", text)

    # Drop \label{...} and \ref{...}
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    text = re.sub(r"\\ref\{[^}]+\}", "§X", text)
    text = re.sub(r"\\S\{[^}]+\}", "§", text)
    text = re.sub(r"\\S~?", "§", text)

    # Headers: extract the title text, keep as uppercase line
    def _header(match):
        return "\n\n" + match.group(2).upper() + "\n\n"

    text = re.sub(r"\\(section|subsection|paragraph)\*?\{([^}]+)\}", _header, text)
    text = re.sub(r"\\chapter\*?\{([^}]+)\}", lambda m: "\n\n" + m.group(1).upper() + "\n\n", text)
    text = re.sub(r"\\title\{([^}]+)\}", lambda m: m.group(1).upper() + "\n\n", text)
    # Remove \rule{...}{...} decorations
    text = re.sub(r"\\rule\{[^}]+\}\{[^}]+\}", "", text)

    # Inline math: keep the content, remove $...$
    text = re.sub(r"\$([^$]+)\$", lambda m: _math(m.group(1)), text)

    # Common text-formatting commands: keep the content
    for cmd in ["textbf", "textit", "emph", "textsc", "texttt"]:
        text = re.sub(rf"\\{cmd}\{{([^{{}}]+)\}}", r"\1", text)
    # repeat in case of nesting
    for cmd in ["textbf", "textit", "emph", "textsc", "texttt"]:
        text = re.sub(rf"\\{cmd}\{{([^{{}}]+)\}}", r"\1", text)

    # Abstract environment
    text = re.sub(r"\\begin\{abstract\}", "\n\nABSTRACT\n\n", text)
    text = re.sub(r"\\end\{abstract\}", "\n\n", text)

    # Standalone commands
    text = re.sub(r"\\maketitle", "", text)
    text = re.sub(r"\\appendix", "\n\nAPPENDIX\n\n", text)
    text = re.sub(r"\\bibliography\{[^}]+\}", "", text)
    text = re.sub(r"\\author\{[^}]+\}", "", text)

    # Generic remaining \command{arg} -> arg (for things like \S)
    text = re.sub(r"\\[a-zA-Z]+\*?\{([^{}]*)\}", r"\1", text)
    # Remaining bare \command -> drop
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)

    # Escaped specials
    text = text.replace(r"\%", "%")
    text = text.replace(r"\&", "&")
    text = text.replace(r"\$", "$")
    text = text.replace(r"\_", "_")
    text = text.replace("---", "—")
    text = text.replace("--", "–")
    text = text.replace("``", '"').replace("''", '"')

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _math(expr: str) -> str:
    """Very light mathmode → plain text."""
    expr = expr.replace(r"\times", "x")
    expr = expr.replace(r"\alpha", "α")
    expr = expr.replace(r"\lambda", "λ")
    expr = expr.replace(r"\theta", "θ")
    expr = expr.replace(r"\pm", "±")
    expr = expr.replace(r"\sim", "~")
    expr = re.sub(r"\\mathbf\{([^}]+)\}", r"\1", expr)
    expr = re.sub(r"\\mathbb\{([^}]+)\}", r"\1", expr)
    expr = re.sub(r"\\text\{([^}]+)\}", r"\1", expr)
    expr = re.sub(r"_\{?([^{}]+?)\}?", r"_\1", expr)
    expr = re.sub(r"\^\{?([^{}]+?)\}?", r"^\1", expr)
    expr = expr.replace("{", "").replace("}", "").replace("\\", "")
    return expr


if __name__ == "__main__":
    path = Path(sys.argv[1])
    out = strip_latex(path.read_text())
    print(out)
