from __future__ import annotations

from pathlib import Path
from string import Template

REPO_TOP_PATH = Path(__file__).resolve().parent.parent
PTX_TEMPLATE_PATH = REPO_TOP_PATH / "src" / "prompts" / "ptx_template.md"


def make_ptx_prompt(problem_src: str) -> str:
    """Fill the PTX prompt template with the given reference architecture source."""
    template_text = PTX_TEMPLATE_PATH.read_text()
    tmpl = Template(template_text)
    return tmpl.safe_substitute(PROBLEM_SRC=problem_src) 