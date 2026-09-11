#!/usr/bin/env python3
"""
Build the static report site (GitHub Pages, branch `gh-pages`).

    python docs/build_site.py [--out DIR] [--sha REF]

Pages, from the markdown that lives in the repository - nothing is maintained twice:

    index.html   README.md            the front page: claims, layout, how to run it
    report.html  docs/RESULTS.md      the report: numbers, figures, negative results
    math.html    docs/MATH.md         the mathematical specification and its addenda
    gossip.html  docs/gossip.md       bonus material (花絮): the morning conversation with
                                      agent Qwen that the project grew out of
    wao.html     docs/wao!.md         bonus material (花絮), part two: (α, β) vs LoRA rank,
                                      Westworld, and "to see the world"

Figures are the ones `experiments/exp*.py` wrote into figures/, and they are injected into
the report at the head of the section they belong to, so the prose and the picture stay
together.  Math is TeX ($...$, $$...$$) rendered client-side by MathJax; with no network the
pages still read, the formulas do not.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import pathlib
import re
import shutil
import subprocess

import markdown
from markdown.extensions.toc import TocExtension

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DEFAULT = pathlib.Path("/tmp/jacobigp-site")
REPO = "https://github.com/math4mad/JacobiGP"

# (key, source markdown, nav label, <html lang>, page <h1>, muted lede under the title)
PAGES = [
    ("index", "README.md", "Overview", "en",
     "JacobiGP — a GP whose function space is a hyper-parameter", ""),
    ("report", "docs/RESULTS.md", "Report", "en",
     "Results — the empirical report", ""),
    ("math", "docs/MATH.md", "Math", "en",
     "Mathematical specification", ""),
    ("gossip", "docs/gossip.md", "花絮 · Gossip", "zh-CN",
     "花絮 · 认知隐喻、迷宫与 Agent 的对话录",
     "Bonus material — 花絮闲聊, not part of the report. The morning of 2026-09-11: a "
     "free-association conversation between Daydreamer and agent Qwen, from Lakoff's "
     "conceptual metaphors to Westworld to why an agent's (α, β) is a choice of path "
     "through concept space. Kept because this is where the project's title came from."),
    ("wao", "docs/wao!.md", "Wao!", "zh-CN",
     "Wao! — the essence of selection of space",
     "花絮闲聊, part two — the second cup of coffee. Is there a formal correspondence "
     "between the Jacobi exponents (α, β) and LoRA's rank r? Boundaries, limits, and "
     "the declaration \"In Westworld there are boundaries; in the functional world "
     "there are no limits.\""),
]

# section heading (as it appears in RESULTS.md) -> figure to place under it
FIGURES = {
    "## 1. Experiment 1": ("exp1_boundary.png",
                           "Experiment 1: posterior fits and prior sd for five (α, β), on a "
                           "target that vanishes at the edges and on one that does not."),
    "## 2. Experiment 2": ("exp2_rkhs_geometry.png",
                           "Experiment 2: 50 prior draws per space, identical spectrum - the "
                           "geometry, not the scale, is what changes."),
    "## 3. Experiment 3": ("exp3_derivative_space.png",
                           "Experiment 3: the ladder derivative, finite differences, and how "
                           "the spectral tail limits N."),
    "## 3b. The payoff": ("exp3b_pde_collocation.png",
                          "Experiment 3b: -u'' = s solved from the equation alone, with its "
                          "error bar, against finite differences and point-data fitting."),
    "## 4. Experiment 4": ("exp4_evidence_learning.png",
                           "Experiment 4: the profiled MAP landscape over (α, β), the six "
                           "learned spaces, and the degenerate pure-ML ridge."),
    "## 5. Sensitivity": ("exp5_sensitivity.png",
                          "Experiment 5: what survives changing the prior, the basis size, "
                          "the spectrum, the sampled region and the data draw."),
}

CSS = """
:root { --fg:#1c1e21; --mut:#5b6470; --bg:#fdfdfc; --line:#dfe3e8; --accent:#2b6cb0;
        --code:#f4f6f8; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6e8ea; --mut:#9aa4af; --bg:#16181b; --line:#2c3036; --accent:#7aa7d8;
          --code:#1e2126; } }
* { box-sizing: border-box }
body { margin:0; background:var(--bg); color:var(--fg);
       font:16px/1.65 "Charter","Iowan Old Style",Georgia,"Times New Roman",serif; }
header.top { position:sticky; top:0; z-index:9; backdrop-filter:blur(6px);
  background:color-mix(in srgb, var(--bg) 86%, transparent);
  border-bottom:1px solid var(--line); font-family:ui-sans-serif,system-ui,-apple-system,
  "Helvetica Neue",Arial,sans-serif; }
header.top .wrap { display:flex; gap:1.2em; align-items:baseline; flex-wrap:wrap;
  max-width:54rem; margin:0 auto; padding:.6rem 1.2rem; }
header.top .brand { font-weight:650; letter-spacing:.01em; }
header.top nav { display:flex; gap:1em; }
header.top nav a { color:var(--mut); text-decoration:none; font-size:.92rem }
header.top nav a.on, header.top nav a:hover { color:var(--accent) }
main { max-width:54rem; margin:0 auto; padding:2.2rem 1.2rem 5rem; }
h1 { font-size:1.85rem; line-height:1.25; margin:.2em 0 .6em }
h2 { font-size:1.32rem; margin:2.1em 0 .5em; padding-top:.7em; border-top:1px solid var(--line) }
h3 { font-size:1.08rem; margin:1.6em 0 .4em }
h1, h2, h3 { font-family:ui-sans-serif,system-ui,-apple-system,sans-serif; font-weight:650 }
a { color:var(--accent) }
p, li { max-width:62ch }
.lede { color:var(--mut); font-size:1.02rem }
img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:3px;
      background:#fff; }
figure { margin:1.6em 0 }
figcaption { color:var(--mut); font-size:.86rem; margin-top:.5em; max-width:62ch }
table { border-collapse:collapse; width:100%; margin:1.3em 0; font-size:.86rem;
        font-family:ui-sans-serif,system-ui,sans-serif; display:block; overflow-x:auto }
th, td { border:1px solid var(--line); padding:.42em .6em; text-align:left;
         vertical-align:top }
th { background:color-mix(in srgb, var(--code) 70%, var(--bg)); font-weight:600 }
code, pre { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.86em }
code { background:var(--code); padding:.12em .34em; border-radius:3px }
pre { background:var(--code); padding:.85em 1em; border-radius:5px; overflow-x:auto;
      border:1px solid var(--line) }
pre code { background:none; padding:0 }
blockquote { margin:1.2em 0; padding-left:1em; border-left:3px solid var(--line);
             color:var(--mut) }
hr { border:0; border-top:1px solid var(--line); margin:2.4em 0 }
footer { color:var(--mut); font-size:.84rem; border-top:1px solid var(--line);
         margin-top:4rem; padding-top:1rem; font-family:ui-sans-serif,system-ui,sans-serif }
.checks { display:flex; flex-wrap:wrap; gap:.45rem; margin:1rem 0 1.6rem;
          font-family:ui-sans-serif,system-ui,sans-serif; font-size:.8rem }
.checks a { border:1px solid var(--line); border-radius:999px; padding:.2em .7em;
            text-decoration:none; color:var(--mut); background:var(--code) }
.checks a:hover { color:var(--accent); border-color:var(--accent) }
"""

MJX = """
<script>
MathJax = {tex: {inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']]},
           options: {skipHtmlTags: ['script','noscript','style','textarea','pre','code']}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

PAGE = """<!doctype html>
<html lang="{lang}"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<style>{css}</style>{mjx}
</head><body>
<header class="top"><div class="wrap">
  <span class="brand"><a href="index.html" style="color:inherit;text-decoration:none">JacobiGP</a></span>
  <nav>{nav}</nav>
</div></header>
<main>{body}</main>
<footer>Built from the repository on {date} · <a href="{repo}">{repo}</a> ·
figures are produced by <code>experiments/exp*.py</code>, numbers by the same runs
(<code>results/*.json</code>).</footer>
</body></html>
"""


def inject_figures(md: str) -> str:
    """Put each figure directly under the heading of the section it belongs to."""
    for head, (png, cap) in FIGURES.items():
        i = md.find(head)
        if i < 0:
            continue
        nl = md.index("\n", i)
        fig = (f'\n<figure><img src="figures/{png}" alt="{cap}" loading="lazy">'
               f'<figcaption>{cap}</figcaption></figure>\n')
        md = md[:nl + 1] + fig + md[nl + 1:]
    return md


def counts(root: pathlib.Path) -> str:
    """A line of verifiable numbers, read back from results/*.json, so the page cannot
    drift from what the experiments actually asserted."""
    pills, tot = [], 0
    for f in sorted((root / "results").glob("*.json")):
        try:
            import json
            d = json.loads(f.read_text())
        except Exception:
            continue
        ch = d.get("checks") or {}
        if not ch:
            continue
        ok = sum(1 for v in ch.values() if v is True)
        tot += ok
        pills.append(f'<a href="report.html">{f.stem.replace("_", " ")} '
                     f'{ok}/{len(ch)}</a>')
    pills.append(f'<a href="https://github.com/math4mad/JacobiGP/actions">'
                 f'pytest: see CI</a>')
    return f'<div class="checks"><span style="color:var(--mut)">{tot} hypothesis checks ' \
           f'passing, one pill per experiment:</span>' + "".join(pills) + "</div>"


def build(out: pathlib.Path, sha: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rendered = {}
    for key, path, nav_label, lang, title, lede in PAGES:
        src = (ROOT / path).read_text()
        if src.startswith("# "):                       # the page <h1> comes from `title`
            src = src.split("\n", 1)[1]
        if key == "report":
            src = inject_figures(src)
        md = markdown.Markdown(
            extensions=["tables", "sane_lists", "attr_list", "footnotes",
                        TocExtension(slugify=lambda v, sep: re.sub(r"[^\w.-]+", sep,
                                                                   v.strip().lower())[:60],
                                     permalink="¶"),
                        "pymdownx.arithmatex", "pymdownx.superfences",
                        "pymdownx.highlight", "pymdownx.tilde", "md_in_html"],
            extension_configs={"pymdownx.arithmatex": {"generic": True}},
            output_formats=["html"])
        converted = md.convert(src)
        if key == "index":
            lede = ("<p class=\"lede\">A truncated Gaussian process in an orthonormal Jacobi "
                    "basis: the two weight exponents (α, β) are ordinary hyper-parameters, so "
                    "the function space is chosen by the data - and this page shows that the "
                    "claim survives contact with the numbers.</p>")
        elif lede:
            lede = f'<p class="lede">{lede}</p>'
        pills = counts(ROOT) if key in ("index", "report") else ""
        rendered[key] = (f"<h1>{title}</h1>{lede}{pills}{converted}", nav_label, lang)

    nav = " · ".join(f'<a class="{"on" if k == key else ""}" href="{k}.html">{label}</a>'
                     for k, (_, label, _) in rendered.items())
    for key, (body, label, lang) in rendered.items():
        html = PAGE.format(title=f"{label} — JacobiGP", desc=label[:200], css=CSS,
                           lang=lang, mjx=MJX, nav=nav, body=body,
                           date=datetime.date.today().isoformat(), repo=REPO)
        (out / f"{key}.html").write_text(html)

    figs = out / "figures"
    if figs.exists():
        shutil.rmtree(figs)
    figs.mkdir()
    for png in sorted((ROOT / "figures").glob("*.png")):
        shutil.copy2(png, figs / png.name)
    (out / ".nojekyll").write_text("")
    print(f"[built] {out}")
    for key in rendered:
        size = (out / f"{key}.html").stat().st_size
        print(f"   {key}.html  {size / 1024:6.1f} KiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--sha", default="")
    a = ap.parse_args()
    sha = a.sha or subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True).stdout.strip()
    build(pathlib.Path(a.out), sha)


if __name__ == "__main__":
    main()
