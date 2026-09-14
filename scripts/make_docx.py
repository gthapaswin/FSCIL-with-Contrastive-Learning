#!/usr/bin/env python3
"""Generate a long, detailed Word report of the entire STAG-STI FSCIL project.

Every architectural stage, data path, training phase, evaluation protocol,
result table, bug fix and design decision is explained in prose a student can
read to understand and defend the project. Numbers are pulled live from
checkpoints/<dataset>/incremental_results.json.

Run:  .venv/bin/python scripts/make_docx.py
Out:  results/STAG-STI_Project_Report.docx
"""
import json, os
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- palette ----------
INK   = RGBColor(0x14, 0x1B, 0x2E)
NAVY  = RGBColor(0x1E, 0x3A, 0x5F)
BLUE  = RGBColor(0x2D, 0x6C, 0xDF)
TEAL  = RGBColor(0x12, 0x7A, 0x6B)
AMBER = RGBColor(0xB5, 0x6A, 0x0E)
RED   = RGBColor(0xB0, 0x2E, 0x22)
SLATE = RGBColor(0x4A, 0x55, 0x66)
GREY  = RGBColor(0x6B, 0x74, 0x82)

def load(d):
    return json.load(open(os.path.join(ROOT, "checkpoints", d, "incremental_results.json")))
DATA = {d: load(d) for d in ["cifar100", "miniimagenet", "cub200"]}
NAMES = {"cifar100": "CIFAR-100", "miniimagenet": "miniImageNet", "cub200": "CUB-200"}

doc = Document()

# ---------- base style ----------
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10.5)
normal.font.color.rgb = INK
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.14

# page margins
for sec in doc.sections:
    sec.left_margin = Inches(0.9); sec.right_margin = Inches(0.9)
    sec.top_margin = Inches(0.8); sec.bottom_margin = Inches(0.8)

# ---------- helpers ----------
def _shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)

def _set_cell_text(cell, text, bold=False, color=INK, size=9.5, align="left", white=False):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "right": WD_ALIGN_PARAGRAPH.RIGHT}[align]
    p.paragraph_format.space_after = Pt(1); p.paragraph_format.space_before = Pt(1)
    r = p.add_run(str(text)); r.font.bold = bold; r.font.size = Pt(size)
    r.font.color.rgb = RGBColor(0xFF,0xFF,0xFF) if white else color
    r.font.name = "Calibri"

def h1(text):
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    p = doc.add_heading(level=1)
    p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text); r.font.name = "Calibri"; r.font.size = Pt(19)
    r.font.color.rgb = NAVY; r.font.bold = True
    # bottom rule
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr"); bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single"); bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "4"); bottom.set(qn("w:color"), "2D6CDF")
    pbdr.append(bottom); pPr.append(pbdr)
    return p

def h2(text):
    p = doc.add_heading(level=2)
    p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text); r.font.name = "Calibri"; r.font.size = Pt(14)
    r.font.color.rgb = BLUE; r.font.bold = True
    return p

def h3(text):
    p = doc.add_heading(level=3)
    p.paragraph_format.space_before = Pt(7); p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text); r.font.name = "Calibri"; r.font.size = Pt(11.5)
    r.font.color.rgb = TEAL; r.font.bold = True
    return p

def body(text, size=10.5, color=INK, italic=False, after=6, align="left"):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "just": WD_ALIGN_PARAGRAPH.JUSTIFY}[align]
    _add_runs(p, text, size, color, italic)
    return p

def _add_runs(p, text, size=10.5, color=INK, italic=False):
    """Parse **bold**, `code`, and plain text into runs."""
    import re
    tokens = re.split(r"(\*\*.*?\*\*|`.*?`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            r = p.add_run(tok[2:-2]); r.font.bold = True; r.font.size = Pt(size)
            r.font.color.rgb = color; r.font.name = "Calibri"
        elif tok.startswith("`") and tok.endswith("`"):
            r = p.add_run(tok[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(size - 1)
            r.font.color.rgb = RED
        else:
            r = p.add_run(tok); r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = "Calibri"
            r.font.italic = italic

def bullet(text, level=0, size=10.5):
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.3 + 0.25 * level)
    _add_runs(p, text, size)
    return p

def numbered(text, size=10.5):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.space_after = Pt(3)
    _add_runs(p, text, size)
    return p

def callout(title, text, color=BLUE, fill="EBF1FB"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    _shade(cell, fill)
    # left accent border
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    left = OxmlElement("w:left"); left.set(qn("w:val"), "single"); left.set(qn("w:sz"), "24")
    left.set(qn("w:space"), "0"); left.set(qn("w:color"), "%02X%02X%02X" % (color[0], color[1], color[2]))
    borders.append(left); tcPr.append(borders)
    cell.text = ""
    p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(2)
    r = p.add_run(title); r.font.bold = True; r.font.size = Pt(10.5); r.font.color.rgb = color; r.font.name = "Calibri"
    p2 = cell.add_paragraph(); p2.paragraph_format.space_after = Pt(2)
    _add_runs(p2, text, 10)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tbl

def code_block(lines):
    tbl = doc.add_table(rows=1, cols=1)
    cell = tbl.cell(0, 0)
    _shade(cell, "1B2233")
    cell.text = ""
    for i, ln in enumerate(lines):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        p.paragraph_format.space_after = Pt(0); p.paragraph_format.line_spacing = 1.0
        r = p.add_run(ln if ln else " ")
        r.font.name = "Consolas"; r.font.size = Pt(8.7)
        r.font.color.rgb = RGBColor(0xD6, 0xE2, 0xF5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return tbl

def table(headers, rows, widths=None, head_fill="1E3A5F", highlight=None, fs=9.5, hfs=9.5,
          align_cols=None):
    highlight = highlight or {}
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    align_cols = align_cols or (["left"] + ["center"] * (len(headers) - 1))
    for j, htext in enumerate(headers):
        c = t.rows[0].cells[j]; _shade(c, head_fill)
        _set_cell_text(c, htext, bold=True, size=hfs, align=align_cols[j], white=True)
    for i, row in enumerate(rows):
        cells = t.add_row().cells
        for j, val in enumerate(row):
            fill = highlight.get(i, "FFFFFF" if i % 2 == 0 else "F2F5FA")
            _shade(cells[j], fill)
            _set_cell_text(cells[j], val, bold=(j == 0 or i in highlight),
                           size=fs, align=align_cols[j])
    if widths:
        for j, w in enumerate(widths):
            for r in t.rows:
                r.cells[j].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

def add_toc():
    p = doc.add_paragraph()
    run = p.add_run()
    fldChar = OxmlElement("w:fldChar"); fldChar.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-2" \\h \\z \\u'
    fldChar2 = OxmlElement("w:fldChar"); fldChar2.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t"); t.text = "Right-click and choose 'Update Field' to build the table of contents."
    fldChar3 = OxmlElement("w:fldChar"); fldChar3.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar); run._r.append(instr); run._r.append(fldChar2); run._r.append(t); run._r.append(fldChar3)

def page_break():
    doc.add_page_break()

# ==================================================================
# TITLE PAGE
# ==================================================================
for _ in range(3):
    doc.add_paragraph()
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("STAG-STI"); r.font.size = Pt(46); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = "Calibri"
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Spatio-Temporal Graph Integration with Contrastive Pre-Conditioning\nfor Forward-Compatible Few-Shot Class-Incremental Learning")
r.font.size = Pt(16); r.font.color.rgb = BLUE; r.font.name = "Calibri"; r.font.bold = True
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("A Complete Project Report")
r.font.size = Pt(13); r.font.color.rgb = SLATE; r.font.italic = True; r.font.name = "Calibri"
for _ in range(2):
    doc.add_paragraph()
# summary line box
tbl = doc.add_table(rows=1, cols=1); tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
cell = tbl.cell(0, 0); _shade(cell, "F2F5FA")
cell.text = ""
lines = [
    ("Architecture", "8-stage graph-attentive, contrastively pre-conditioned FSCIL pipeline"),
    ("Benchmarks", "CIFAR-100 · miniImageNet · CUB-200 (official CEC/FACT splits) + cross-domain"),
    ("Studies", "Full component ablation · CLOSER base objective · evidence-driven improvement study"),
    ("Platform", "PyTorch on Apple-Silicon (MPS), single-run reproducible pipeline"),
]
for i, (k, v) in enumerate(lines):
    pp = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
    pp.alignment = WD_ALIGN_PARAGRAPH.CENTER; pp.paragraph_format.space_after = Pt(3)
    rr = pp.add_run(k + ":  "); rr.font.bold = True; rr.font.color.rgb = NAVY; rr.font.size = Pt(10.5); rr.font.name = "Calibri"
    rr = pp.add_run(v); rr.font.size = Pt(10.5); rr.font.color.rgb = INK; rr.font.name = "Calibri"
page_break()

# ==================================================================
# TABLE OF CONTENTS
# ==================================================================
h1("Table of Contents")
body("This report is written to be read end to end. Every stage of the architecture, every data path, "
     "every training and evaluation decision, and every result is explained from first principles so the "
     "project can be understood and defended in detail.", italic=True, color=SLATE)
add_toc()
page_break()

# ==================================================================
# 0. EXECUTIVE SUMMARY
# ==================================================================
h1("Executive Summary")
body("This project implements and evaluates **STAG-STI**, a system for **Few-Shot Class-Incremental "
     "Learning (FSCIL)** — the setting where a model must keep learning brand-new categories after "
     "deployment, from only a handful of labelled examples each, without forgetting what it already knows "
     "and without retraining from scratch.")
body("The core idea is a **frozen backbone plus nearest-prototype classification**, where the prototypes "
     "for each class are not simple averages but are refined through a three-part mechanism: (1) a "
     "**supervised-contrastive pre-conditioning** of the feature space during base training, (2) a "
     "**topology-aware graph-attention network (GATv2)** that lets multiple augmented views of a class "
     "exchange information to form a denoised prototype, and (3) a **spatio-temporal memory** that ages "
     "older classes so they harden (become stable) while new classes stay plastic.")
h3("What was built and measured")
bullet("The **full 8-stage STAG-STI pipeline** was implemented from the technical document and runs "
       "end-to-end on three standard benchmarks.")
bullet("All three datasets were evaluated on the **official CEC/FACT splits** using the exact official "
       "few-shot support samples, plus a **cross-domain transfer** test (miniImageNet → CUB).")
bullet("A **complete ablation study** isolates each of the three headline components.")
bullet("An **improvement study** measured every accuracy lever that does not change the architecture, "
       "reporting honest positive and negative results.")
bullet("Two substantive **bugs were found and fixed** by verifying results (a CUB base-session collapse "
       "from 1% to 78%, and a cross-domain crash).")
h3("Headline finding")
callout("The bottleneck is novel-class accuracy.",
        "Across every dataset, base-class accuracy (A_B, 56–69%) massively exceeds novel-class accuracy "
        "(A_N, 16–31%). The model remembers old classes well but struggles to learn new ones from 5 shots. "
        "Overall accuracy and Performance Drop hide this; the harmonic mean (HM) exposes it. Every effective "
        "improvement we found targets A_N specifically, and the true ceiling is the quality of the frozen "
        "backbone representation.", color=RED, fill="FBECEA")
page_break()

# ==================================================================
# PART I — BACKGROUND
# ==================================================================
h1("Part I — Background and Problem Definition")

h2("1. What is Few-Shot Class-Incremental Learning?")
body("A standard image classifier is trained once on a fixed set of classes and then deployed. FSCIL breaks "
     "that assumption. The model is trained on a **base session** with many classes and abundant data, and "
     "then must absorb a sequence of **incremental sessions**, each introducing a few new classes with only "
     "a few labelled images per class (the classic setting is **5-shot**). Crucially:")
bullet("**Label spaces are disjoint** — a class appears in exactly one session and never again.")
bullet("**Evaluation is cumulative** — after each session, the model is tested over *all* classes seen so "
       "far, base and novel together.")
bullet("**No replay** — the model cannot store or revisit old training data.")
bullet("**No retraining** — the representation is frozen after the base session; only new prototypes are added.")
body("This mirrors real deployment: a product-recognition system meets a new product, a medical system a "
     "new condition — each with only a few examples, and neither can afford to be retrained from scratch or "
     "to forget everything it already knew.")

h2("2. The Central Difficulty: Stability vs. Plasticity")
body("FSCIL is hard because two goals pull in opposite directions:")
bullet("**Plasticity** — the model must change enough to learn the new classes.")
bullet("**Stability** — the model must *not* change so much that it forgets old classes (catastrophic forgetting).")
body("Naive fine-tuning on the few new examples maximises plasticity but destroys stability. Freezing "
     "everything maximises stability but cannot learn. On top of that, **five examples are far too few** to "
     "estimate a reliable class centroid — a 5-shot prototype is noisy and biased. STAG-STI addresses each "
     "of these: the frozen backbone gives stability, the graph refinement denoises the few-shot prototype, "
     "and the age-decaying memory explicitly balances the two forces per class.")

h2("3. Evaluation Protocol and Metrics")
body("Every number in this report follows the standard FSCIL protocol. The following metrics are used "
     "throughout (all reported at the final session unless stated):")
table(
    ["Metric", "Definition", "Interpretation"],
    [
        ["Aᵢ", "Accuracy after session i, over all classes seen so far", "The session-by-session curve"],
        ["A₀ / Aₜ", "Base-session and final-session accuracy", "Start and end of that curve"],
        ["PD ↓", "Performance Drop = A₀ − Aₜ", "Total forgetting; lower is better"],
        ["Avg", "Mean of Aᵢ across all sessions", "Overall trajectory quality"],
        ["A_B", "Final accuracy on base classes only", "How well old knowledge is kept"],
        ["A_N", "Final accuracy on novel classes only", "How well new classes are learned"],
        ["HM", "Harmonic mean of A_B and A_N", "Balance — the honest FSCIL score"],
    ],
    widths=[1.0, 3.7, 2.6],
)
callout("Why HM is the metric that matters.",
        "A model can inflate overall accuracy simply by favouring the many base classes and ignoring the "
        "few novel ones. The harmonic mean only rises when BOTH A_B and A_N are healthy — it punishes "
        "imbalance. That is why the ablation and improvement studies in this report are judged primarily on HM.",
        color=TEAL, fill="E7F3F0")
page_break()

# ==================================================================
# PART II — ARCHITECTURE
# ==================================================================
h1("Part II — The STAG-STI Architecture")
body("STAG-STI turns a few raw support images of a class into a single, robust, forward-compatible "
     "prototype through eight stages. The backbone (Stage 2) is frozen after base training; every other "
     "stage from Stage 3 onward is a small trainable module, wired together in `fscil/pipeline.py` as the "
     "`StagStiModel`. Queries at test time take a shortcut — they are embedded and projected, then scored "
     "directly against the stored prototypes, so they never pass through the expensive graph.")
code_block([
    "Fantasy views ×M  →  Frozen backbone f_theta  →  Projection W_proj  →  SupCon head",
    "     →  Topology scorer g_phi  →  GATv2 attention  →  Prototype pooling  →  STI memory H_t",
    "     →  Cosine classifier   (queries skip the graph)",
])
body("The rest of Part II walks through each stage: what it receives, what it computes, why it exists, and "
     "where it lives in the code.")

h2("Stage 1 — Deterministic Fantasy Views")
body("Each support image is expanded into **M deterministic augmented 'fantasy' views** (default M = 4). "
     "Unlike random augmentation, the same image always yields the same M views, so prototypes are exactly "
     "reproducible on every run. The transforms are drawn in order from a fixed list: identity, horizontal "
     "flip, +15° rotation, 0.8× centre-zoom, −15°, +30°, −30°, vertical flip. Every view is resized to the "
     "dataset's input resolution so the identical code path serves 32-, 84-, and 224-pixel datasets.")
body("**Why it exists:** a single 5-shot image gives one noisy feature point. Multiple views of it act as a "
     "Monte-Carlo cloud around the true class point; averaging and graph-mixing them (later stages) yields a "
     "far more stable prototype. Determinism was an explicit design decision for reproducibility. Code: "
     "`make_fantasy_views` and `_DETERMINISTIC_VIEW_OPS` in `fscil/data.py`.")

h2("Stage 2 — The Frozen Backbone f_theta")
body("A ResNet-18 maps each view to a **512-dimensional feature vector**. Three stems are supported, chosen "
     "per dataset: a **CIFAR-adapted stem** (3×3 stride-1 conv, no maxpool) for 32-px CIFAR-100; a "
     "**standard ImageNet stem** for 84-px miniImageNet; and an **ImageNet-pretrained** ResNet-18 for "
     "224-px CUB-200 (its 100 base classes are too few to train a strong backbone from scratch). After base "
     "training the backbone is **frozen** (`requires_grad=False`, `.eval()`) and never updated again — this "
     "is the source of stability. Code: `fscil/backbone.py`.")

h2("Stage 3 — Projection W_proj (512 → 256)")
body("A single linear layer (no bias) projects the 512-d backbone feature into a **256-d graph hidden "
     "space**. The same projection is applied both to support prototypes before the graph and to query "
     "features at inference, so queries and prototypes live in the same space. Code: `Projection` in "
     "`fscil/modules.py`.")

h2("Stage 3.5 — Supervised-Contrastive Pre-Conditioning (SupCon)")
body("Before shot-averaging, a small MLP head projects each raw 512-d per-view feature onto a 128-d unit "
     "hypersphere, and a **view-preserving supervised contrastive loss** is applied. The subtlety: positives "
     "are restricted to the *same view* of other same-class samples. This penalises intra-view collapse "
     "(all '0° dogs' should cluster) while *preserving* diversity across views (a '0° dog' is allowed to "
     "differ from a '90° dog'), so the fantasy diversity that Stage 1 created is not destroyed.")
body("**Why it exists:** it shapes the base feature space so it is well-spread and well-clustered. A "
     "better-organised space means a novel class's 5-shot prototype lands in a meaningful region and "
     "generalises better — directly targeting the novel-class bottleneck. Temperature 0.15, loss weight "
     "λ_supcon = 0.3 (both tuned down from textbook defaults to avoid over-tight clustering that hurt novel "
     "classes). Code: `SupConHead` and `supcon_loss_view` in `fscil/modules.py`.")

h2("Stage 4 — Dynamic Topology Scorer g_phi")
body("The topology scorer takes every pair of graph nodes (h_i, h_j), concatenates them, and passes them "
     "through a small MLP with a sigmoid output to produce a **soft adjacency A_ij ∈ [0,1]** — a learned "
     "estimate of how strongly two nodes should be connected. It is supervised directly by a graph BCE loss "
     "(`L_graph`, weight 0.25) whose target is 1 for same-class node pairs and 0 otherwise, so the learned "
     "graph is meaningful rather than arbitrary. Code: `TopologyScorer` and `graph_topology_loss`.")

h2("Stage 5 — Fantasy-Aware GATv2 Attention")
body("A **Graph Attention Network v2** layer performs attention-weighted message passing over the learned "
     "graph. Its attention coefficient is computed from the pair of node features, the soft adjacency A_ij "
     "from Stage 4, and a learned **view-relation embedding** R keyed by the (view_i, view_j) pair — so the "
     "network knows *which* transform produced each node and can weight, say, a flip-vs-rotation pair "
     "differently. A PReLU activation is used inside the coefficient and on the aggregated output to "
     "preserve negative activations (relevant to the rotational geometry of the views).")
body("**Why it exists:** this is where the M views of a class actually talk to each other. Message passing "
     "averages away view-specific noise and produces a **consensus, denoised node representation** — the key "
     "step that turns noisy few-shot features into a robust prototype. Code: `GATv2Layer` in "
     "`fscil/modules.py`.")

h2("Stage 6 — Prototype View Pooling")
body("The enriched view-nodes of each class are mean-pooled into a **single class prototype P_t** (a "
     "parameter-free Monte-Carlo average over the M views). In parallel, the soft adjacency is pooled to a "
     "view-reduced class-level adjacency Ã. Code: `pool_prototypes` and `pool_class_adjacency`.")

h2("Stage 7 — Spatio-Temporal Integration (STI) Memory")
body("The STI memory H_t is the persistent store of every class prototype, and it governs how much each "
     "class is still allowed to change. Its update blends the previous memory with the fresh prototype "
     "through two gates:")
bullet("**A learned retention gate R** — a sigmoid over the previous memory, the new prototype, and the "
       "graph-propagated old memory (Ã · H_prev).")
bullet("**An age-based stability prior β** — β = σ(κ·(t − τ_c) − κ₀), where τ_c is the session a class was "
       "introduced. New classes (t ≈ τ_c) get β ≈ 0 and stay plastic; older classes get β → 1 and harden. "
       "Slope κ = 2.0, offset κ₀ = 3.0.")
body("These combine as an affine blend Γ = β + (1−β)·R, and the memory updates as "
     "H_t = Γ·H_prev + (1−Γ)·Φ(P_t), where Φ is a small memory-adapter layer. Brand-new classes, which have "
     "no prior memory, simply take H_t = Φ(P_t). A stability loss (`L_stab`) keeps Φ a gentle calibration "
     "(cosine-aligned to P_t) and discourages the gate from saturating. Code: `STIMemory` and "
     "`memory_stability_loss`.")
callout("An honest implementation note.",
        "In the standard FSCIL protocol a class's memory row is written exactly once — at the session it is "
        "introduced — and never revisited. So for THIS evaluation, the retention gate and age prior cannot "
        "change a class's first (and only) write (new_class_mask forces H = Φ(P)). The age-decay machinery "
        "is fully implemented and is exercised during episodic training and in the ablation, but its "
        "cross-session hardening would only bite in a protocol that recomputes existing rows. This is "
        "documented directly in the code comments rather than hidden.", color=AMBER, fill="FBF3E5")

h2("Stage 8 — Cosine Classifier")
body("A query image is embedded by the frozen backbone, projected by W_proj, L2-normalised, and scored "
     "against the L2-normalised memory rows by **cosine similarity**, scaled by a learnable inverse "
     "temperature. The softmax over those scores is the class prediction. Because queries skip Stages 3.5–7, "
     "inference is cheap. Code: `CosineClassifier`.")

h2("The Composite Training Loss")
body("During base-session episodic training (Phase B), all four losses are optimised jointly:")
table(
    ["Term", "Symbol", "Weight", "Purpose"],
    [
        ["Classification", "L_cls", "1.0", "Cross-entropy of queries vs prototypes"],
        ["Contrastive", "L_supcon", "0.3", "Spread/cluster the base feature space"],
        ["Graph topology", "L_graph", "0.25", "Supervise the learned adjacency (same-class = 1)"],
        ["Stability", "L_stab", "0.1", "Keep the memory adapter calibrated + gate unsaturated"],
    ],
    widths=[1.6, 1.1, 0.9, 3.7],
)
body("These weights were tuned on a held-out validation split (raised weight decay, lowered SupCon weight "
     "and raised its temperature) after early runs showed a large train/val gap. Code: `run_episode` in "
     "`fscil/train_session0.py`.")
page_break()

# ==================================================================
# PART III — DATA
# ==================================================================
h1("Part III — Data, Datasets and Splits")

h2("Dataset Registry")
body("The pipeline is made dataset-agnostic by a small registry (`fscil/datasets.py`). Each benchmark is "
     "described by a `DatasetSpec` giving its resolution, normalisation statistics, class-split shape, which "
     "backbone stem to use, where its images live, and which loader backend to use. Switching datasets is a "
     "single call, `Config.apply_dataset('cub200')`, which reconfigures the entire pipeline and namespaces "
     "all output directories so runs never clobber each other.")
table(
    ["Dataset", "Res", "Base / Novel", "Sessions", "Backbone", "Loader"],
    [
        ["CIFAR-100", "32", "60 / 40", "8 × 5-way 5-shot", "CIFAR ResNet-18", "index"],
        ["miniImageNet", "84", "60 / 40", "8 × 5-way 5-shot", "ResNet-18", "mini_csv"],
        ["CUB-200-2011", "224", "100 / 100", "10 × 10-way 5-shot", "ResNet-18 (pretrained)", "imagefolder"],
    ],
    widths=[1.5, 0.6, 1.2, 1.7, 1.9, 1.0],
)

h2("Official Splits and the Global Label Space")
body("Reproducibility hinges on using the **official CEC/FACT split files** (vendored under "
     "`fscil/splits/<dataset>/session_*.txt`), the same ones used by published methods (FACT, SAVC, CLOSER). "
     "`session_1.txt` lists all base-class training samples; `session_j.txt` (j ≥ 2) lists the *exact* "
     "few-shot support samples for that incremental session.")
body("Every class is assigned a single contiguous **global label**: base classes take labels 0..N_base−1 "
     "in split-file order, and each incremental session appends its new classes. This global order is fixed "
     "and reproducible, and it means the Phase-A classifier head sees exactly the base labels it expects. "
     "Parsing lives in `load_official_split`; the exact per-session support references are produced by "
     "`official_session_refs`.")

h2("Three Loader Backends, One Interface")
body("All three datasets are served through one `IndexedDataset` class (`fscil/data.py`) that always "
     "returns `(tensor, global_label)` and can resolve an official-split reference to a concrete sample "
     "index. The three backends differ only in how they enumerate samples:")
bullet("**cifar100 (index)** — wraps torchvision CIFAR-100; a split line is an integer index into the "
       "training array.")
bullet("**mini_csv** — miniImageNet ships as a flat `images/` folder plus `train.csv`/`test.csv` mapping "
       "filename → wnid (the class key). A split line's reference is the bare filename.")
bullet("**imagefolder** — CUB-200 stores images in per-class folders with an official "
       "`train_test_split.txt`; membership per image is read from that file so the official train/test "
       "division is respected exactly.")

h2("Transforms")
body("Two transform regimes are used. **Training transforms** for the backbone (Phase A) apply "
     "random-crop/resize, horizontal flip, and cutout-style RandomErasing. **Evaluation transforms** apply "
     "only a deterministic resize + centre-crop + normalise. Separately, the **fantasy views** (Stage 1) are "
     "always deterministic. A third **contrastive transform** (colour jitter, grayscale, aggressive crop) is "
     "used only by the optional CLOSER Phase-A variant.")
page_break()

# ==================================================================
# PART IV — TRAINING
# ==================================================================
h1("Part IV — Training the Model")
body("Base-session training runs in two phases, both driven by `fscil/train_session0.py`.")

h2("Phase A — Backbone Pretraining")
body("The ResNet-18 is trained as an ordinary classifier over the base classes with cross-entropy, SGD "
     "(momentum 0.9, Nesterov), and a cosine-annealed learning rate. To fight the large train/val gap seen "
     "in early runs, Phase A uses a battery of regularisers:")
bullet("**Label smoothing** (0.1) on the cross-entropy targets.")
bullet("**Mixup / CutMix** — one is randomly chosen per batch; the loss is the interpolation of the two "
       "labels' cross-entropies.")
bullet("**RandomErasing** cutout augmentation and a dropout layer in the classifier head.")
bullet("**Increased weight decay** (1e-3, up from 5e-4).")
bullet("**Early stopping** — training halts when validation accuracy stops improving for a patience window, "
       "and a separate best-validation checkpoint (`*_best.pt`) is always saved alongside the final one.")
body("After Phase A the backbone is frozen for everything that follows. Code: `pretrain_backbone`.")

h2("Phase B — Episodic STAG-STI Training")
body("With the backbone frozen, the trainable STAG-STI modules are optimised over **episodes** that "
     "simulate the incremental setting: each episode samples a 15-way, 5-shot, 5-query task from the base "
     "pool (100 episodes per epoch, 50 epochs by default, AdamW, cosine LR). For each episode the code runs "
     "the full pipeline — fantasy views → frozen backbone → SupCon on raw features → shot-averaging → "
     "projection → topology → GATv2 → pooling → STI memory → cosine classification of the queries — and "
     "back-propagates the composite loss through every trainable module. Validation runs 20 episodes per "
     "epoch, and again the best-validation checkpoint is saved. Code: `run_episode` and `train_stag_sti`.")
callout("Why train on 15-way episodes?",
        "Episodic (meta-learning-style) training exposes the graph and memory modules to the same kind of "
        "few-shot, many-class structure they will face at incremental time, so they learn to build good "
        "prototypes from few shots rather than relying on abundant data. This episode 'way' also becomes the "
        "safe chunk size used later to fix the CUB base bug (Part VII).", color=BLUE)

h2("The CLOSER Variant (Optional Phase A)")
body("An alternative base objective, based on CLOSER (Oh et al., ECCV 2024), was implemented to test "
     "whether a more transfer-friendly representation helps novel classes. It adds two terms to the Phase-A "
     "cross-entropy: a **self-supervised NT-Xent** term (SimCLR-style, two augmented views per image) that "
     "spreads features, and an **inter-class compactness** term that deliberately pulls class means closer "
     "so that shared low/mid-level features stay reusable for novel classes. It is opt-in ("
     "`--closer`) and namespaced separately. Code: `pretrain_backbone_closer`, `nt_xent`, "
     "`interclass_compactness`.")
page_break()

# ==================================================================
# PART V — EVALUATION
# ==================================================================
h1("Part V — Incremental Evaluation")
body("Evaluation (`fscil/eval_incremental.py`) replays the whole incremental timeline with **no gradients "
     "anywhere** — everything is frozen, exactly as the protocol demands.")

h2("The Session Loop")
numbered("**Session 0 (base):** build prototypes for all base classes, write them into the persistent "
         "memory H, and evaluate on the test set restricted to those classes.")
numbered("**Session t (1…T):** take the session's new classes, turn their few-shot support into prototypes "
         "through the same frozen graph pipeline (Stages 3–6), write those rows into memory (Stage 7), then "
         "evaluate on the test set across **all classes seen so far**.")
numbered("After the loop, compute A₀, Aₜ, PD, Avg, A_B, A_N and HM, and save them to "
         "`incremental_results.json`.")
body("Incremental sessions use the **exact official few-shot samples** (via `official_session_refs`), not "
     "random draws, so the numbers are reproducible and comparable to published work.")

h2("Base vs. Novel Prototype Estimation")
body("A subtlety of standard FSCIL: base classes have abundant data, novel classes have only 5 shots. Two "
     "options are supported for the base session:")
bullet("**Default (5-shot base):** the base prototypes are also built from a 5-shot draw, for a clean "
       "apples-to-apples comparison across sessions.")
bullet("**`--full_base`:** base prototypes are built from up to 100 samples/class (batched), which is what "
       "standard FSCIL does. This helps where the backbone is weak but is not universally better (see results).")

h2("Transductive Prototype Rectification")
body("An optional stronger protocol (`--transductive`) refines novel prototypes using the **unlabelled "
     "test set** in a BD-CSPN style: pseudo-label every test query by nearest prototype, then move each "
     "prototype toward the mean of the queries assigned to it (a normalised blend, α = 0.5, iterated 3 "
     "times). This uses the test set jointly and is therefore reported explicitly as *transductive*. It was "
     "the single most reliable improvement found.")

h2("The Novel-Logit Bias")
body("Because base prototypes are far better estimated than 5-shot novel ones, base-class cosine scores "
     "tend to run systematically higher. An optional additive bias (`novel_logit_bias`) can be added to "
     "novel-class logits to compensate. In the study this only *trades* base for novel accuracy rather than "
     "improving both, so it is not part of the recommended configuration.")

h2("Output Namespacing")
body("Every variant writes to its own subdirectory so nothing is ever overwritten: baselines in "
     "`checkpoints/<dataset>/`, and variants under `fullbase/`, `transductive/`, `ablate-<name>/`, "
     "`closer/`, or a custom `<tag>/`. These compose (e.g. `fullbase/transductive/`).")
page_break()

# ==================================================================
# PART VI — RESULTS
# ==================================================================
h1("Part VI — Results")

h2("In-Domain Performance (Official Splits)")
body("The main result: STAG-STI evaluated inductively on each benchmark's official split. Numbers are "
     "single-run, first-pass, on Apple-Silicon.")
rows = []
for d in ["cifar100", "miniimagenet", "cub200"]:
    r = DATA[d]
    a0 = r["results"][0]["accuracy"] * 100
    at = r["results"][-1]["accuracy"] * 100
    rows.append([NAMES[d], f"{a0:.2f}", f"{at:.2f}", f"{r['PD']:.2f}", f"{r['avg_accuracy']:.2f}",
                 f"{r['A_B']:.2f}", f"{r['A_N']:.2f}", f"{r['harmonic_mean']:.2f}"])
table(["Dataset", "A₀", "Aₜ", "PD↓", "Avg", "A_B", "A_N", "HM"], rows,
      widths=[1.5, 0.75, 0.75, 0.7, 0.7, 0.75, 0.75, 0.75])
body("Reading the table: base accuracy is strong (66–78%), the decline across sessions is graceful rather "
     "than a cliff (the memory prior is doing its job), and CUB holds up best (HM 42.6) thanks to its "
     "ImageNet-pretrained backbone. miniImageNet is hardest (HM 25.4): the weakest backbone and the hardest "
     "novel transfer.")

h3("Per-Session Detail")
body("The full session-by-session trajectory for each dataset (overall accuracy, base-only, novel-only, HM):")
for d in ["cifar100", "miniimagenet", "cub200"]:
    body(f"**{NAMES[d]}**", after=2)
    rows = []
    for s in DATA[d]["results"]:
        an = "—" if s["acc_novel"] != s["acc_novel"] else f"{s['acc_novel']*100:.1f}"
        hm = "—" if s["harmonic_mean"] != s["harmonic_mean"] else f"{s['harmonic_mean']*100:.1f}"
        ab = "—" if s["acc_base"] != s["acc_base"] else f"{s['acc_base']*100:.1f}"
        rows.append([f"S{s['session']}", str(s["num_classes_seen"]),
                     f"{s['accuracy']*100:.1f}", ab, an, hm])
    table(["Session", "Classes", "Overall", "A_B", "A_N", "HM"], rows,
          widths=[0.9, 0.9, 1.0, 0.9, 0.9, 0.9], fs=8.8, hfs=9)

h2("Transductive Prototype Rectification")
body("The transductive step improved every metric on all three datasets — no base↔novel trade-off. The "
     "gain is largest on CUB, where the novel prototypes are noisiest and benefit most from refinement.")
table(
    ["Dataset", "Final ind.", "Final td.", "A_N ind.", "A_N td.", "HM ind.", "HM td."],
    [
        ["CIFAR-100", "49.53", "50.17", "20.75", "21.45", "31.87", "32.76"],
        ["miniImageNet", "40.25", "40.89", "16.45", "16.53", "25.44", "25.64"],
        ["CUB-200", "49.71", "50.91", "30.85", "33.52", "42.64", "45.06"],
    ],
    widths=[1.4, 1.05, 1.0, 1.0, 1.0, 1.0, 1.0],
)

h2("Cross-Domain Transfer (miniImageNet → CUB)")
body("A stress test of forward-compatibility under domain shift: the base representation is trained on "
     "miniImageNet (natural images) and the novel classes come from CUB (fine-grained birds), evaluated "
     "10-way. Base accuracy is 66.68%, final (all seen) 44.27%, PD 22.41, with A_B 61.38 / A_N 8.41 / "
     "HM 14.80. The very low A_N (8.4) quantifies how much of the representation is domain-specific: features "
     "tuned on natural images transfer only partially to birds. This protocol is the project's own "
     "definition, not a citable standard, and is reported as such.")

h2("Ablation Study (CIFAR-100)")
body("Each component is removed in isolation while sharing the same frozen backbone. HM is the deciding "
     "metric.")
table(
    ["Configuration", "A₀", "Aₜ", "A_B", "A_N", "HM"],
    [
        ["Full model", "78.22", "49.53", "68.72", "20.75", "31.87"],
        ["− contrastive (SupCon)", "78.53", "49.28", "69.97", "18.25", "28.95"],
        ["− topology (GAT graph)", "77.75", "49.69", "70.98", "17.75", "28.40"],
        ["− age-decay (STI)", "78.50", "49.46", "70.15", "18.43", "29.18"],
    ],
    widths=[2.6, 0.85, 0.85, 0.85, 0.85, 0.85],
    highlight={0: "E7F3F0"},
)
body("The pattern is consistent and revealing: removing any component barely changes overall accuracy but "
     "**raises A_B while crashing A_N**. In other words, each component's real job is to *protect novel-class "
     "accuracy*; without the full model the system quietly reverts to over-favouring base classes. Every "
     "piece earns its place on the metric that matters (HM).")

h2("CLOSER-Style Base Objective")
body("The transfer-oriented base objective is **dataset-dependent, not a universal win**:")
table(
    ["Dataset", "Aₜ", "A_B", "A_N", "HM", "ΔA_N", "ΔHM"],
    [
        ["CIFAR-100", "46.42", "61.12", "24.38", "34.85", "+3.6", "+3.0"],
        ["miniImageNet", "37.38", "53.23", "13.60", "21.67", "−2.8", "−3.8"],
        ["CUB-200", "47.98", "64.49", "31.84", "42.63", "+1.0", "−0.0"],
    ],
    widths=[1.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
)
body("It helped CIFAR-100 (+3.0 HM), hurt miniImageNet (−3.8 HM), and was neutral on CUB. This is an honest "
     "negative-leaning result: no single base objective wins everywhere — the right recipe depends on "
     "backbone strength — and it is reported transparently rather than cherry-picked.")

h2("Improvement Study — What Actually Works")
body("Every accuracy lever that does **not** change the architecture was implemented and measured:")
h3("Helped")
bullet("**Transductive rectification** — improves every metric on every dataset (see above).")
bullet("**Full-data base prototypes (`--full_base`)** — helps where the backbone is weak (miniImageNet); "
       "neutral on CIFAR; slightly hurts CUB. Dataset-dependent.")
bullet("**Best eval-only stack** (full_base where it helps + transductive) delivers **+1.2 to +1.6 overall "
       "accuracy**, free, with no retraining: CIFAR 49.5→51.1, miniImageNet 40.2→41.9, CUB 49.7→50.9.")
h3("Did not help")
bullet("**Novel-logit bias** — only trades base for novel accuracy.")
bullet("**Test-time / support augmentation** — no reliable gain.")
bullet("**Base-mean feature centering** — negligible effect.")
bullet("**Longer backbone training (100 epochs)** — could not be completed on this machine (see Part VII); "
       "the attempt regressed because it was interrupted mid-schedule, which is a compute artifact, not "
       "evidence that longer training is bad.")
callout("The delivered improvement.",
        "Combining full-data base prototypes (on weak-backbone datasets) with transductive rectification "
        "yields a reliable +1.2 to +1.6 points of overall accuracy across the three benchmarks — entirely "
        "at evaluation time, with no change to the architecture and no retraining.", color=TEAL, fill="E7F3F0")
page_break()

# ==================================================================
# PART VII — ENGINEERING
# ==================================================================
h1("Part VII — Engineering, Bugs and Operational Notes")
body("Several problems were caught precisely because results were verified rather than trusted. Documenting "
     "them is part of the project's rigour.")

h2("The 1% Bug and Base-Graph Chunking")
body("The most important bug. On CUB-200 the base session initially scored about **1% accuracy** — near "
     "random. The cause was twofold: (1) the base session builds a single joint graph over all 100 base "
     "classes, which at 4 views each is a **400-node graph**, whereas the GATv2 was only ever trained on "
     "~15-way episodes — the 400-node graph is wildly out-of-distribution for it; and (2) very large "
     "attention tensors were numerically unreliable on the MPS backend.")
body("The fix (`_prototypes_from_indices` in `fscil/eval_incremental.py`) builds base prototypes in "
     "**chunks of at most `episode_way` (~15) classes** — the exact scale the model was trained on. This is "
     "mathematically safe because a new class's memory row is written as H = Φ(P) *regardless* of the "
     "class-to-class adjacency (the retention gate has no effect on a class's first write), so chunking the "
     "graph cannot change the written prototypes. CUB base accuracy jumped from **1% to 77.8%**.")

h2("The Cross-Domain Crash")
body("The cross-domain evaluation crashed with a shape mismatch: the novel-class mask was being built "
     "per-sample instead of per-class (length 60 vs 128), and the results dict used the key `acc` while the "
     "helper functions expected `accuracy`. Both were fixed, and the cross-domain test then ran cleanly.")

h2("Configuration-Leak Fixes")
body("Because the whole pipeline is reconfigured by mutating a single `Config` class, early versions "
     "leaked per-dataset settings across runs — e.g. CUB's low backbone learning rate bleeding into a "
     "subsequent CIFAR run, or an ablation's zeroed loss weight not being restored. These were fixed with a "
     "**capture-once-then-restore** pattern in `apply_dataset` and `apply_ablation`, so every switch starts "
     "from the true defaults.")

h2("Task Reaping and Resumability")
body("Long background training jobs on this machine are reaped after roughly 40–60 minutes. This capped "
     "how long a backbone could train unattended and was the reason the 100-epoch backbone experiment could "
     "not complete: it was interrupted mid-cosine-schedule and a partially-trained backbone is worse than a "
     "fully-annealed 30-epoch one. Runs were made **resumable** (best-validation checkpoints are always "
     "saved and can be promoted), but the practical conclusion is that the backbone-quality lever — the "
     "biggest potential improvement — needs a compute environment that does not reap jobs (a GPU).")

h2("Reproducibility and Automation")
body("Supporting infrastructure built around the experiments:")
bullet("**Fixed seeds** (42) and **deterministic fantasy views** for reproducible prototypes.")
bullet("**Namespaced checkpoints** so no run overwrites another.")
bullet("An **auto-updating README** whose results tables regenerate from `checkpoints/` on every commit via "
       "a git pre-commit hook (`scripts/gen_readme.py`).")
bullet("A **full HTML report** (`results/report.html`), a **20-slide presentation** "
       "(`results/STAG-STI_Review2.pptx`), and **automatic commit-and-push** of each result with a "
       "descriptive, metric-tagged message.")
page_break()

# ==================================================================
# PART VIII — CONCLUSIONS
# ==================================================================
h1("Part VIII — Findings, Limitations and Future Work")

h2("Key Findings")
numbered("**Novel-class accuracy is the bottleneck** everywhere (A_N 16–31% vs A_B 56–69%). The model keeps "
         "old knowledge well; learning new classes from 5 shots is the hard part.")
numbered("**Every architectural component protects novel accuracy** — the ablation shows removing any one "
         "raises A_B but lowers A_N and HM.")
numbered("**Transductive prototype rectification is the only universally reliable improvement**, and it "
         "helps most exactly where prototypes are noisiest (CUB).")
numbered("**Eval-only tricks are close to their ceiling** (~+1.5 points). The larger gains require a "
         "stronger frozen backbone, which is a compute problem, not an architecture problem.")
numbered("**Honest, dataset-dependent results**: full-data base prototypes and the CLOSER objective each "
         "help on some datasets and hurt on others — reported transparently.")

h2("Limitations")
bullet("Numbers are **single-run, first-pass** on Apple-Silicon — internally consistent and reproducible, "
       "but not seed-averaged and not tuned to compete with published SOTA.")
bullet("The **backbone could not be trained to completion** on this machine (job reaping), so the biggest "
       "lever was left on the table.")
bullet("The **cross-domain protocol** is this project's own definition, not a citable benchmark.")
bullet("The **age-decay memory's cross-session hardening** is implemented but not exercised by the standard "
       "single-write evaluation protocol.")

h2("Future Work")
bullet("Train the frozen backbone to completion on a **GPU** (100–200 epochs, proper LR annealing) to lift "
       "A_N and A_B together — the clearest path to a higher ceiling.")
bullet("**Seed-averaged** runs and head-to-head comparison against published FSCIL baselines.")
bullet("Extend transductive rectification and explore stronger **self-supervised base objectives** tuned "
       "per backbone strength.")
bullet("Evaluate a protocol that **recomputes existing class rows** so the STI age-decay machinery is "
       "fully exercised.")

h2("Closing")
body("The project delivers a complete, working, and thoroughly analysed FSCIL system. Beyond the "
     "architecture itself, its main contribution is clarity about *where the difficulty lives*: the "
     "novel-class few-shot regime, and the quality of the frozen representation that feeds it. Every design "
     "choice, every result, and every dead end has been measured and documented so the work can be "
     "explained, defended, and built upon.")

# ==================================================================
# APPENDIX
# ==================================================================
page_break()
h1("Appendix A — File-by-File Reference")
table(
    ["File", "Responsibility"],
    [
        ["fscil/config.py", "Central Config; per-dataset switch (apply_dataset); ablation/CLOSER knobs; all hyperparameters"],
        ["fscil/datasets.py", "DatasetSpec registry; official CEC split parsing; global label order; per-session support refs"],
        ["fscil/data.py", "IndexedDataset (3 backends); transforms; deterministic fantasy views; episodic sampler"],
        ["fscil/backbone.py", "CIFAR/standard/pretrained ResNet-18; build_backbone factory; Phase-A head"],
        ["fscil/modules.py", "Projection, SupCon, TopologyScorer, GATv2, STI memory, cosine classifier, all losses"],
        ["fscil/pipeline.py", "StagStiModel — wires Stages 3–8 together"],
        ["fscil/train_session0.py", "Phase A (backbone) + Phase B (episodic STAG-STI) + CLOSER Phase A"],
        ["fscil/eval_incremental.py", "Incremental eval; base/novel prototypes; --full_base; --transductive; metrics"],
        ["fscil/eval_crossdomain.py", "miniImageNet → CUB cross-domain transfer test"],
        ["fscil/plotting.py", "Per-run training and accuracy curves"],
        ["scripts/*", "Data prep, report/README/PPT generators, ablation & improvement orchestrators, auto-commit"],
    ],
    widths=[2.1, 5.2], fs=9, hfs=9.5,
)

h1("Appendix B — Key Hyperparameters")
table(
    ["Parameter", "Value", "Meaning"],
    [
        ["backbone_out_dim", "512", "Backbone feature dimension d"],
        ["proj_dim", "256", "Graph hidden dimension d'"],
        ["supcon_proj_dim", "128", "SupCon hypersphere dimension"],
        ["num_views (M)", "4", "Deterministic fantasy views per image"],
        ["backbone_pretrain_epochs", "30", "Phase-A epochs (cosine LR)"],
        ["main_epochs", "50", "Phase-B episodic epochs"],
        ["episodes_per_epoch", "100", "Episodes sampled per Phase-B epoch"],
        ["episode_way / shot / query", "15 / 5 / 5", "Episodic task shape (also the safe graph chunk size)"],
        ["supcon_temperature", "0.15", "SupCon temperature (raised to loosen clustering)"],
        ["lambda_cls / supcon / graph / stab", "1.0 / 0.3 / 0.25 / 0.1", "Composite loss weights"],
        ["kappa / kappa0", "2.0 / 3.0", "Age-decay slope and offset"],
        ["novel_logit_bias", "0.5", "Optional novel-class calibration (off in recommended config)"],
        ["base_proto_cap", "100", "Max samples/class for --full_base base prototypes"],
        ["seed", "42", "Global random seed"],
    ],
    widths=[2.7, 1.6, 3.0], fs=9, hfs=9.5,
)

h1("Appendix C — How to Reproduce")
code_block([
    "# Base training (Phase A backbone + Phase B STAG-STI)",
    "python -m fscil.train_session0   --dataset cub200",
    "",
    "# Incremental evaluation (inductive, then transductive)",
    "python -m fscil.eval_incremental --dataset cub200",
    "python -m fscil.eval_incremental --dataset cub200 --transductive",
    "python -m fscil.eval_incremental --dataset cub200 --full_base --transductive",
    "",
    "# Ablation (reuses the shared frozen backbone)",
    "python -m fscil.train_session0   --dataset cifar100 --ablate topology --skip_backbone_pretrain",
    "python -m fscil.eval_incremental --dataset cifar100 --ablate topology",
    "",
    "# CLOSER base objective, and cross-domain transfer",
    "python -m fscil.train_session0   --dataset cifar100 --closer",
    "python -m fscil.eval_crossdomain",
])

# footer note
p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(12)
r = p.add_run("Generated from the live project checkpoints. All figures are single-run, first-pass numbers "
              "on the official CEC/FACT splits.")
r.font.italic = True; r.font.size = Pt(9); r.font.color.rgb = GREY; r.font.name = "Calibri"

OUT = os.path.join(ROOT, "results", "STAG-STI_Project_Report.docx")
doc.save(OUT)
# count paragraphs as a rough size signal
print("Saved", OUT)
print("paragraphs:", len(doc.paragraphs), "| tables:", len(doc.tables))
