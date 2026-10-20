"""
Build EE136IU Topic 4 Capstone Presentation
Format : Bắt chước EE136IU_Capstone_Mini_Project_AMC_Brief.pptx
Style  : Tuân thủ slide_style_requirements.md 100%
Font   : Projector-ready — body Pt(13.5), title Pt(26), card-header Pt(15)

STRUCTURE (12 slides, 5 sections):
  S1  — Title
  S2  — Table of Contents (5 sections)
  S3  — [Vấn Đề]          Problem Framing & Low-SNR Physics
  S4  — [Giải Pháp]       Technical Foundation & Key Concepts
  S5  — [Giải Pháp]       RF Augmentation Pipeline (Topic 04)
  S6  — [Implementation]  1D CNN Neural Receiver Architecture
  S7  — [Implementation]  Training Benchmark & Convergence
  S8  — [Kết Quả]         Accuracy vs. SNR Grid Results
  S9  — [Kết Quả]         Confusion Matrix & Failure Analysis
  S10 — [Demo]            Interactive Demo & Fail-Safe Video
  S11 — [Q&A]             Conclusion & Open Q&A
  S12 — Thank You
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ── COLOR PALETTE (per slide_style_requirements.md) ──────────────
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)  # #FFFFFF
C_DARK   = RGBColor(0x0F, 0x17, 0x2A)  # #0F172A body/title text
C_CARD   = RGBColor(0xF8, 0xFA, 0xFC)  # #F8FAFC container card
C_BORDER = RGBColor(0xE2, 0xE8, 0xF0)  # #E2E8F0 card border
C_BLUE   = RGBColor(0x02, 0x84, 0xC7)  # #0284C7 primary — accent bar, keyword
C_DEEP   = RGBColor(0x25, 0x63, 0xEB)  # #2563EB secondary — badge, milestone
C_FOOTER = RGBColor(0x94, 0xA3, 0xB8)  # #94A3B8 footer text

# ── FONT HIERARCHY ────────────────────────────────────────────────
FONT_H = "Manrope"   # Slide titles & card headers
FONT_B = "Inter"     # Body text & bullet points
FONT_C = "Consolas"  # Math, tensor shapes, code

# ── PROJECTOR-READY FONT SIZES ───────────────────────────────────
SZ_SLIDE_TITLE  = Pt(30)    # Slide main title
SZ_BADGE        = Pt(12)    # Section badge text
SZ_CARD_HEADER  = Pt(18)    # Card / block header
SZ_BODY         = Pt(16)    # Body bullets — large enough for projector
SZ_FOOTER       = Pt(9)     # Footer (small, edge info)
SZ_SUBTITLE     = Pt(13.5)  # Under-title subtitle line

TOTAL = 12
prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


# ── HELPERS ──────────────────────────────────────────────────────

def add_shape(slide, left, top, w, h, fill=None, border=None, rounded=True):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        left, top, w, h
    )
    if rounded:
        shape.adjustments[0] = 0.04
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if border:
        shape.line.color.rgb = border
        shape.line.width = Pt(0.75)
    else:
        shape.line.fill.background()
    return shape


def add_tb(slide, l, t, w, h):
    return slide.shapes.add_textbox(l, t, w, h)


def run(para, text, font=FONT_B, size=SZ_BODY, bold=False, color=C_DARK):
    r = para.add_run()
    r.text = text
    r.font.name = font
    r.font.size = size
    r.font.bold = bold
    r.font.color.rgb = color
    return r


def footer(slide, slide_num):
    add_shape(slide, Inches(0.8), Inches(6.6), Inches(11.73), Inches(0.01),
              fill=C_BORDER, rounded=False)
    tb_l = add_tb(slide, Inches(0.8), Inches(6.72), Inches(8.5), Inches(0.35))
    run(tb_l.text_frame.paragraphs[0],
        "EE136: Machine Learning for Wireless Communications | Topic 4 Capstone Defense",
        FONT_B, SZ_FOOTER, color=C_FOOTER)
    tb_r = add_tb(slide, Inches(10.5), Inches(6.72), Inches(2.03), Inches(0.35))
    p_r = tb_r.text_frame.paragraphs[0]
    p_r.alignment = PP_ALIGN.RIGHT
    run(p_r, f"Slide {slide_num} of {TOTAL}", FONT_B, SZ_FOOTER, color=C_FOOTER)


def chrome(slide, section_label, title, subtitle, slide_num):
    """Standard chrome: accent bar + badge + title + subtitle + footer."""
    add_shape(slide, Inches(0.8), Inches(0.42), Inches(0.15), Inches(4.85),
              fill=C_BLUE, rounded=True)
    badge_w = Inches(len(section_label) * 0.115 + 0.55)
    add_shape(slide, Inches(1.1), Inches(0.42), badge_w, Inches(0.36),
              fill=C_DEEP, rounded=True)
    tb_b = add_tb(slide, Inches(1.18), Inches(0.44), badge_w, Inches(0.33))
    run(tb_b.text_frame.paragraphs[0], section_label, FONT_H, SZ_BADGE, bold=True, color=C_WHITE)

    tb_t = add_tb(slide, Inches(1.1), Inches(0.88), Inches(11.4), Inches(0.72))
    tf_t = tb_t.text_frame; tf_t.word_wrap = True
    run(tf_t.paragraphs[0], title, FONT_H, SZ_SLIDE_TITLE, bold=True, color=C_DARK)

    if subtitle:
        tb_s = add_tb(slide, Inches(1.1), Inches(1.52), Inches(11.4), Inches(0.35))
        run(tb_s.text_frame.paragraphs[0], subtitle, FONT_B, SZ_SUBTITLE, color=C_DARK)

    footer(slide, slide_num)


def card(slide, left, top, w, h, title, bullets):
    """
    Container Card — Fill #F8FAFC, border #E2E8F0
    bullets: list of (keyword|None, desc, mono:bool)
    """
    add_shape(slide, left, top, w, h, fill=C_CARD, border=C_BORDER, rounded=True)
    tb = add_tb(slide, left + Inches(0.2), top + Inches(0.15),
                w - Inches(0.4), h - Inches(0.3))
    tf = tb.text_frame; tf.word_wrap = True

    # Card header — Manrope Bold, Primary Blue
    p0 = tf.paragraphs[0]; p0.space_after = Pt(10)
    run(p0, title, FONT_H, SZ_CARD_HEADER, bold=True, color=C_BLUE)

    for kw, desc, mono in bullets:
        p = tf.add_paragraph(); p.space_after = Pt(7)
        if kw:
            run(p, f"• {kw}: ", FONT_H, SZ_BODY, bold=True, color=C_BLUE)
        run(p, desc, FONT_C if mono else FONT_B, SZ_BODY, color=C_DARK)


def two_cards(slide, l_title, l_bullets, r_title, r_bullets):
    CARD_TOP = Inches(1.92)
    CARD_H   = Inches(4.45)
    card(slide, Inches(0.80), CARD_TOP, Inches(5.70), CARD_H, l_title, l_bullets)
    card(slide, Inches(6.83), CARD_TOP, Inches(5.70), CARD_H, r_title, r_bullets)


# ════════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ════════════════════════════════════════════════════════════════════
def s01_title():
    s = prs.slides.add_slide(BLANK)
    add_shape(s, Inches(0.8), Inches(1.5), Inches(0.15), Inches(4.7), fill=C_BLUE)

    # Course badge
    add_shape(s, Inches(1.15), Inches(1.5), Inches(5.8), Inches(0.44), fill=C_DEEP)
    tb_b = add_tb(s, Inches(1.22), Inches(1.53), Inches(5.7), Inches(0.40))
    run(tb_b.text_frame.paragraphs[0],
        "EE136IU • CAPSTONE MINI-PROJECT DEFENSE", FONT_H, Pt(12), bold=True, color=C_WHITE)

    # Main title — Manrope Bold, large
    tb_t = add_tb(s, Inches(1.15), Inches(2.1), Inches(11.0), Inches(1.7))
    tf_t = tb_t.text_frame; tf_t.word_wrap = True
    p1 = tf_t.paragraphs[0]; p1.space_after = Pt(4)
    run(p1, "TOPIC 04: Data Augmentation &", FONT_H, Pt(32), bold=True, color=C_DARK)
    p2 = tf_t.add_paragraph()
    run(p2, "Noise Resilience for Low-SNR", FONT_H, Pt(32), bold=True, color=C_DARK)

    # Subtitle
    tb_s = add_tb(s, Inches(1.15), Inches(3.92), Inches(11.0), Inches(0.55))
    run(tb_s.text_frame.paragraphs[0],
        "Enhancing Deep Receiver Sensitivity in Noise-Limited Regimes  (−10 dB → +18 dB)",
        FONT_B, Pt(15), color=C_DEEP)

    # Team card
    card(s, Inches(1.15), Inches(4.6), Inches(11.0), Inches(1.75),
         "Team Members & Speaking Roles",
         [
             ("Thanh vien 1 — Team Lead", "Signal Processing · Preprocessing Pipeline · RMS Normalization", False),
             ("Thanh vien 2 — ML Engineer", "1D CNN Architecture · RF Augmentation PyTorch Pipeline", False),
             ("Thanh vien 3 — Demo & Eval", "SNR Benchmarking · Failure Analysis · Streamlit Web UI", False),
         ])

    # Footer
    add_shape(s, Inches(0.8), Inches(6.6), Inches(11.73), Inches(0.01), fill=C_BORDER, rounded=False)
    tb_fl = add_tb(s, Inches(0.8), Inches(6.72), Inches(8.5), Inches(0.35))
    run(tb_fl.text_frame.paragraphs[0],
        "EE136: Machine Learning for Wireless Communications | Capstone Mini-Project Brief",
        FONT_B, SZ_FOOTER, color=C_FOOTER)
    tb_fr = add_tb(s, Inches(10.5), Inches(6.72), Inches(2.03), Inches(0.35))
    p_r = tb_fr.text_frame.paragraphs[0]; p_r.alignment = PP_ALIGN.RIGHT
    run(p_r, f"Slide 1 of {TOTAL}", FONT_B, SZ_FOOTER, color=C_FOOTER)


# ════════════════════════════════════════════════════════════════════
# SLIDE 2 — TABLE OF CONTENTS (5 SECTIONS)
# ════════════════════════════════════════════════════════════════════
def s02_toc():
    s = prs.slides.add_slide(BLANK)
    chrome(s, "TABLE OF CONTENTS", "Agenda — 5 Main Sections", None, 2)

    sections = [
        ("01", "Van De",
         "Channel Physics · Low-SNR Limitations · Why Baseline Fails",
         "Slide 3"),
        ("02", "Giai Phap & Khai Niem",
         "Technical Foundation · Data Augmentation · Key Concepts",
         "Slides 4-5"),
        ("03", "Implementation",
         "1D CNN Architecture · Training Setup & Convergence",
         "Slides 6-7"),
        ("04", "Ket Qua & Demo San Pham",
         "Accuracy vs. SNR · Confusion Matrix · Interactive Demo",
         "Slides 8-10"),
        ("05", "Q&A",
         "Conclusion · Engineering Takeaways · Open Questions",
         "Slide 11"),
    ]

    # Single wide card
    CARD_TOP = Inches(1.85)
    CARD_H   = Inches(4.55)
    add_shape(s, Inches(0.80), CARD_TOP, Inches(11.73), CARD_H,
              fill=C_CARD, border=C_BORDER)
    tb = add_tb(s, Inches(1.05), CARD_TOP + Inches(0.25),
                Inches(11.23), CARD_H - Inches(0.5))
    tf = tb.text_frame; tf.word_wrap = True

    for i, (num, title, desc, pg) in enumerate(sections):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(14)

        run(p, f"{num}.  ", FONT_H, Pt(18), bold=True, color=C_DEEP)
        run(p, title + "    ", FONT_H, Pt(18), bold=True, color=C_DARK)
        run(p, f"— {desc}", FONT_B, Pt(16), color=C_DARK)
        run(p, f"   ({pg})", FONT_B, Pt(14), color=C_FOOTER)


# ════════════════════════════════════════════════════════════════════
# SLIDE 3 — VẤN ĐỀ: PROBLEM FRAMING
# ════════════════════════════════════════════════════════════════════
def s03_problem():
    s = prs.slides.add_slide(BLANK)
    chrome(s, "01 — VAN DE",
           "Channel Physics & Low-SNR Limitations",
           "Why Modulation Classification Fails in Noise-Limited Regimes", 3)
    two_cards(s,
        "Baseband Signal Model",
        [
            ("Model", "r[n] = h · s[n] + w[n]", True),
            ("Noise w[n]", "AWGN: w[n] ~ CN(0, σ²)  — complex Gaussian", True),
            ("Phase shift", "CFO rotation e^(jθ),  θ ∈ [0, 2π)", True),
        ],
        "Why Baseline Fails",
        [
            ("Phase overfitting", "Baseline locks to fixed phase orientation in training", False),
            ("Accuracy collapse", "Below 25% accuracy at SNR = −10 dB", False),
            ("Topic 4 solution", "Stochastic RF augmentation forces phase-invariant features", False),
        ]
    )


# ════════════════════════════════════════════════════════════════════
# SLIDE 4 — GIẢI PHÁP: TECHNICAL FOUNDATION & KEY CONCEPTS
# ════════════════════════════════════════════════════════════════════
def s04_concepts():
    s = prs.slides.add_slide(BLANK)
    chrome(s, "02 — GIAI PHAP & KHAI NIEM",
           "Technical Foundation & Key Concepts",
           "Standardized I/Q Tensor · RMS Normalization · 8-Class Classification", 4)
    two_cards(s,
        "Baseband Tensor Specification",
        [
            ("Input tensor", "[N_batch, 2, 128]  — float32", True),
            ("Channel layout", "Row 0: I[n]   |   Row 1: Q[n]", True),
            ("RMS Norm", "r_norm[n] = r[n] / sqrt( mean(I^2 + Q^2) )", True),
        ],
        "Evaluation Protocol",
        [
            ("SNR Grid", "15 points: −10 dB to +18 dB  (step 2 dB)", False),
            ("Data split", "70% Train / 15% Val / 15% Test  — zero leakage", False),
            ("Metrics", "Accuracy vs. SNR · Macro-F1 · Row-normalized Confusion Matrix", False),
        ]
    )


# ════════════════════════════════════════════════════════════════════
# SLIDE 5 — GIẢI PHÁP: RF AUGMENTATION PIPELINE
# ════════════════════════════════════════════════════════════════════
def s05_augment():
    s = prs.slides.add_slide(BLANK)
    chrome(s, "02 — GIAI PHAP & KHAI NIEM",
           "Topic 04 — RF Data Augmentation Pipeline",
           "4 Physical Transforms Applied Per Training Batch to Force Noise Robustness", 5)
    two_cards(s,
        "RFAugmentationPipeline — 4 Transforms",
        [
            ("Phase rotation", "θ ~ U[0, 2π)  →  rotate (I, Q) by θ", True),
            ("AWGN injection", "std ~ U[0.01, 0.08]  — low-SNR sensitivity boost", True),
            ("Time-shift jitter", "Circular shift k ~ U[−8, +8] samples", True),
        ],
        "Key Insight",
        [
            ("Model A — Baseline", "Standard 1D CNN, no augmentation", False),
            ("Model B — Augmented", "Same 1D CNN with RFAugmentationPipeline", False),
            ("Result", "Physical augmentation acts as domain-aware regularizer", False),
        ]
    )


# ════════════════════════════════════════════════════════════════════
# SLIDE 6 — IMPLEMENTATION: 1D CNN ARCHITECTURE
# ════════════════════════════════════════════════════════════════════
def s06_arch():
    s = prs.slides.add_slide(BLANK)
    chrome(s, "03 — IMPLEMENTATION",
           "3-Block Deep 1D Convolutional Neural Receiver",
           "Input: [N_batch, 2, 128]   →   Output: [N_batch, 8]", 6)
    two_cards(s,
        "1D Convolutional Feature Extractor",
        [
            ("Block 1", "Conv1D(2→64, k=7) → BN → ReLU → MaxPool    [128→64]", True),
            ("Block 2", "Conv1D(64→128, k=5) → BN → ReLU → MaxPool  [64→32]", True),
            ("Block 3", "Conv1D(128→256, k=3) → BN → ReLU → AvgPool [32→1]", True),
        ],
        "Classifier Head",
        [
            ("Flatten", "[N, 256, 1]  →  [N, 256]  feature vector", True),
            ("FC 1 & 2", "Linear(256→128) → ReLU → Dropout → Linear(128→8)", True),
            ("Config", "175,432 params · Adam · CosineAnnealingLR", False),
        ]
    )


# ════════════════════════════════════════════════════════════════════
# SLIDE 7 — IMPLEMENTATION: TRAINING BENCHMARK
# ════════════════════════════════════════════════════════════════════
def s07_training():
    s = prs.slides.add_slide(BLANK)
    chrome(s, "03 — IMPLEMENTATION",
           "Training Convergence — Baseline vs. Topic 04",
           "25 Epochs · 18,000 Bursts · Identical Hyperparameters for Fair Comparison", 7)
    two_cards(s,
        "Model A — Baseline (No Augmentation)",
        [
            ("Train / Val Acc", "76.95%  /  65.33%", True),
            ("Train / Val Loss", "0.4774  /  0.8757", True),
            ("Overfit gap", "Delta Loss = 0.399 — large overfitting", False),
        ],
        "Model B — Augmented (Topic 04)",
        [
            ("Train / Val Acc", "63.13%  /  66.07%", True),
            ("Train / Val Loss", "0.8069  /  0.7670", True),
            ("Key win", "Val Loss 0.1087 lower than Baseline — better generalization", False),
        ]
    )


# ════════════════════════════════════════════════════════════════════
# SLIDE 8 — KẾT QUẢ: ACCURACY vs SNR
# ════════════════════════════════════════════════════════════════════
def s08_accuracy():
    s = prs.slides.add_slide(BLANK)
    chrome(s, "04 — KET QUA & DEMO",
           "Accuracy vs. SNR Grid — Baseline vs. Topic 04",
           "Progressive Recovery from Noise-Limited to High-SNR Asymptote", 8)

    img = r"plots/accuracy_vs_snr.png"
    if os.path.exists(img):
        s.shapes.add_picture(img, Inches(0.80), Inches(1.92), width=Inches(6.8))
        card_l, card_w = Inches(7.90), Inches(4.63)
    else:
        card_l, card_w = Inches(0.80), Inches(11.73)

    card(s, card_l, Inches(1.92), card_w, Inches(4.45),
         "Quantitative Gains",
         [
             ("SNR = −8 dB", "25.86%  →  30.17%   (+4.31% gain)", False),
             ("SNR = −2 dB", "57.85%  →  61.16%   (+3.31% gain)", False),
             ("Val Loss", "Baseline 0.8757  vs  Augmented 0.7670  (down 0.1087)", False),
         ])


# ════════════════════════════════════════════════════════════════════
# SLIDE 9 — KẾT QUẢ: CONFUSION MATRIX & FAILURE ANALYSIS
# ════════════════════════════════════════════════════════════════════
def s09_confusion():
    s = prs.slides.add_slide(BLANK)
    chrome(s, "04 — KET QUA & DEMO",
           "Confusion Matrix & Root-Cause Failure Analysis",
           "Row-Normalized at Low SNR (−4 dB) — Class-Level Misclassification Breakdown", 9)

    img = r"plots/confusion_matrices_low_snr.png"
    if os.path.exists(img):
        s.shapes.add_picture(img, Inches(0.80), Inches(1.92), width=Inches(6.8))
        card_l, card_w = Inches(7.90), Inches(4.63)
    else:
        card_l, card_w = Inches(0.80), Inches(11.73)

    card(s, card_l, Inches(1.92), card_w, Inches(4.45),
         "Failure Analysis",
         [
             ("16QAM vs 64QAM", "Multi-ring amplitude smear under high AWGN power", False),
             ("Root cause", "Baseline anchors to fixed phase — fails under CFO rotation", False),
             ("Topic 4 fix", "Phase rotation training forces radial density features", False),
         ])


# ════════════════════════════════════════════════════════════════════
# SLIDE 10 — DEMO: INTERACTIVE WEB UI & FAIL-SAFE VIDEO
# ════════════════════════════════════════════════════════════════════
def s10_demo():
    s = prs.slides.add_slide(BLANK)
    chrome(s, "04 — KET QUA & DEMO",
           "Interactive Demo & Fail-Safe Video Backup",
           "py -m streamlit run app.py   →   http://localhost:8501", 10)
    two_cards(s,
        "Fail-Safe Video  [EMBED MP4 HERE]",
        [
            ("Purpose", "30-60s screen recording embedded in this slide", False),
            ("Crash shield", "Guards against terminal crash or GPU error during defense", False),
            ("Key moment", "16QAM at -4 dB:  Baseline WRONG  vs  Topic 4 CORRECT", False),
        ],
        "Streamlit Web UI",
        [
            ("SNR Slider", "Real-time adjustment from -10 dB to +18 dB", False),
            ("Side-by-side", "Baseline prediction vs. Topic 4 prediction live", False),
            ("Launch command", "py -m streamlit run app.py", True),
        ]
    )


# ════════════════════════════════════════════════════════════════════
# SLIDE 11 — Q&A: CONCLUSION & OPEN Q&A
# ════════════════════════════════════════════════════════════════════
def s11_qa():
    s = prs.slides.add_slide(BLANK)
    chrome(s, "05 • Q&A",
           "Conclusion & Open Technical Q&A",
           "Rigorous Engineering · Physical Grounding · Professional Delivery", 11)
    two_cards(s,
        "💡 Core Takeaways",
        [
            ("Physics first", "Every decision anchored in wireless physical reality", False),
            ("Sensitivity gains", "+4.31% @ −8 dB  |  +3.31% @ −2 dB  |  Val Loss ↓ 0.1087", False),
            ("Phase invariance", "Augmented model robust to arbitrary CFO offsets", False),
            ("Reproducibility", "seed=42 · 1-command demo · full modular codebase", False),
        ],
        "📊 Summary Metrics",
        [
            ("Architecture", "3-Block 1D CNN  |  175,432 params  |  [N, 2, 128]", True),
            ("Best Val Acc", "Augmented 66.07%  vs  Baseline 65.33%", False),
            ("Best Val Loss", "Augmented 0.7670  vs  Baseline 0.8757", True),
            ("Floor is open", "Ready to defend all algorithmic & design decisions ✅", False),
        ]
    )


# ════════════════════════════════════════════════════════════════════
# SLIDE 12 — THANK YOU
# ════════════════════════════════════════════════════════════════════
def s12_thankyou():
    s = prs.slides.add_slide(BLANK)
    add_shape(s, Inches(0.8), Inches(1.0), Inches(0.15), Inches(5.5), fill=C_BLUE)

    # "Thank You" large
    tb_ty = add_tb(s, Inches(1.15), Inches(1.7), Inches(11.2), Inches(2.6))
    tf_ty = tb_ty.text_frame; tf_ty.word_wrap = True
    p1 = tf_ty.paragraphs[0]; p1.space_after = Pt(10)
    run(p1, "Thank You", FONT_H, Pt(58), bold=True, color=C_DARK)
    p2 = tf_ty.add_paragraph()
    run(p2, "for your time & attention!", FONT_B, Pt(28), color=C_DEEP)

    tb_sub = add_tb(s, Inches(1.15), Inches(4.45), Inches(11.0), Inches(0.55))
    run(tb_sub.text_frame.paragraphs[0],
        "We are happy to answer any questions about our implementation, results, or design decisions.",
        FONT_B, Pt(15), color=C_DARK)

    # 3 info cards
    info = [
        ("Topic", "Data Augmentation & Noise Resilience for Low-SNR", False),
        ("Course", "EE136IU — ML for Wireless Communications", False),
        ("Demo", "py -m streamlit run app.py", True),
    ]
    for i, (label, val, mono) in enumerate(info):
        cl = Inches(0.8) + i * Inches(4.18)
        add_shape(s, cl, Inches(5.1), Inches(3.95), Inches(1.25),
                  fill=C_CARD, border=C_BORDER)
        tb_c = add_tb(s, cl + Inches(0.15), Inches(5.18), Inches(3.65), Inches(1.1))
        tf_c = tb_c.text_frame; tf_c.word_wrap = True
        p_l = tf_c.paragraphs[0]; p_l.space_after = Pt(5)
        run(p_l, label, FONT_H, Pt(12), bold=True, color=C_BLUE)
        p_v = tf_c.add_paragraph()
        run(p_v, val, FONT_C if mono else FONT_B, Pt(12.5), color=C_DARK)

    # Footer
    add_shape(s, Inches(0.8), Inches(6.6), Inches(11.73), Inches(0.01),
              fill=C_BORDER, rounded=False)
    tb_fl = add_tb(s, Inches(0.8), Inches(6.72), Inches(8.5), Inches(0.35))
    run(tb_fl.text_frame.paragraphs[0],
        "EE136: Machine Learning for Wireless Communications | Topic 4 Capstone Defense",
        FONT_B, SZ_FOOTER, color=C_FOOTER)
    tb_fr = add_tb(s, Inches(10.5), Inches(6.72), Inches(2.03), Inches(0.35))
    p_r = tb_fr.text_frame.paragraphs[0]; p_r.alignment = PP_ALIGN.RIGHT
    run(p_r, f"Slide {TOTAL} of {TOTAL}", FONT_B, SZ_FOOTER, color=C_FOOTER)


# ════════════════════════════════════════════════════════════════════
# BUILD
# ════════════════════════════════════════════════════════════════════
s01_title()
s02_toc()
s03_problem()
s04_concepts()
s05_augment()
s06_arch()
s07_training()
s08_accuracy()
s09_confusion()
s10_demo()
s11_qa()
s12_thankyou()

OUT = r"d:\Emerging Technology\Mid_Presentation\EE136IU_Topic4_Capstone_Presentation.pptx"
prs.save(OUT)
print(f"✅ Done! Saved: {OUT}")
print(f"   Total slides: {len(prs.slides)}")
print(f"   Font body size: {SZ_BODY.pt}pt (projector-ready)")
