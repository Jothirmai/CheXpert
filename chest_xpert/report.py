"""Structured, template-based report generation for Chest Xpert.

Reports are grounded strictly in model outputs and user-supplied information.
The generator never invents clinical observations. Every report is stamped as an
AI-generated research draft requiring review by a qualified radiologist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    HRFlowable,
)

from chest_xpert.config import APP_TITLE, DISCLAIMER
from chest_xpert.label_mapping import LabelMapping
from chest_xpert.verification import VerificationResult, VerificationItem, ReportFinding

# ---------------------------------------------------------------------------
# Report disclaimer
# ---------------------------------------------------------------------------
DISCLAIMER_TEXT = (
    f"{APP_TITLE} — AI-GENERATED RESEARCH DRAFT. "
    "Requires review by a qualified radiologist. NOT a final clinical diagnosis. "
    "Prediction scores are not calibrated probabilities or clinical confidence. "
    + DISCLAIMER
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ExamDetails:
    patient_id: Optional[str] = None
    exam_date: Optional[str] = None
    modality: str = "Chest X-ray"
    view: Optional[str] = None
    institution: Optional[str] = None
    referring_physician: Optional[str] = None
    device: Optional[str] = None


@dataclass
class ReportData:
    exam: ExamDetails = field(default_factory=ExamDetails)
    model_name: str = ""
    model_version: str = ""
    device: str = ""
    latency_ms: float = 0.0
    inference_only_ms: float = 0.0
    modality_ms: float = 0.0
    verification_ms: float = 0.0
    mappings: List[LabelMapping] = field(default_factory=list)
    unsupported_labels: List[str] = field(default_factory=list)
    modality_accepted: Optional[bool] = None
    modality_decision: Optional[str] = None
    modality_reason: Optional[str] = None
    modality_model: Optional[str] = None
    verification_result: Optional[VerificationResult] = None
    verification_available: bool = False
    verification_error: Optional[str] = None
    image_quality_note: Optional[str] = None
    heatmap_path: Optional[str] = None
    heatmap_label: Optional[str] = None
    impression: str = ""
    reviewer_name: Optional[str] = None
    review_status: str = "Pending"
    notes: str = ""
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


# ---------------------------------------------------------------------------
# Severity scoring helper
# ---------------------------------------------------------------------------

def _severity(score: Optional[float]) -> str:
    """Map classifier score to a qualitative severity level."""
    if score is None:
        return "not assessed"
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "not assessed"
    if score >= 0.7:
        return "high"
    if score >= 0.5:
        return "moderate"
    if score >= 0.3:
        return "low"
    return "minimal"


_SEVERITY_COLOR = {
    "high": "#dc2626",
    "moderate": "#ea580c",
    "low": "#ca8a04",
    "minimal": "#16a34a",
    "not assessed": "#6b7280",
}


def _severity_badge(score: Optional[float]) -> str:
    sev = _severity(score)
    color = _SEVERITY_COLOR[sev]
    label = sev.capitalize()
    return (
        f'<span class="badge" style="background:{color}15;color:{color};'
        f'border:1px solid {color};">{label}</span>'
    )


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _supported_rows(mappings: List[LabelMapping]) -> List[List[str]]:
    rows = [["Finding", "Score", "Severity", "Status"]]
    for m in mappings:
        if m.supported:
            score = f"{m.score:.4f}" if m.score is not None else "n/a"
            sev = _severity_badge(m.score)
            rows.append([m.requested_label, score, sev, "Supported"])
        else:
            rows.append([m.requested_label, "—", "—", "Unsupported"])
    return rows


def _verification_rows(vr: Optional[VerificationResult]) -> List[List[str]]:
    rows = [["Pathology", "Score", "LLM Assessment", "Rationale"]]
    if vr is None or not vr.available or not vr.items:
        rows.append(["—", "—", "Not available", "LLM verification was not completed."])
        return rows
    for it in vr.items:
        score = f"{it.classifier_score:.4f}" if it.classifier_score is not None else "—"
        rows.append([it.pathology, score, it.assessment, it.rationale])
    return rows


def generate_html(data: ReportData) -> str:
    """Render a professional, self-contained HTML radiology report."""

    # ---------- Finding rows ----------
    rows = _supported_rows(data.mappings)
    table_html = "\n".join(
        f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>"
        for r in rows[1:]
    )

    # ---------- Heatmap ----------
    heatmap_html = ""
    if data.heatmap_path and data.heatmap_label:
        heatmap_html = (
            "<div class='section'>"
            "<h2>Visual Explanation — Grad-CAM</h2>"
            f"<p><b>Target finding:</b> {data.heatmap_label}. "
            "Highlighted regions are model explanations, not proof of disease.</p>"
            f'<img src="{data.heatmap_path}" class="gradcam" alt="Grad-CAM heatmap" />'
            "</div>"
        )

    # ---------- Modality ----------
    modality_html = ""
    if data.modality_decision is not None:
        mod_cls = {
            "accepted": "#166534",
            "rejected": "#991b1b",
            "uncertain": "#92400e",
            "technical_failure": "#6b7280",
        }.get(data.modality_decision, "#6b7280")
        modality_html = (
            "<div class='section'>"
            "<h2>Stage 0 — Image &amp; Modality Validation</h2>"
            f"<p><b>Decision:</b> "
            f'<span class="badge" style="background:{mod_cls}15;color:{mod_cls};'
            f'border:1px solid {mod_cls};">{data.modality_decision}</span></p>'
            f"<p><b>Reason:</b> {data.modality_reason or '—'}</p>"
            f"<p><b>Model:</b> {data.modality_model or '—'} | "
            f"Latency: {data.modality_ms:.1f} ms</p>"
            "</div>"
        )

    # ---------- Verification ----------
    vr = data.verification_result
    vrows = _verification_rows(vr)
    vtable_html = "\n".join(
        f"<tr><td>{r[0]}</td><td>{r[1]}</td>"
        f"<td>{_verification_badge(r[2])}</td><td>{r[3]}</td></tr>"
        for r in vrows[1:]
    )
    verification_html = ""
    if data.verification_available or vr is not None:
        n_consistent = sum(1 for it in (vr.items or []) if it.assessment == "consistent") if vr else 0
        n_inconsistent = sum(1 for it in (vr.items or []) if it.assessment == "inconsistent") if vr else 0
        n_uncertain = sum(1 for it in (vr.items or []) if it.assessment == "uncertain") if vr else 0
        verification_html = (
            "<div class='section'>"
            "<h2>Stage B — Prediction Verification (LLM #1)</h2>"
            f"<p><b>Available:</b> {'Yes' if data.verification_available else 'No'}</p>"
            f"<p><b>Model:</b> {vr.model_id if vr else '—'} | "
            f"Latency: {data.verification_ms:.1f} ms</p>"
        )
        if vr and vr.image_assessment:
            verification_html += f"<p><b>Image assessment:</b> {vr.image_assessment}</p>"
        if vr and vr.error:
            verification_html += f"<p><b>Note:</b> {vr.error}</p>"
        verification_html += (
            f"<p><b>Agreement:</b> {n_consistent} consistent · "
            f"{n_inconsistent} inconsistent · {n_uncertain} uncertain</p>"
            f"<table class='data-table'><thead><tr>"
            f"<th>Pathology</th><th>Score</th><th>LLM Assessment</th><th>Rationale</th>"
            f"</tr></thead><tbody>{vtable_html}</tbody></table>"
            f"<p class='subtitle'>LLM verification is a secondary model-generated "
            f"assessment, not ground truth or a substitute for a radiologist.</p>"
            "</div>"
        )

    # ---------- Evidence-based findings ----------
    findings_html = ""
    if vr and vr.report_findings:
        rows_html = "\n".join(
            f"<tr><td>{f.pathology}</td><td>{f.status}</td><td>{f.recommended_wording or '—'}</td></tr>"
            for f in vr.report_findings
        )
        findings_html = (
            "<div class='section'>"
            "<h2>Stage C — Evidence-Based Findings (LLM)</h2>"
            "<p>Findings are only reported when supported by available evidence "
            "(classifier score + verification + GradCAM), not based on high scores alone.</p>"
            f"<table class='data-table'><thead><tr>"
            f"<th>Finding</th><th>Evidence-Based Status</th><th>Suggested Wording</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>"
            "<p class='subtitle'>Findings are only reported when supported by "
            "available evidence, not based on high scores alone.</p>"
            "</div>"
        )
    elif data.verification_available:
        findings_html = (
            "<div class='section'>"
            "<h2>Stage C — Evidence-Based Findings (LLM)</h2>"
            "<p><em>No evidence-based findings generated. Classifier scores are "
            "available above but were not interpreted as definite findings "
            "without supporting evidence.</em></p>"
            "</div>"
        )

    # ---------- Full HTML ----------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{APP_TITLE} — Radiology Report</title>
<style>
@page {{ size: A4; margin: 15mm 15mm 15mm 15mm; }}
* {{ box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    color: #1f2937;
    line-height: 1.5;
    font-size: 11px;
    margin: 0;
    padding: 0;
}}
.header-bar {{
    background: #0f172a;
    color: #ffffff;
    padding: 20px;
    border-radius: 8px 8px 0 0;
    margin-bottom: 20px;
}}
.header-bar h1 {{
    margin: 0;
    font-size: 22px;
    letter-spacing: 0.5px;
}}
.header-bar .subtitle {{
    margin: 4px 0 0 0;
    font-size: 11px;
    color: #94a3b8;
}}
.report-meta {{
    display: flex;
    gap: 20px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}}
.meta-card {{
    flex: 1;
    min-width: 200px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 12px;
}}
.meta-card h3 {{
    margin: 0 0 6px 0;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b;
}}
.meta-card p {{
    margin: 0;
    font-size: 12px;
    color: #334155;
    font-weight: 500;
}}
.section {{
    margin-bottom: 20px;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    overflow: hidden;
}}
.section h2 {{
    margin: 0;
    padding: 12px 16px;
    background: #f1f5f9;
    font-size: 13px;
    color: #334155;
    border-bottom: 1px solid #e2e8f0;
}}
.section-content {{
    padding: 16px;
}}
.findings-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    margin-top: 12px;
}}
.finding-card {{
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 12px;
    background: #ffffff;
}}
.finding-card h4 {{
    margin: 0 0 6px 0;
    font-size: 12px;
    color: #0f172a;
}}
.finding-card .score {{
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 4px;
}}
.finding-card .sev-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
}}
.data-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 12px;
    font-size: 11px;
}}
.data-table th {{
    background: #f1f5f9;
    color: #475569;
    font-weight: 600;
    padding: 8px 10px;
    text-align: left;
    border-bottom: 2px solid #e2e8f0;
}}
.data-table td {{
    padding: 8px 10px;
    border-bottom: 1px solid #f1f5f9;
    vertical-align: top;
}}
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
}}
.gradcam {{
    max-width: 100%;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    margin-top: 10px;
}}
.disclaimer {{
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-radius: 6px;
    padding: 12px;
    margin-top: 20px;
    font-size: 10px;
    color: #92400e;
}}
.print-only {{ display: none; }}
@media print {{
    body {{ padding: 0; font-size: 10px; }}
    .no-print {{ display: none; }}
    .print-only {{ display: block; }}
    .section {{ break-inside: avoid; }}
    .header-bar {{ background: #0f172a !important; -webkit-print-color-adjust: exact; }}
}}
</style>
</head>
<body>

<!-- Header Bar -->
<div class="header-bar">
    <h1>{APP_TITLE}</h1>
    <p class="subtitle">Chest X-ray Radiology Report &nbsp;·&nbsp; AI-Generated Research Draft</p>
</div>

<!-- Patient & Exam Meta -->
<div class="report-meta">
    <div class="meta-card">
        <h3>Patient ID</h3><p>{data.exam.patient_id or '—'}</p>
    </div>
    <div class="meta-card">
        <h3>Exam Date</h3><p>{data.exam.exam_date or '—'}</p>
    </div>
    <div class="meta-card">
        <h3>Modality</h3><p>{data.exam.modality}</p>
    </div>
    <div class="meta-card">
        <h3>View</h3><p>{data.exam.view or '—'}</p>
    </div>
    <div class="meta-card">
        <h3>Institution</h3><p>{data.exam.institution or '—'}</p>
    </div>
    <div class="meta-card">
        <h3>Referring Physician</h3><p>{data.exam.referring_physician or '—'}</p>
    </div>
</div>

{modality_html}

<!-- Stage A: Classifier Scores -->
<div class="section">
<h2>Stage A — Disease Classification (Pretrained Model)</h2>
<div class="section-content">
<p class="subtitle">
    Model: <b>{data.model_name}</b> &nbsp;|&nbsp; Version: <b>{data.model_version or 'pre-trained weights'}</b> &nbsp;|&nbsp;
    Device: <b>{data.device or '—'}</b> &nbsp;|&nbsp; Latency: {data.latency_ms:.1f} ms
</p>
<p class="subtitle">Scores are raw model outputs, not calibrated probabilities or clinical confidence.</p>
<div class="findings-grid">
{f"".join(_finding_card(r) for r in rows[1:])}
</div>
</div>
</div>

{heatmap_html}

{verification_html}

{findings_html}

<!-- Impression -->
<div class="section">
<h2>Impression (Draft)</h2>
<div class="section-content">
<p>{data.impression or 'Impression pending radiologist review.'}</p>
</div>
</div>

<!-- Reviewer -->
<div class="section no-print">
<h2>Reviewer</h2>
<div class="section-content">
<p><b>Reviewer name:</b> {data.reviewer_name or '—'}</p>
<p><b>Review status:</b> {data.review_status}</p>
<p><b>Notes:</b> {data.notes or '—'}</p>
</div>
</div>

<!-- Disclaimer -->
<div class="disclaimer">
<b>DISCLAIMER:</b> {DISCLAIMER_TEXT}
</div>

</body>
</html>"""

    return html


def _finding_card(row: List[str]) -> str:
    """Render a single finding as a card."""
    label, score, sev_html, status = row
    score_val = None
    try:
        score_val = float(score)
    except (TypeError, ValueError):
        score_val = None
    sev_label = _severity(score_val).capitalize() if score_val is not None else "Not assessed"
    status_color = "#166534" if status == "Supported" else "#6b7280"
    return (
        f"<div class='finding-card'>"
        f"<h4>{label}</h4>"
        f"<div class='score'>{score}</div>"
        f"<div class='sev-badge' style='background:{_SEVERITY_COLOR.get(_severity(score_val), '#6b7280')}15;color:{_SEVERITY_COLOR.get(_severity(score_val), '#6b7280')};'>{sev_label}</div>"
        f"<p style='margin:6px 0 0 0;font-size:10px;color:{status_color};'>{status}</p>"
        f"</div>"
    )


def _verification_badge(assessment: str) -> str:
    color = {
        "consistent": "#166534",
        "inconsistent": "#991b1b",
        "uncertain": "#92400e",
    }.get(assessment, "#6b7280")
    return (
        f'<span class="badge" style="background:{color}15;color:{color};border:1px solid {color};">'
        f"{assessment}</span>"
    )


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def generate_pdf(data: ReportData, output_path: str | Path) -> Path:
    """Generate a PDF report via reportlab."""
    output_path = Path(output_path)
    styles = getSampleStyleSheet()
    if "Subtitle" not in styles.byName:
        styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#64748b")))
    if "FindingTitle" not in styles.byName:
        styles.add(ParagraphStyle(name="FindingTitle", parent=styles["Normal"], fontSize=9, fontName="Helvetica-Bold", textColor=colors.HexColor("#0f172a")))
    if "FindingScore" not in styles.byName:
        styles.add(ParagraphStyle(name="FindingScore", parent=styles["Normal"], fontSize=11, fontName="Helvetica-Bold", textColor=colors.HexColor("#334155")))
    if "FindingStatus" not in styles.byName:
        styles.add(ParagraphStyle(name="FindingStatus", parent=styles["Normal"], fontSize=7, textColor=colors.HexColor("#64748b")))

    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                            leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=12*mm, bottomMargin=12*mm)
    story = []

    def P(text, style="Normal"):
        return Paragraph(text, styles[style])

    # Header
    story.append(P(f"<b>{APP_TITLE}</b>", "Heading1"))
    story.append(P("Chest X-ray Radiology Report — AI-Generated Research Draft", "Subtitle"))
    story.append(Spacer(1, 8))

    # Patient meta table
    meta_rows = [
        ["Patient ID:", data.exam.patient_id or "—", "Exam Date:", data.exam.exam_date or "—"],
        ["Modality:", data.exam.modality, "View:", data.exam.view or "—"],
        ["Institution:", data.exam.institution or "—", "Referring Physician:", data.exam.referring_physician or "—"],
        ["Device:", data.device or "—", "Generated:", data.generated_at or "—"],
    ]
    meta_table = Table(meta_rows, colWidths=[28*mm, 42*mm, 28*mm, 52*mm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7.5),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#475569")),
        ("TEXTCOLOR", (2,0), (2,-1), colors.HexColor("#475569")),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING", (0,0), (-1,-1), 2),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Stage 0 modality
    if data.modality_decision is not None:
        story.append(P("Stage 0 — Image &amp; Modality Validation", "Heading2"))
        story.append(P(f"<b>Decision:</b> {data.modality_decision}"))
        story.append(P(f"<b>Reason:</b> {data.modality_reason or '—'}"))
        story.append(P(f"<b>Model:</b> {data.modality_model or '—'} | Latency: {data.modality_ms:.1f} ms"))
        story.append(Spacer(1, 6))

    # Stage A: Classifier findings as cards/table
    story.append(P("Stage A — Disease Classification (Pretrained Model)", "Heading2"))
    story.append(P(f"Model: <b>{data.model_name}</b> | Version: <b>{data.model_version or 'pre-trained weights'}</b> | "
                   f"Device: <b>{data.device or '—'}</b> | Latency: {data.latency_ms:.1f} ms"))
    story.append(P("Scores are raw model outputs, not calibrated probabilities or clinical confidence.", "Italic"))
    story.append(Spacer(1, 6))

    rows = [["Finding", "Score", "Severity", "Status"]]
    for m in data.mappings:
        if m.supported:
            score = f"{m.score:.4f}" if m.score is not None else "n/a"
            rows.append([m.requested_label, score, _severity(score).capitalize() if m.score is not None else "n/a", "Supported"])
        else:
            rows.append([m.requested_label, "—", "—", "Unsupported"])
    t = Table(rows, colWidths=[50*mm, 20*mm, 30*mm, 40*mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("FONTSIZE", (0,0), (-1,-1), 7.5),
        ("TOPPADDING", (0,0), (-1,-1), 2.5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2.5),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # GradCAM
    if data.heatmap_path and data.heatmap_label and Path(data.heatmap_path).exists():
        story.append(P("Visual Explanation — Grad-CAM", "Heading2"))
        story.append(P(f"Target finding: <b>{data.heatmap_label}</b>. "
                       "Highlighted regions are model explanations, not proof of disease.", "Italic"))
        img = RLImage(str(data.heatmap_path), width=75*mm, height=75*mm)
        story.append(img)
        story.append(Spacer(1, 8))

    # Stage B verification
    vr = data.verification_result
    if data.verification_available or vr is not None:
        story.append(P("Stage B — Prediction Verification (LLM)", "Heading2"))
        story.append(P(f"Available: <b>{'Yes' if data.verification_available else 'No'}</b>"))
        if vr:
            story.append(P(f"Model: <b>{vr.model_id}</b> | Latency: {data.verification_ms:.1f} ms"))
            if vr.image_assessment:
                story.append(P(f"Image assessment: {vr.image_assessment}"))
            if vr.error:
                story.append(P(f"Note: {vr.error}"))
            n_consistent = sum(1 for it in vr.items if it.assessment == "consistent")
            n_inconsistent = sum(1 for it in vr.items if it.assessment == "inconsistent")
            n_uncertain = sum(1 for it in vr.items if it.assessment == "uncertain")
            story.append(P(f"Agreement: {n_consistent} consistent · {n_inconsistent} inconsistent · {n_uncertain} uncertain"))
            vrows = [["Pathology", "Score", "Assessment", "Rationale"]]
            for it in vr.items:
                score = f"{it.classifier_score:.4f}" if it.classifier_score is not None else "—"
                vrows.append([it.pathology, score, it.assessment, it.rationale])
            vtable = Table(vrows, colWidths=[30*mm, 18*mm, 25*mm, 77*mm], repeatRows=1)
            vtable.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f1f5f9")),
                ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
                ("FONTSIZE", (0,0), (-1,-1), 7),
                ("TOPPADDING", (0,0), (-1,-1), 2),
                ("BOTTOMPADDING", (0,0), (-1,-1), 2),
            ]))
            story.append(vtable)
            story.append(P("LLM verification is a secondary model-generated assessment, not ground truth.", "Italic"))
        story.append(Spacer(1, 6))

    # Stage C findings
    if vr and vr.report_findings:
        story.append(P("Stage C — Evidence-Based Findings (LLM)", "Heading2"))
        rows = [["Finding", "Status", "Suggested Wording"]]
        for f in vr.report_findings:
            rows.append([f.pathology, f.status, f.recommended_wording or "—"])
        table = Table(rows, colWidths=[35*mm, 25*mm, 90*mm], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f1f5f9")),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ("FONTSIZE", (0,0), (-1,-1), 7),
            ("TOPPADDING", (0,0), (-1,-1), 2),
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ]))
        story.append(table)
        story.append(P("Findings are only reported when supported by available evidence "
                       "(classifier score + verification + GradCAM), not based on high scores alone.", "Italic"))
        story.append(Spacer(1, 6))

    # Impression
    story.append(P("Impression (Draft)", "Heading2"))
    story.append(P(data.impression or "Impression pending radiologist review."))
    story.append(Spacer(1, 8))

    # Reviewer
    story.append(P("Reviewer", "Heading2"))
    story.append(P(f"<b>Reviewer name:</b> {data.reviewer_name or '—'}"))
    story.append(P(f"<b>Review status:</b> {data.review_status}"))
    story.append(P(f"<b>Notes:</b> {data.notes or '—'}"))
    story.append(Spacer(1, 10))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 4))
    story.append(P(f"<b>DISCLAIMER:</b> {DISCLAIMER_TEXT}", "Italic"))

    doc.build(story)
    return output_path
