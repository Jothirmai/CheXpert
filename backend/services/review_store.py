"""Disk-backed reports, editable template definitions, and review sign-offs."""
import json
import threading
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from uuid import UUID

from backend.config import STORAGE_CONFIG

lock = threading.RLock()
DEFAULT_TEMPLATE = {"id": "standard", "template_name": "Standard", "sections": ["Clinical indication", "Comparison", "Technique", "Findings", "Impression"], "layout": {"font_size": 12, "accent": "#164e63", "header": "Chest X-ray Report", "footer": "", "columns": 1}}


def location(identifier, kind="analysis"):
    UUID(identifier)
    folder = STORAGE_CONFIG.reports_dir / kind
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{identifier}.json"


def read(identifier, kind="analysis"):
    return json.loads(location(identifier, kind).read_text())


def write(identifier, data, kind="analysis"):
    with lock:
        target = location(identifier, kind)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)


def templates():
    folder = STORAGE_CONFIG.reports_dir / "template"
    return [DEFAULT_TEMPLATE] + [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]


def template(identifier):
    return DEFAULT_TEMPLATE.copy() if identifier == "standard" else read(identifier, "template")


def render(data):
    from chest_xpert.report_quality import validate_clinical_sections
    validate_clinical_sections(data["sections"], [s["heading"] for s in data["sections"]])
    t = data.get("template", DEFAULT_TEMPLATE)
    layout = t.get("layout", {})
    font = max(9, min(20, int(layout.get("font_size", 12))))
    accent = layout.get("accent", "#164e63")
    import re
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", accent):
        accent = "#164e63"
    body = "".join(f"<section><h2>{escape(s['heading'])}</h2><p>{escape(s['text'])}</p></section>" for s in data["sections"])
    signed = data.get("signature")
    sign = f"Checked by: {escape(signed['name'])} · {escape(signed['license'])}<br>Electronically signed {escape(signed['signed_at'])}" if signed else "Checked by: ____________________<br>Electronic sign-off pending"
    patient = escape(str(data.get("patient_id") or "Not provided"))
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>Chest X-ray Report</title><style>@page{{size:A4;margin:18mm}}body{{font:{font}pt Arial;color:#17202a;max-width:780px;margin:auto}}h1,h2{{color:{accent}}}h1{{border-bottom:2px solid {accent}}}h2{{font-size:1.05em}}p{{white-space:pre-wrap;line-height:1.5}}section{{break-inside:avoid}}footer{{margin-top:32px;border-top:1px solid #aaa;padding-top:12px}}.draft{{color:#946200}}</style></head><body><h1>{escape(str(layout.get('header', 'Chest X-ray Report')))}</h1><p>Patient ID: {patient}</p><p class="draft">{'Reviewed and signed' if signed else 'AI-generated draft — pending radiologist review'}</p>{body}<footer>{sign}<p>{escape(str(layout.get('footer', '')))}</p></footer></body></html>'''
