"""Persistent review, template extraction and electronic acknowledgement APIs."""
import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Security, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from backend.security.auth import get_api_key, reviewer_identity
from backend.services import review_store as store
from backend.services.inference_service import inference_service
from backend.config import STORAGE_CONFIG
from chest_xpert.report_quality import ClinicalContentError, validate_clinical_sections

router = APIRouter(dependencies=[Security(get_api_key)])


def load(identifier):
    try:
        return store.read(identifier)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Report not found")


def require_clinical_text(sections):
    try:
        validate_clinical_sections(sections, [s['heading'] for s in sections])
    except ClinicalContentError as exc:
        raise HTTPException(422, str(exc))


class Section(BaseModel):
    heading: str = Field(min_length=1, max_length=120)
    text: str = Field(max_length=15000)


class Review(BaseModel):
    sections: list[Section] = Field(min_length=1, max_length=30)
    template_id: str = "standard"
    revision: int


class Signoff(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    license: str = Field(min_length=1, max_length=120)
    confirmed: bool
    revision: int


class TemplateLayout(BaseModel):
    font_size: int = Field(default=12, ge=9, le=20)
    accent: str = Field(default="#164e63", pattern=r"^#[0-9A-Fa-f]{6}$")
    header: str = Field(default="Chest X-ray Report", max_length=300)
    footer: str = Field(default="", max_length=1000)


class TemplateEdit(BaseModel):
    template_name: str = Field(min_length=1, max_length=120)
    sections: list[str] = Field(min_length=1, max_length=30)
    layout: TemplateLayout = Field(default_factory=TemplateLayout)

    @field_validator("sections")
    @classmethod
    def validate_headings(cls, headings):
        headings = [heading.strip() for heading in headings]
        if any(not heading or len(heading) > 120 for heading in headings) or len(set(headings)) != len(headings):
            raise ValueError("Section headings must be unique, nonempty, and at most 120 characters")
        return headings


@router.get('/identity')
def identity(api_key: str | None = Security(get_api_key)):
    return {'reviewer': reviewer_identity(api_key)}


@router.get('/reviews')
def history():
    folder = STORAGE_CONFIG.reports_dir / 'analysis'
    return [{"analysis_id": d["analysis_id"], "patient_id": d.get("patient_id"), "created_at": d["created_at"], "signed": bool(d.get("signature"))} for p in sorted(folder.glob('*.json'), reverse=True) for d in [json.loads(p.read_text())]]


@router.get('/reviews/{identifier}')
def get_review(identifier: str):
    return load(identifier)


@router.put('/reviews/{identifier}')
def save_review(identifier: str, review: Review):
    with store.lock:
        data = load(identifier)
        if review.revision != data['revision']:
            raise HTTPException(409, 'Report changed; reload before saving.')
        try:
            data['template'] = store.template(review.template_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, 'Template not found')
        sections = [s.model_dump() for s in review.sections]
        require_clinical_text(sections)
        data['sections'] = sections
        from chest_xpert.report_quality import report_warnings, report_context_warnings
        data['quality_warnings'] = [w for w in data.get('quality_warnings', [])
                                    if not w.startswith('Conflicting statements about ')
                                    and not w.startswith('Report describes change from a prior study')]
        data['quality_warnings'] += report_warnings(data['sections'])
        data['quality_warnings'] += report_context_warnings(
            data['sections'], next((s['text'] for s in data['sections']
                                    if s['heading'].strip().lower() in {'comparison', 'comparison study'}), None))
        data['signature'] = None
        data['revision'] += 1
        store.write(identifier, data)
        return data


@router.post('/reviews/{identifier}/sign')
def sign(identifier: str, signoff: Signoff, api_key: str | None = Security(get_api_key)):
    identity = reviewer_identity(api_key)
    if identity and (signoff.name.strip() != identity["name"] or signoff.license.strip() != identity["license"]):
        raise HTTPException(403, "Reviewer details must match the authenticated account.")
    if not signoff.confirmed or not signoff.name.strip() or not signoff.license.strip():
        raise HTTPException(400, 'Confirm review and provide name and registration number.')
    with store.lock:
        data = load(identifier)
        if signoff.revision != data['revision']:
            raise HTTPException(409, 'Report changed; reload before signing.')
        require_clinical_text(data['sections'])
        from chest_xpert.report_quality import report_warnings, report_context_warnings
        if report_warnings(data['sections']):
            raise HTTPException(409, 'Reconcile conflicting report statements before signing.')
        comparison = next((s['text'] for s in data['sections']
                           if s['heading'].strip().lower() in {'comparison', 'comparison study'}), None)
        if report_context_warnings(data['sections'], comparison):
            raise HTTPException(409, 'Remove unsupported claims of change before signing.')
        data['signature'] = {'reviewer_id': identity['id'] if identity else None, 'identity_source': 'provisioned_account' if identity else 'self_asserted', 'name': signoff.name.strip(), 'license': signoff.license.strip(), 'signed_at': datetime.now(timezone.utc).isoformat(), 'report_sha256': hashlib.sha256(json.dumps(data['sections'], sort_keys=True).encode()).hexdigest()}
        data['revision'] += 1
        store.write(identifier, data)
        return data


@router.get('/reviews/{identifier}/preview', response_class=HTMLResponse)
def preview(identifier: str):
    data = load(identifier)
    require_clinical_text(data['sections'])
    return store.render(data)


@router.get('/templates')
def list_templates():
    return store.templates()


@router.put('/templates/{identifier}')
def edit_template(identifier: str, edit: TemplateEdit):
    data = edit.model_dump()
    data['id'] = str(uuid4()) if identifier == 'standard' else identifier
    try:
        store.write(data['id'], data, 'template')
    except ValueError:
        raise HTTPException(400, 'Invalid template ID')
    return data


@router.post('/templates/upload')
async def extract_template(file: UploadFile = File(...)):
    from pathlib import Path
    ext = Path(file.filename or '').suffix.lower()
    if ext not in {'.pdf', '.png', '.jpg', '.jpeg'}:
        raise HTTPException(400, 'Upload a scanned PNG/JPG or PDF template.')
    content = await file.read(15 * 1024 * 1024 + 1)
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(413, 'Template must be smaller than 15 MB.')
    identifier = str(uuid4())
    path = STORAGE_CONFIG.temp_dir / f'{identifier}{ext}'
    path.write_bytes(content)
    def extract():
        from PIL import Image
        from chest_xpert.verification import VerificationEngine
        if ext == '.pdf':
            import fitz
            with fitz.open(path) as document:
                if not len(document):
                    raise ValueError('PDF has no pages')
                pix = document[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                image = path.with_suffix('.png')
                pix.save(image)
        else:
            image = path
            with Image.open(image) as opened:
                opened.verify()
        engine = getattr(inference_service, '_verification_engine', None)
        if engine is None:
            engine = VerificationEngine()
            inference_service._verification_engine = engine
        prompt = 'Extract the reusable report template from this scanned hospital report. Ignore patient data, findings, signatures and identifiers. Return ONLY JSON with template_name, sections (ordered list of section heading strings), layout (font_size integer 9-20, accent hex color, header hospital/report title, footer generic footer). Capture visual typography, header and section order. Do not copy any patient information.'
        text = engine.client._call_local_generate(prompt, str(image), None)
        match = re.search(r'\{.*\}', text or '', re.S)
        data = TemplateEdit.model_validate_json(match.group(0) if match else '').model_dump()
        data['id'] = identifier
        data['extraction_note'] = 'First page scanned; editable approximation of typography, header, footer and section order.'
        store.write(identifier, data, 'template')
        return data
    try:
        async with inference_service._semaphore:
            return await asyncio.to_thread(extract)
    except Exception as exc:
        raise HTTPException(422, f'Template extraction failed: {exc}')
    finally:
        path.unlink(missing_ok=True)
        path.with_suffix('.png').unlink(missing_ok=True)


@router.get('/reviews/{identifier}/image/{kind}')
def artifact(identifier: str, kind: str):
    from fastapi.responses import FileResponse
    load(identifier)
    suffix = {"original": "original", "gradcam": "gradcam_contact_sheet"}.get(kind)
    if re.fullmatch(r'original_[2-5]', kind):
        suffix = 'original_view_' + kind.split('_')[1]
    if suffix is None:
        raise HTTPException(404, "Image not found")
    path = STORAGE_CONFIG.temp_dir / f"{identifier}_{suffix}.png"
    if not path.is_file():
        raise HTTPException(404, "Temporary image is no longer available")
    return FileResponse(path, media_type="image/png")
