from chest_xpert.report_quality import report_context_warnings
from chest_xpert.report_quality import parse_generated_sections
from chest_xpert.report import ReportData, generate_html, generate_pdf
from scripts.score_vlm_pair_standard import strict_prediction


def test_comparative_claim_requires_supplied_prior_study():
    sections = [
        {'heading': 'Comparison', 'text': 'NOT PROVIDED'},
        {'heading': 'Findings', 'text': 'Interval placement of a central venous catheter.'},
        {'heading': 'Impression', 'text': 'No pneumothorax.'},
    ]
    assert report_context_warnings(sections, 'NOT PROVIDED')
    assert not report_context_warnings(sections, 'Chest radiograph from yesterday')


def test_noncomparative_report_passes_context_guard():
    sections = [
        {'heading': 'Findings', 'text': 'A right central venous catheter terminates in the SVC.'},
        {'heading': 'Impression', 'text': 'No pneumothorax.'},
    ]
    assert report_context_warnings(sections) == []


def test_vlm_draft_export_omits_internal_evidence(tmp_path):
    report = ReportData()
    report.v7_report = '{"sections":[{"heading":"Findings","text":"A right pleural effusion is present."},{"heading":"Impression","text":"Right pleural effusion."}]}'
    report.contact_sheet_path = '/tmp/private_gradcam.png'
    report.model_name = 'Internal model name'
    html = generate_html(report)
    assert 'Right pleural effusion.' in html
    assert 'AI-generated draft' in html
    for forbidden in ('Grad-CAM', 'CNN', 'Internal model name', 'private_gradcam'):
        assert forbidden not in html
    pdf = tmp_path / 'draft.pdf'
    generate_pdf(report, pdf)
    assert pdf.read_bytes().startswith(b'%PDF')


def test_report_sections_alias_keeps_exact_clinical_text():
    raw = '{"report_sections":[{"heading":"Findings","text":"A right pleural effusion."},{"heading":"Impression","text":"Right pleural effusion."}]}'
    sections, normalized = parse_generated_sections(raw, ['Findings', 'Impression'])
    assert normalized is True
    assert sections[0]['text'] == 'A right pleural effusion.'
    assert strict_prediction(raw, ['Findings', 'Impression']) == 'A right pleural effusion. Right pleural effusion.'


def test_lowercase_heading_keys_keep_exact_clinical_text():
    raw = '{"findings":"A right pleural effusion.","impression":"Right pleural effusion."}'
    sections, normalized = parse_generated_sections(raw, ['Findings', 'Impression'])
    assert normalized is True
    assert [s['text'] for s in sections] == ['A right pleural effusion.', 'Right pleural effusion.']
    assert strict_prediction(raw, ['Findings', 'Impression']) == 'A right pleural effusion. Right pleural effusion.'
