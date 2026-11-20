"""Clinical output checks; publication fields never belong in a patient report."""
import re

PUBLICATION = re.compile(r'https?://|\bdoi\s*:|\bcopyright\b|all rights reserved|published by|supplied by publisher|creative commons|Indiana University Chest X.ray Collection|\bMedlineCitation\b|\bArticleTitle\b', re.I)


class ClinicalContentError(ValueError):
    """Clinical text contains implementation or publication material."""


TECHNICAL_METADATA = re.compile(r"\bQwen\b|\bDenseNet\b|torchxrayvision|Grad[\s_\-\u2010-\u2015]*CAM|\bCNN\b|\bVLM\b|classifier scores?|model documentation|heat[- ]?maps?|saliency|model attention|attention maps?|class activation", re.I)

def validate_clinical_sections(sections, expected):
    if [section['heading'] for section in sections] != expected:
        raise ValueError('Generated report sections do not match the selected template')
    for section in sections:
        if TECHNICAL_METADATA.search(section['text']):
            raise ClinicalContentError('Clinical report contains model metadata; describe radiographic findings without model or heatmap terminology. Please revise or regenerate.')
        if PUBLICATION.search(section['text']):
            raise ClinicalContentError('Generated report contains publication metadata and cannot be shown as a clinical draft; please retry')
    return sections


def evidence_warnings(scores):
    values=[value for value in scores.values() if value is not None]
    if values and max(values)<1e-4:
        return ['CNN outputs are uniformly very low. This model signal cannot exclude disease; review the original image independently.']
    return []



def normalize_assertion_headings(sections):
    """Keep a JSON field label from becoming an asserted positive finding.

    For lexical checks only: 'Effusion: No effusion' is a negative statement.
    Original report text is never rewritten by this helper.
    """
    from chest_xpert.disease_targets import DISEASES, PATTERNS
    patterns = {name.casefold(): pattern for name,pattern in zip(DISEASES,PATTERNS)}
    labels = {label.casefold() for label in DISEASES} | {'opacity','granuloma','calcification','fracture','hyperinflation'}
    result=[]
    for section in sections:
        lines=[]
        for line in section['text'].splitlines():
            label,sep,description=line.partition(':')
            if sep and label.strip().casefold() in labels and description.strip():
                pattern = patterns.get(label.strip().casefold(), re.escape(label.strip()))
                if re.search(pattern, description, re.I):
                    lines.append(description.strip())
                else:
                    lines.append(description.strip().rstrip('.;')+' ('+label.strip()+').')
            else:
                lines.append(line)
        result.append({**section,'text':'\n'.join(lines)})
    return result

def report_warnings(sections, scorer_version="v2"):
    """Flag contradictory assertions for human review, without diagnosing."""
    if scorer_version == "v2":
        from chest_xpert.report_scoring_v2 import extract
        return [f'Conflicting statements about {fact} within or between report sections; reconcile before signing.'
                for fact,value in extract(sections).items() if value['state']=='conflicting']
    if scorer_version != "v1":
        raise ValueError('Unknown report warning scorer version')
    from chest_xpert.disease_targets import DISEASES, disease_targets
    states = []
    for section in normalize_assertion_headings(sections):
        if section['heading'].strip().lower() in {'findings', 'impression'}:
            values, known = disease_targets({'target': {'sections': [section]}})
            states.append((values, known))
    warnings = []
    for i, label in enumerate(DISEASES):
        observed = {values[i] for values, known in states if known[i]}
        if 0.0 in observed and 1.0 in observed:
            warnings.append(f'Conflicting statements about {label} between report sections; reconcile before signing.')
    return warnings


COMPARATIVE_LANGUAGE = re.compile(
    r"\b(?:interval|compared? (?:with|to)|since (?:the )?(?:prior|previous)|"
    r"previous(?:ly)?|prior (?:study|exam|radiograph|film)|"
    r"new(?:ly)?|unchanged|stable|improv(?:ed|ing|ement)|"
    r"worsen(?:ed|ing)|resolv(?:ed|ing|ution))\b", re.I
)


def report_context_warnings(sections, comparison_study=None):
    """Flag claims about change when no prior examination was supplied."""
    if comparison_study and comparison_study.strip().upper() != 'NOT PROVIDED':
        return []
    clinical = ' '.join(s['text'] for s in sections
                        if s['heading'].strip().lower() in {'findings', 'impression'})
    if COMPARATIVE_LANGUAGE.search(clinical):
        return ['Report describes change from a prior study, but no comparison study was supplied; remove or substantiate the claim before signing.']
    return []


def parse_generated_sections(text, expected):
    """Normalize alternate JSON layouts without inventing clinical text.

    Only requested headings are accepted; every heading must already be present.
    The returned flag records whether an alternate layout needed normalization.
    """
    import json
    match = re.search(r'\{.*\}', text or '', re.S)
    if not match:
        raise ValueError('Vision model did not return a JSON report')
    data = json.loads(match.group())
    normalized = 'sections' not in data
    if 'sections' in data:
        sections = data['sections']
    elif set(data) == {'report_sections'} and isinstance(data['report_sections'], list):
        sections = data['report_sections']
    else:
        keys = {heading.lower() for heading in expected}
        if set(data) == keys and all(isinstance(data[key], str) for key in keys):
            sections = [{'heading': heading, 'text': data[heading.lower()]}
                        for heading in expected]
        elif set(data) != set(expected):
            raise ValueError('Generated report is missing headings or contains unexpected fields')
        else:
            sections = []
            for heading in expected:
                value = data[heading]
                if isinstance(value, dict) and value and all(isinstance(k, str) and isinstance(v, str) for k,v in value.items()):
                    value = '\n'.join(f'{key}: {description}' for key,description in value.items())
                if not isinstance(value, str):
                    raise ValueError('Unsupported clinical report section content')
                sections.append({'heading':heading, 'text':value})
    from backend.api.v1.workflow import Section
    sections = [Section.model_validate(section).model_dump() for section in sections]
    return validate_clinical_sections(sections, expected), normalized
