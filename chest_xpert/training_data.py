"""Clinical-only IU X-ray parsing and study-isolated multimodal training manifests."""
from __future__ import annotations
import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

HEADINGS = ['Clinical indication', 'Comparison', 'Technique', 'Findings', 'Impression']
ALLOWED = {'INDICATION', 'CLINICAL HISTORY', 'COMPARISON', 'TECHNIQUE', 'FINDINGS', 'IMPRESSION'}


def clean_clinical(text):
    # Remove publication boilerplate even if accidentally embedded in a clinical field.
    text = re.sub(r'https?://\S+|\bdoi\s*:\s*\S+', '', text, flags=re.I)
    sentences = re.split(r'(?<=[.!?])\s+|[\r\n]+', text)
    sentences = [s for s in sentences if not re.search(r'\b(copyright|published by|all rights reserved|supplied by publisher|creative commons|chest x.ray collection)\b', s, re.I)]
    return re.sub(r'\s+', ' ', ' '.join(sentences)).strip()


def parse_clinical_xml(path):
    root = ET.parse(path).getroot()
    sections = {}
    for node in root.findall('.//AbstractText'):
        heading = node.get('Label', '').upper().strip()
        if heading in ALLOWED:
            text = clean_clinical(' '.join(node.itertext()))
            if text:
                sections[heading] = (sections.get(heading, '') + ' ' + text).strip()
    # Captions are never copied into the target. Only explicitly stated view
    # descriptions are mapped into controlled acquisition metadata.
    if not sections.get('TECHNIQUE'):
        caption = ' '.join(' '.join(node.itertext()) for node in root.findall('.//parentImage/caption'))
        if re.search(r'\bPA\s*(?:and|&)\s*lateral\b', caption, re.I):
            sections['TECHNIQUE'] = 'PA and lateral chest radiographs.'
        elif re.search(r'\bAP\s*(?:and|&)\s*lateral\b', caption, re.I):
            sections['TECHNIQUE'] = 'AP and lateral chest radiographs.'
        elif re.search(r'\b(?:two|2)[ -]views?\b', caption, re.I):
            sections['TECHNIQUE'] = 'Two-view chest radiographs.'
        elif re.search(r'\bportable\s+(?:AP\s+)?chest\b|\bAP\s+portable\b', caption, re.I):
            sections['TECHNIQUE'] = 'Portable chest radiograph.'
    uid = root.find('uId')
    study_id = uid.get('id') if uid is not None else 'CXR' + Path(path).stem
    return {'study_id':study_id, 'sections':sections,
            'image_ids':[node.get('id') for node in root.findall('parentImage') if node.get('id')],
            'mesh_terms':[clean_clinical(' '.join(node.itertext())) for node in root.findall('.//MeSH/*')]}


def clinical_report_text(record):
    return '\n'.join(f'{key.title()}: {record["sections"][key]}' for key in ['FINDINGS','IMPRESSION'] if record['sections'].get(key))


def structured_target(sections):
    lookup = {'Clinical indication':sections.get('INDICATION') or sections.get('CLINICAL HISTORY'),
              'Comparison':sections.get('COMPARISON'), 'Technique':sections.get('TECHNIQUE'),
              'Findings':sections.get('FINDINGS'), 'Impression':sections.get('IMPRESSION')}
    # Redactions remain explicit; never guess missing anatomy or patient information.
    return {'sections':[{'heading':heading,'text':re.sub(r'\bX{2,}\b','[REDACTED]',lookup[heading] or 'NOT PROVIDED')} for heading in HEADINGS]}


def prepare(root, output, seed=42):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    def decoded(path):
        try:
            with Image.open(path) as image:
                image.load()
            return path
        except (OSError, ValueError, Image.DecompressionBombError):
            return None
    candidates = [p for p in (root/'images').rglob('*.png') if p.stat().st_size]
    with ThreadPoolExecutor(max_workers=8) as pool:
        checked = list(pool.map(decoded, candidates))
    invalid_images = [str(path.relative_to(root)) for path, valid in zip(candidates, checked) if valid is None]
    images = {p.stem:p.resolve() for p in checked if p is not None}
    by_study = defaultdict(list)
    for key, path in images.items():
        match = re.match(r'CXR\d+', key)
        if match:by_study[match.group()].append(path)
    studies=[]; counts=Counter(); lengths=[]; fields=Counter()
    for path in sorted((root/'reports').rglob('*.xml')):
        counts['source_xml_reports']+=1
        try:record=parse_clinical_xml(path)
        except ET.ParseError:
            counts['invalid_xml']+=1
            continue
        sections=record['sections']; fields.update(sections.keys())
        if not sections.get('FINDINGS') or not sections.get('IMPRESSION'):
            counts['excluded_missing_findings_or_impression']+=1
            continue
        paired = [images[key] for key in record['image_ids'] if key in images] or sorted(by_study.get(record['study_id'], []))
        if not paired:
            counts['excluded_no_image']+=1
            continue
        lengths.append(len(clinical_report_text(record).split()))
        studies.append({**record, 'image_paths':[str(p) for p in paired], 'target':structured_target(sections)})
    random.Random(seed).shuffle(studies)
    # One split per study, including every associated image. Never randomly split images.
    n=len(studies); splits={'train':studies[:int(.7*n)],'validation':studies[int(.7*n):int(.9*n)],'test':studies[int(.9*n):]}
    seen_hashes={}; summary={}
    for split, records in splits.items():
        rows=[]
        for record in records:
            for image in record['image_paths']:
                digest=hashlib.sha256(Path(image).read_bytes()).hexdigest()
                if digest in seen_hashes and seen_hashes[digest]!=split:
                    counts['excluded_duplicate_image_across_splits']+=1
                    continue
                seen_hashes[digest]=split
                rows.append({'study_id':record['study_id'],'image_path':image,'image_sha256':digest,
                             'clinical_indication':re.sub(r'\bX{2,}\b','[REDACTED]',record['sections'].get('INDICATION') or record['sections'].get('CLINICAL HISTORY') or '') or None,
                             'comparison_study':re.sub(r'\bX{2,}\b','[REDACTED]',record['sections'].get('COMPARISON') or '') or None, 'technique':record['sections'].get('TECHNIQUE'),
                             'target':record['target'], 'mesh_terms':record['mesh_terms']})
        (output/f'{split}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
        summary[split]={'studies':len({r['study_id'] for r in rows}),'images':len(rows)}
    counts['excluded_corrupt_images'] = len(invalid_images)
    audit={'excluded_corrupt_image_files':invalid_images,'counts':dict(counts),'nonempty_source_sections':dict(fields),'splits':summary,
           'clinical_report_words':{'median':sorted(lengths)[len(lengths)//2] if lengths else 0,'max':max(lengths,default=0)},
           'seed':seed,'excluded_fields':['publisher','authors','journal','article title','license','URLs','affiliation','arbitrary image-caption text and extraction metadata (explicit view descriptors mapped separately)'],
           'limitations':['Original reports may describe multiple views; individual image examples may not show every reported finding.','Redacted source words stay REDACTED; technique is not invented.','Study identifiers provide grouping; distinct patient linkage is not available.']}
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    return audit


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='data/iu_xray');parser.add_argument('--output',default='data/report_training')
    args=parser.parse_args()
    print(json.dumps(prepare(args.root,args.output),indent=2))
