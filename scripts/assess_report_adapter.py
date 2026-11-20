"""Conservative validation report checks; never claims clinical accuracy."""
import argparse,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from chest_xpert.disease_targets import DISEASES,disease_targets,NEG,UNCERTAIN
from chest_xpert.report_quality import validate_clinical_sections


def parsed(text):
    match=re.search(r'\{.*\}',text,re.S)
    if not match:raise ValueError('No JSON object')
    return json.loads(match.group())['sections']


EXTRA_FACTS={'Opacity':r'opacit(?:y|ies)|airspace disease','Granuloma':r'granuloma\w*','Calcification':r'calcifi\w*','Fracture':r'fractur\w*','Hyperinflation':r'hyperinflation|hyperinflated'}


def fact_states(sections):
    from chest_xpert.report_quality import normalize_assertion_headings
    sections=normalize_assertion_headings(sections)
    target,known=disease_targets({'target':{'sections':sections}})
    states={label:(target[i] if known[i] else None) for i,label in enumerate(DISEASES)}
    text=' '.join(section['text'] for section in sections if section['heading'].lower() in {'findings','impression'})
    for name,pattern in EXTRA_FACTS.items():
        assertions=[]
        for sentence in re.split(r'[.;]|\bbut\b|\bhowever\b',text.lower()):
            for match in re.finditer(r'\b(?:'+pattern+r')\b',sentence):
                prefix=sentence[:match.start()]
                # A stated opacity can be definite while its cause (e.g. possible
                # pneumonia) is uncertain later in the sentence.
                if UNCERTAIN.search(prefix[-100:]):assertions.append(None)
                else:assertions.append(0.0 if NEG.search(prefix) else 1.0)
        states[name]=1.0 if 1.0 in assertions else (0.0 if 0.0 in assertions else None)
    return states


def analyze(text,reference,mesh_terms=None):
    result={'valid_clinical_json':False,'reference_positive_terms':0,'matched_positive_terms':0,'false_normal_on_positive_reference':False,'positive_terms_in_normal_reference':0}
    truth=fact_states(reference['sections'])
    positive=[name for name,value in truth.items() if value==1]
    result['reference_positive_terms']=len(positive)
    result['missing_reference_terms']=positive.copy()
    expected=[section['heading'] for section in reference['sections']]
    try:
        sections=parsed(text);validate_clinical_sections(sections,expected)
    except (ValueError,KeyError,TypeError,json.JSONDecodeError):return result
    result['valid_clinical_json']=True
    truth=fact_states(reference['sections']);prediction=fact_states(sections)
    if any(term.lower()=='normal' for term in (mesh_terms or [])):
        truth={name:(0.0 if value is None else value) for name,value in truth.items()}
    positive=[name for name,value in truth.items() if value==1]
    result['reference_positive_terms']=len(positive)
    result['matched_positive_terms']=sum(prediction.get(name)==1 for name in positive)
    result['false_normal_on_positive_reference']=bool(positive and not result['matched_positive_terms'])
    result['positive_terms_in_normal_reference']=sum(prediction.get(name)==1 and value==0 for name,value in truth.items())
    result['missing_reference_terms']=[name for name in positive if prediction.get(name)!=1]
    return result


def assess(path):
    path=Path(path);data=json.loads(path.read_text());details=[]
    reference_rows={}
    config_path=path.parent/'run_config.json'
    if config_path.exists():
        import hashlib
        configuration=json.loads(config_path.read_text())
        manifest=Path(configuration['validation'])
        expected=configuration.get('manifest_sha256',{}).get('validation')
        if expected and hashlib.sha256(manifest.read_bytes()).hexdigest()!=expected:
            raise ValueError('Validation manifest changed after the run started')
        reference_rows={row['study_id']:row for row in (json.loads(line) for line in manifest.read_text().splitlines() if line.strip())}
    for case in data['comparisons']:
        details.append({'study_id':case['study_id'],'base':analyze(case['base_output'],case['reference'],reference_rows.get(case['study_id'],{}).get('mesh_terms')),'adapter':analyze(case['adapter_output'],case['reference'],reference_rows.get(case['study_id'],{}).get('mesh_terms'))})
    summaries={}
    for key in ['base','adapter']:
        cases=[case[key] for case in details]
        reference=sum(c['reference_positive_terms'] for c in cases)
        summaries[key]={'generated_cases':len(cases),'valid_clinical_json':sum(c['valid_clinical_json'] for c in cases),'reference_positive_terms':reference,'matched_positive_terms':sum(c['matched_positive_terms'] for c in cases),'false_normal_cases':sum(c['false_normal_on_positive_reference'] for c in cases),'contradictory_positive_terms':sum(c['positive_terms_in_normal_reference'] for c in cases)}
    # Format or token loss alone cannot approve an adapter. Require a real
    # positive-reference comparison and no regression in this small proxy check.
    a,b=summaries['adapter'],summaries['base']
    accept=bool(data['validation_loss_improved'] and a['generated_cases']>=4 and a['reference_positive_terms']>=2 and a['valid_clinical_json']==a['generated_cases'] and a['matched_positive_terms']>b['matched_positive_terms'] and a['false_normal_cases']<=b['false_normal_cases'] and a['contradictory_positive_terms']<=b['contradictory_positive_terms'])
    result={'eligible_for_experimental_selection':accept,'summaries':summaries,'details':details,'limitations':'Small generated validation comparison with lexical assertion proxies, not expert clinical adjudication. Token loss, AUROC and lexical recall are not diagnosis accuracy.'}
    (path.parent/'assessment.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('evaluation');args=parser.parse_args();print(json.dumps(assess(args.evaluation),indent=2))
