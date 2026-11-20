import json
from pathlib import Path
from chest_xpert.training_data import parse_clinical_xml, structured_target, prepare

XML='''<eCitation><uId id="CXR1"/><publisher>Publisher Secret</publisher><title>Dataset Title</title><Author><LastName>Author Secret</LastName></Author><Abstract><AbstractText Label="INDICATION">Cough.</AbstractText><AbstractText Label="COMPARISON">None.</AbstractText><AbstractText Label="FINDINGS">There is no focal consolidation. Small left pleural effusion.</AbstractText><AbstractText Label="IMPRESSION">Small left pleural effusion.</AbstractText></Abstract><parentImage id="CXR1_1"><caption>PA and lateral chest XXXX. Publisher noise.</caption></parentImage></eCitation>'''


def test_parser_excludes_publisher_and_preserves_negation(tmp_path):
    path=tmp_path/'1.xml';path.write_text(XML)
    record=parse_clinical_xml(path)
    target=json.dumps(structured_target(record['sections']))
    assert 'Publisher' not in target and 'Dataset Title' not in target and 'Author Secret' not in target
    assert 'no focal consolidation' in target and 'Small left pleural effusion' in target
    assert record['sections']['TECHNIQUE']=='PA and lateral chest radiographs.'
    assert record['sections']['INDICATION']=='Cough.'


def test_missing_sections_never_invented():
    target=structured_target({'FINDINGS':'A XXXX opacity.'})
    assert target['sections'][0]['text']=='NOT PROVIDED'
    assert target['sections'][2]['text']=='NOT PROVIDED'
    assert '[REDACTED]' in target['sections'][3]['text']


def test_manifests_keep_studies_and_images_separate(tmp_path):
    source=tmp_path/'source';images=source/'images';reports=source/'reports';images.mkdir(parents=True);reports.mkdir()
    for index in range(1,21):
        report=XML.replace('CXR1',f'CXR{index}')
        (reports/f'{index}.xml').write_text(report)
        for view in [1,2]:
            __import__('PIL.Image', fromlist=['new']).new('RGB',(8,8),(index,view,index*view)).save(images/f'CXR{index}_{view}.png')
    output=tmp_path/'manifests';audit=prepare(source,output)
    groups=[];hashes=[]
    for split in ['train','validation','test']:
        rows=[json.loads(line) for line in (output/f'{split}.jsonl').read_text().splitlines()]
        groups.append({row['study_id'] for row in rows});hashes.append({row['image_sha256'] for row in rows})
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert not hashes[0]&hashes[1] and not hashes[0]&hashes[2]
    assert audit['nonempty_source_sections']['FINDINGS']==20


def test_publication_contamination_rejected_and_cnn_warning_explicit():
    import pytest
    from chest_xpert.report_quality import validate_clinical_sections, evidence_warnings
    with pytest.raises(ValueError, match='publication metadata'):
        validate_clinical_sections([{'heading':'Findings','text':'Published by Example Journal.'}],['Findings'])
    assert validate_clinical_sections([{'heading':'Findings','text':'No focal consolidation.'}],['Findings'])
    assert evidence_warnings({'Edema':0.000001})
    assert evidence_warnings({'Edema':0.2})==[]


def test_adapter_assessment_detects_missed_positive_and_negation():
    from scripts.assess_report_adapter import analyze
    reference={'sections':[{'heading':'Findings','text':'Small pleural effusion.'},{'heading':'Impression','text':'Pleural effusion.'}]}
    missing='{"sections":[{"heading":"Findings","text":"No pleural effusion."},{"heading":"Impression","text":"Normal chest."}]}'
    result=analyze(missing,reference)
    assert result['valid_clinical_json'] and result['false_normal_on_positive_reference']
    assert result['matched_positive_terms']==0


def test_adapter_assessment_checks_opacity_even_if_cause_uncertain():
    from scripts.assess_report_adapter import analyze
    reference={'sections':[{'heading':'Findings','text':'Right upper lobe opacities may represent atelectasis or infiltrate.'},{'heading':'Impression','text':'Right upper lobe opacities.'}]}
    missing='{"sections":[{"heading":"Findings","text":"Lungs are clear."},{"heading":"Impression","text":"Normal chest."}]}'
    result=analyze(missing,reference)
    assert 'Opacity' in result['missing_reference_terms']
    assert result['false_normal_on_positive_reference']


def test_normal_source_cannot_hide_hallucinated_disease_in_adapter_assessment():
    from scripts.assess_report_adapter import analyze
    reference={'sections':[{'heading':'Findings','text':'Heart and lungs are normal.'},{'heading':'Impression','text':'Normal chest.'}]}
    output='{"sections":[{"heading":"Findings","text":"Small pleural effusion."},{"heading":"Impression","text":"Pleural effusion."}]}'
    result=analyze(output,reference,['normal'])
    assert result['positive_terms_in_normal_reference']>0


def test_model_metadata_does_not_belong_in_clinical_sections():
    import pytest
    from chest_xpert.report_quality import validate_clinical_sections
    with pytest.raises(ValueError,match='model metadata'):
        validate_clinical_sections([{'heading':'Findings','text':'CNN scores from Qwen indicate normality.'}],['Findings'])


def test_candidate_acceptance_requires_generated_abnormality_improvement(tmp_path):
    from scripts.assess_report_adapter import assess
    import json
    reference={'sections':[{'heading':'Findings','text':'Small pleural effusion.'},{'heading':'Impression','text':'Pleural effusion.'}]}
    base=json.dumps({'sections':[{'heading':'Findings','text':'No pleural effusion.'},{'heading':'Impression','text':'Normal chest.'}]})
    good=json.dumps(reference)
    path=tmp_path/'evaluation.json'
    data={'validation_loss_improved':True,'comparisons':[{'study_id':f'CXR{i}','reference':reference,'base_output':base,'adapter_output':good} for i in range(4)]}
    path.write_text(json.dumps(data));assert assess(path)['eligible_for_experimental_selection']
    for case in data['comparisons']:case['adapter_output']=base
    path.write_text(json.dumps(data));assert not assess(path)['eligible_for_experimental_selection']


def test_adapter_for_different_foundation_model_is_rejected(tmp_path):
    from peft import LoraConfig
    from chest_xpert.verification import LLMModelClient
    from types import SimpleNamespace
    config=LoraConfig(base_model_name_or_path='Other/Model',target_modules=['q_proj'],r=8)
    config.save_pretrained(tmp_path)
    client=LLMModelClient.__new__(LLMModelClient)
    client.model_id='Qwen/Qwen2.5-VL-7B-Instruct'
    client._model=SimpleNamespace(config=SimpleNamespace(_name_or_path=client.model_id))
    client._lora_loaded=False;client._load_error=None
    client._load_lora_adapter(str(tmp_path))
    assert not client._lora_loaded and 'requires base model' in client._load_error
