import json
from scripts.assess_full_study_adapter import score,states,summarize

def row(text,mesh=()):
    return {'target':{'sections':[{'heading':'Findings','text':text},{'heading':'Impression','text':'No definite acute findings.'}]},'mesh_terms':list(mesh)}

def output(text):
    return json.dumps({'sections':[{'heading':'Findings','text':text},{'heading':'Impression','text':'No definite acute findings.'}]})

def test_quality_findings_are_counted_and_effusion_negation_is_preserved():
    reference=row('Low lung volumes with bronchovascular crowding. Rotated frontal view. No pleural effusion.')
    result=score(output('Low lung volumes. No effusion.'),reference)
    assert result['accepted']
    assert 'Low lung volumes' not in result['missed_facts']
    assert 'Bronchovascular crowding' in result['missed_facts']
    assert 'Rotation' in result['missed_facts']
    assert result['unsupported_facts']==[]
    wrong=score(output('Low lung volumes. Right pleural effusion.'),reference)
    assert 'Effusion' in wrong['unsupported_facts']

def test_invalid_reports_do_not_receive_matches():
    result=score('not JSON',row('Low lung volumes.'))
    assert not result['accepted'] and result['matched_facts']==0
    assert result['missed_facts']==['Low lung volumes']

def test_normal_reference_and_quality_uncertainty():
    result=score(output('Cardiomegaly.'),row('Clear lungs. Normal heart size.', ['normal']))
    assert 'Cardiomegaly' in result['unsupported_facts']
    assert states([{'heading':'Findings','text':'Possible low lung volumes.'}])['Low lung volumes'] is None
    assert summarize([result])['unsupported_facts']>=1


def test_selection_rejects_loss_only_and_new_hallucinations():
    from scripts.assess_full_study_adapter import eligible_for_selection
    base={'cases':12,'accepted_reports':12,'reference_positive_facts':20,'missed_facts':12,'unsupported_facts':4,'contradictions':0}
    improved={**base,'missed_facts':8,'unsupported_facts':2}
    data={'validation_loss_improved':True}
    totals={'base':base,'adapter':improved}
    assert eligible_for_selection(data,totals,{'abnormal':totals})
    assert not eligible_for_selection(data,{'base':base,'adapter':base},{'abnormal':totals})
    worse={**improved,'unsupported_facts':5}
    assert not eligible_for_selection(data,{'base':base,'adapter':worse},{'abnormal':totals})
    assert not eligible_for_selection({'validation_loss_improved':False},totals,{'abnormal':totals})


def test_full_assessment_exports_raw_comparisons(tmp_path,monkeypatch):
    from scripts.assess_full_study_adapter import assess
    monkeypatch.chdir(tmp_path)
    folder=tmp_path/'experiment';folder.mkdir()
    reference=row('Low lung volumes with bronchovascular crowding. No effusion.')
    rows=[{'study_id':f'CXR{i}',**reference} for i in range(12)]
    (folder/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (folder/'cohort_audit.json').write_text(json.dumps({'generated_validation_studies':[r['study_id'] for r in rows],'validation_categories':{r['study_id']:('low_volumes','technical','abnormal','normal')[i%4] for i,r in enumerate(rows)}}))
    comparisons=[{'study_id':r['study_id'],'reference':r['target'],'base_output':output('Pleural effusion.'),'adapter_output':output('Low lung volumes with bronchovascular crowding. No effusion.')} for r in rows]
    (folder/'evaluation.json').write_text(json.dumps({'comparisons':comparisons,'validation_loss_improved':True,'base_validation_loss':1.,'adapter_validation_loss':.7}))
    result=assess(folder)
    assert result['eligible_for_experimental_selection']
    assert result['metrics']['adapter']['missed_facts']==0
    assert (folder/'comparison.pdf').read_bytes().startswith(b'%PDF')
    assert 'Pleural effusion.' in (folder/'COMPARISON.md').read_text()
    assert json.loads((tmp_path/'reports/full_study_training/metrics.json').read_text())['eligible_for_experimental_selection']
