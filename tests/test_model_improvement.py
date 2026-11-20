from scripts.assess_report_adapter import analyze


def test_invalid_generation_still_counts_reference_findings():
    reference={'sections':[{'heading':'Findings','text':'There is pneumothorax.'},{'heading':'Impression','text':'Pneumothorax.'}]}
    result=analyze('not valid JSON',reference)
    assert not result['valid_clinical_json']
    assert result['reference_positive_terms']==1
    assert result['matched_positive_terms']==0
    assert result['missing_reference_terms']==['Pneumothorax']


def test_matching_negation_does_not_create_positive_recall():
    reference={'sections':[{'heading':'Findings','text':'No pneumothorax.'}]}
    result=analyze('{"sections":[{"heading":"Findings","text":"No pneumothorax."}]}',reference)
    assert result['reference_positive_terms']==0
    assert result['matched_positive_terms']==0


def test_alternate_json_normalizes_without_rewriting_findings():
    from chest_xpert.report_quality import parse_generated_sections
    sections,normalized=parse_generated_sections('{"Findings":{"Heart":"Borderline enlarged.","Pleura":"No effusion."},"Impression":"No acute process."}', ['Findings','Impression'])
    assert normalized
    assert sections[0]['text']=='Heart: Borderline enlarged.\nPleura: No effusion.'
    assert sections[1]['text']=='No acute process.'


def test_normalization_rejects_missing_or_unsafe_content():
    import pytest
    from chest_xpert.report_quality import parse_generated_sections
    for text in ['{"Findings":"Clear."}', '{"Findings":"copyright publisher","Impression":"Clear."}', '{"Findings":["Clear."],"Impression":"Clear."}']:
        with pytest.raises(ValueError):
            parse_generated_sections(text,['Findings','Impression'])


def test_finding_balanced_sampling_is_deterministic_and_study_unique():
    from scripts.train_report_adapter import select_rows
    normal=[{'study_id':f'N{i}','mesh_terms':['normal'],'target':{'sections':[{'heading':'Findings','text':'Clear.'}]}} for i in range(20)]
    positive=[{'study_id':f'P{i}','mesh_terms':['Pneumothorax'],'target':{'sections':[{'heading':'Findings','text':'Pneumothorax.'}]}} for i in range(20)]
    rows=normal+positive+[positive[0].copy()]
    chosen=select_rows(rows,12,'finding_balanced',42)
    assert len(chosen)==12 and len({r['study_id'] for r in chosen})==12
    assert sum(r['study_id'].startswith('N') for r in chosen)<=3
    assert chosen==select_rows(rows,12,'finding_balanced',42)



def test_disease_json_labels_do_not_count_as_positive_findings():
    from scripts.assess_report_adapter import fact_states
    from chest_xpert.report_quality import report_warnings
    sections=[{'heading':'Findings','text':'Effusion: No pleural effusion detected.\nPneumothorax: No pneumothorax detected.'},{'heading':'Impression','text':'No pleural effusion or pneumothorax.'}]
    states=fact_states(sections)
    assert states['Effusion']==0 and states['Pneumothorax']==0
    assert report_warnings(sections)==[]


def test_positive_reference_not_matched_by_a_negated_json_field():
    import json
    reference={'sections':[{'heading':'Findings','text':'Pleural effusion.'}]}
    generated={'sections':[{'heading':'Findings','text':'Effusion: No pleural effusion detected.'}]}
    result=analyze(json.dumps(generated),reference)
    assert result['reference_positive_terms']==1
    assert result['matched_positive_terms']==0


def test_json_field_heading_is_not_reintroduced_after_a_second_sentence():
    from scripts.assess_report_adapter import fact_states
    states=fact_states([{'heading':'Findings','text':'Effusion: No pleural effusion. Heart size is normal.'}])
    assert states['Effusion']==0
