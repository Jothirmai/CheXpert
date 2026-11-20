"""Write an honest experiment summary, retaining unsuccessful candidates."""
import html,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
root=Path(__file__).resolve().parents[1]
cnn=json.loads((root/'checkpoints/cnn_validation_blend/comparison.json').read_text())
vlm=json.loads((root/'checkpoints/vlm_evidence_ablation/normalized_summary.json').read_text())
cases=json.loads((root/'checkpoints/vlm_evidence_ablation/results_scored.json').read_text())['cases']
out=root/'reports/model_improvements';out.mkdir(exist_ok=True)
metrics={'cnn':cnn,'vlm':vlm,'cnn_bootstrap':json.loads((root/'checkpoints/cnn_validation_blend/bootstrap.json').read_text()),'active_models_changed':False,'test_set_used_for_selection':False}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
lines=['# Model improvement experiments','', 'Validation-only experiments. Active models have not been changed. No test metrics were used for selecting these candidates. The existing test set has previously been inspected, so it is not a fresh blinded evaluation.','', '| CNN metric | Existing | Candidate | Change |','|---|---:|---:|---:|']
for k in ['macro_auroc','macro_average_precision']:
 b=cnn['baseline'][k];a=cnn['candidate'][k];lines.append(f'| {k} | {b:.6f} | {a:.6f} | {a-b:+.6f} |')
lines+=['','The candidate blends 10% fine-tuned weights with 90% existing weights. Training used study-balanced loss and reduced positive-class weighting. Selection required nonregression in both metrics across the same 563 validation studies and 14 evaluable labels. These small adaptive-validation gains have no confidence interval or clinical significance established.','', '## VLM paired validation comparison','', '| Metric | CNN evidence + current prompt | Image only + restrained prompt |','|---|---:|---:|']
a=vlm['metrics']['cnn_evidence'];b=vlm['metrics']['image_only']
for k in a:lines.append(f'| {k} | {a[k]} | {b[k]} |')
lines+=['', 'CNN paired-bootstrap intervals: '+json.dumps(metrics['cnn_bootstrap']), '',vlm['limitations'],'','Positive-term recall should be calculated as matched_positive_terms / reference_positive_terms. A lower false-normal count can reflect invented positives; inspect both positive recall and unsupported findings. Invalid JSON is a generation failure even if its prose seems plausible.','', '## Case-by-case reports']
for c in cases:
 lines+=['',f"### {c['study_id']}",'','Reference:',json.dumps(c['reference'],ensure_ascii=False,indent=2)]
 for mode in ['cnn_evidence','image_only']:lines+=['',mode+':',c['outputs'][mode]['text']]
(out/'COMPARISON.md').write_text('\n'.join(lines))
def table(headers,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
body='<h1>Model improvement experiments</h1><p>Validation only. Active models unchanged. These are experimental results, not clinical validation.</p><h2>CNN</h2>'+table(['Metric','Existing','Candidate'],[[k,f"{cnn['baseline'][k]:.6f}",f"{cnn['candidate'][k]:.6f}"] for k in ['macro_auroc','macro_average_precision']])
body+='<p>Paired-bootstrap intervals: '+html.escape(json.dumps(metrics['cnn_bootstrap']))+'</p>'
body+='<p>Candidate: 90% existing weights + 10% fine-tuned weights. Both validation metrics improved slightly. Validation tuning does not establish external generalization.</p><h2>VLM</h2>'+table(['Metric','CNN evidence','Image only'],[[k,a[k],b[k]] for k in a])+'<p>'+html.escape(vlm['limitations'])+'</p>'
def display_sections(output):
    try:
        data=json.loads(re.search(r'\{.*\}',output,re.S).group())
        if 'sections' in data:return {s['heading']:s['text'] for s in data['sections']}
        return {k:(v if isinstance(v,str) else '\n'.join(f'{a}: {b}' for a,b in v.items())) for k,v in data.items()}
    except (ValueError,TypeError,KeyError,AttributeError):return {'Findings':output}
for c in cases:
 reference={s['heading']:s['text'] for s in c['reference']['sections']}
 current=display_sections(c['outputs']['cnn_evidence']['text'])
 candidate=display_sections(c['outputs']['image_only']['text'])
 body+='<h2>'+html.escape(c['study_id'])+'</h2>'+table(['Section','Reference','Current evidence prompt','Candidate image-only prompt'],[[heading,reference.get(heading,''),current.get(heading,''),candidate.get(heading,'')] for heading in ['Findings','Impression']])

report='<!doctype html><html><head><meta charset="utf-8"><style>body{font:13px Arial;margin:30px;color:#17324a}table{border-collapse:collapse;width:100%;margin:15px 0}td,th{border:1px solid #aaa;padding:10px;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}th{background:#edf3f5}h2{break-after:avoid}@page{size:A4 landscape;margin:12mm}</style></head><body>'+body+'</body></html>'
(out/'comparison.html').write_text(report)
from weasyprint import HTML
HTML(string=report).write_pdf(out/'comparison.pdf')
print(out/'comparison.pdf')
