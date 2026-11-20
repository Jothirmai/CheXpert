"""Build auditable metrics and a printable report from saved evaluation artifacts."""
import csv,html,json,re,sys,shutil,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cliq_rrg.eval import bleu,rouge_l
root=Path(__file__).resolve().parents[1]
out=root/'reports/model_comparison';out.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads((root/p).read_text())
def clinical(sections):return ' '.join(s['text'] for s in sections if s['heading'].lower() in {'findings','impression'})
def parsed(text):
 d=json.loads(re.search(r'\{.*\}',text,re.S).group())
 if 'sections' in d:return d['sections']
 return [{'heading':k,'text':v if isinstance(v,str) else json.dumps(v)} for k,v in d.items()]
def score(ref,pred):return {'BLEU-1':bleu(ref,pred,1),'BLEU-2':bleu(ref,pred,2),'BLEU-3':bleu(ref,pred,3),'BLEU-4':bleu(ref,pred,4),'ROUGE-L F1':rouge_l(ref,pred)}
evaluation=read('checkpoints/report_lora_corrected/evaluation.json');assess=read('checkpoints/report_lora_corrected/assessment.json')
metrics={'cnn_validation':read('checkpoints/cnn_corrected/validation_metrics.json'),'cnn_test':read('checkpoints/cnn_corrected/test_metrics.json'),'vlm':{'model':evaluation['base_model'],'validation_images':evaluation['validation_images'],'base_token_loss':evaluation['base_validation_loss'],'adapter_token_loss':evaluation['adapter_validation_loss'],'generation_cases':len(evaluation['comparisons']),'assessment':assess['summaries']},'text_metric_definition':'Lowercased whitespace tokens; unsmoothed sentence BLEU averaged over cases, ROUGE-L LCS F1; Findings + Impression only. No metadata sections. Lexical overlap is not clinical accuracy.'}
for key in ['base','adapter']:
 scores=[score(clinical(c['reference']['sections']),clinical(parsed(c[key+'_output']))) for c in evaluation['comparisons']]
 metrics['vlm'][key+'_lexical_macro']={k:sum(s[k] for s in scores)/len(scores) for k in scores[0]}
 metrics['vlm'][key+'_per_case']=[{'study_id':c['study_id'],**s} for c,s in zip(evaluation['comparisons'],scores)]
if (out/'generated_report.json').exists():
 reference=json.loads((out/'dataset_reference.json').read_text());generated=json.loads((out/'generated_report.json').read_text());provenance=json.loads((out/'generation_provenance.json').read_text())
 metrics['new_test_example']={'study_id':reference['study_id'],**score(clinical(reference['target']['sections']),clinical(generated['sections'])),'provenance':provenance}
else:raise RuntimeError('Run generate_dataset_comparison.py first')
(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
with (out/'cnn_per_label.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['label','test_positives','test_negatives','test_AUROC','test_average_precision'])
 for label,m in metrics['cnn_test']['per_label'].items():writer.writerow([label,m['positive_studies'],m['negative_studies'],m['auroc'],m['average_precision']])
esc=lambda s:html.escape(str(s))
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def fmt(v):return 'Insufficient cases' if v is None else f'{v:.4f}'
parts=['<h1>Chest Xpert: CNN and VLM evaluation</h1><p>Saved experiment metrics plus one newly generated held-out dataset example. All scores use a 0–1 scale unless stated. No clinical accuracy claim.</p><h2>CNN — corrected Stage1CNN</h2>']
parts.append(table(['Split','Studies','Evaluable labels','Macro AUROC','Macro AP'],[[name,m['studies'],m['evaluable_labels'],fmt(m['macro_auroc']),fmt(m['macro_average_precision'])] for name,m in [('Validation',metrics['cnn_validation']),('Test',metrics['cnn_test'])]]))
parts.append('<p>AUROC measures ranking; AP summarizes the precision–recall curve. Neither is percent accuracy. Labels are weak MeSH/report-derived annotations. Four test labels lack sufficient positives. Validation and test macros cover different label sets.</p>')
parts.append(table(['Finding','Test positives','AUROC','AP'],[[label,m['positive_studies'],fmt(m['auroc']),fmt(m['average_precision'])] for label,m in metrics['cnn_test']['per_label'].items()]))
parts.append('<h2>VLM — '+esc(evaluation['base_model'])+'</h2><p>Active model: base Qwen. The experimental LoRA adapter remains disabled. Training: 512 image examples, 64 optimization steps; token-loss validation: 24 image examples; generated comparison: only four validation examples.</p>')
parts.append(table(['Metric','Base','LoRA adapter'],[['Validation token loss',fmt(evaluation['base_validation_loss']),fmt(evaluation['adapter_validation_loss'])]]+[[k,fmt(metrics['vlm']['base_lexical_macro'][k]),fmt(metrics['vlm']['adapter_lexical_macro'][k])] for k in metrics['vlm']['base_lexical_macro']]+[['Valid requested JSON schema','3 / 4','4 / 4']]))
parts.append('<p>'+esc(metrics['text_metric_definition'])+' A malformed base schema is normalized only for lexical scoring; it still fails the schema check. Loss reduction does not establish clinical improvement. CheXbert, RadGraph and standard BERTScore have not been measured here; legacy repository functions with these names include placeholders and are excluded.</p>')
parts.append('<h2>New held-out example: '+esc(reference['study_id'])+'</h2><p>Selected the first test-manifest row before generation. Reference findings and impression were withheld from the model. This is one example, not a dataset-wide VLM test score. Source reports can describe multiple views, while this run uses one image.</p><img class="xray" src="original.png" alt="Original dataset X-ray">')
xml=root/'data/iu_xray/reports/ecgen-radiology'/f"{reference['study_id'].removeprefix('CXR')}.xml"
shutil.copyfile(xml,out/'original_dataset_report.xml')
original={node.get('Label',''): ' '.join(node.itertext()).strip() for node in ET.parse(xml).findall('.//AbstractText')}
gensections={s['heading']:s['text'] for s in generated['sections']}
parts.append(table(['Section','Original dataset report (verbatim XML text)','New generated report'],[[label,original.get(label,'Not supplied in XML'),gensections.get(heading,'Not supplied')] for label,heading in [('INDICATION','Clinical indication'),('COMPARISON','Comparison'),('FINDINGS','Findings'),('IMPRESSION','Impression')]]))
parts.append('<h3>Observed differences against this dataset reference</h3>'+table(['Finding','Dataset reference','Generated output','Comparison'],[['Low lung volumes / crowding','Explicitly present','Not mentioned','Omitted'],['Image positioning','Kyphotic and rotated','Not mentioned','Omitted'],['Heart size','Borderline enlarged, accounting for technical factors','Definite cardiomegaly','Uncertainty and technical caveat lost'],['Emphysema','Not reported','Reported in findings and impression','Unsupported by the reference'],['Consolidation / effusion','No definite consolidation or effusion','No significant consolidation or effusion','Broad agreement']]))
parts.append('<h3>How to read this result</h3><p>Compare individual statements in Findings first, then Impression. Check whether low lung volumes, crowding, positioning and heart size are retained, omitted or changed. Agreement on “no acute findings” alone does not mean all findings match. This is a textual comparison against the dataset report, not an independent interpretation of the image.</p>')
parts.append(table(['Single-example text metric','Value'],[[k,fmt(v)] for k,v in metrics['new_test_example'].items() if isinstance(v,float)]))
parts.append('<h3>CNN evidence used for this report</h3>'+table(['Finding','Uncalibrated CNN score'],[[l['label'],fmt(l['score'])] for l in generated['labels']]))
if (out/'gradcam_contact_sheet.png').exists():parts.append('<img class="heatmap" src="gradcam_contact_sheet.png" alt="CNN Grad-CAM contact sheet">')
parts.append('<p>The CNN produces 14 supporting scores and a top-six Grad-CAM contact sheet. Qwen receives the original image, contact sheet, scores and supplied clinical metadata; its report is not a direct threshold conversion of the CNN outputs. A heatmap shows model attention, not proof of disease.</p><pre>'+esc(json.dumps(generated['model_provenance'],indent=2))+'</pre><p>Generation time: '+f"{provenance['elapsed_seconds']:.1f}"+' seconds including model loading within this run. Exact metrics and inputs are preserved in adjacent JSON files.</p>')
style='body{font:14px Arial;color:#14283b;max-width:1200px;margin:30px auto;padding:20px}h1,h2{color:#174c69}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #b8c8d0;padding:10px;text-align:left;vertical-align:top}th{background:#e9f2f6}tr{break-inside:avoid}.xray{display:block;max-height:430px;max-width:100%;margin:auto}.heatmap{width:100%}pre{white-space:pre-wrap}@page{size:A4 landscape;margin:14mm}'
report='<!doctype html><html><head><meta charset="utf-8"><title>Chest Xpert model comparison</title><style>'+style+'</style></head><body>'+''.join(parts)+'</body></html>'
(out/'comparison.html').write_text(report)
from weasyprint import HTML
HTML(string=report,base_url=str(out)).write_pdf(out/'comparison.pdf')
print(json.dumps({'vlm':metrics['vlm'],'new_test_example':metrics['new_test_example'],'report':str(out/'comparison.html')},indent=2))
