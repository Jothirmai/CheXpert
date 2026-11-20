"""Build auditable CNN/VLM tables from saved results and cited paper numbers."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/sota_comparison_2026'
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    'chexpert_leaderboard': 'https://stanfordmlgroup.github.io/competitions/chexpert/',
    'cliq': 'https://arxiv.org/html/2610.13059v1',
    'cxpmrg': 'https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_CXPMRG-Bench_Pre-training_and_Benchmarking_for_X-ray_Medical_Report_Generation_on_CVPR_2025_paper.pdf',
}


def load(path):
    return json.loads((ROOT / path).read_text())


def fmt(value):
    return 'NR' if value is None else f'{value:.3f}'


def markdown(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + ''.join(
        '| ' + ' | '.join(map(str, row)) + ' |\n' for row in rows)


def tex_table(caption, label, headers, rows):
    def esc(value):
        return str(value).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
    return ('\\begin{table*}[t]\n\\centering\\scriptsize\n'
            f'\\caption{{{caption}}}\n'
            f'\\begin{{tabular}}{{{"l" * len(headers)}}}\\hline\n'
            + ' & '.join(map(esc, headers)) + r'\\' + '\n\\hline\n'
            + ''.join(' & '.join(map(esc, row)) + r'\\' + '\n' for row in rows)
            + f'\\hline\\end{{tabular}}\\label{{{label}}}\n\\end{{table*}}\n')


def main():
    cnn_test = load('reports/chexpert_densenet201_v12_tta_blends_test_exploratory.json')
    assert cnn_test['selected_variant'] == 'equal_v6_v12_tta'
    own_five = cnn_test['macro_auroc']
    five = [
        ('Published†', 'DeepAUC-v1 ensemble', '2020', '5', '500-study official test', '0.930', 'Stanford leaderboard'),
        ('Published†', 'Hierarchical-Learning-V1', '2019', '5', '500-study official test', '0.930', 'Stanford leaderboard'),
        ('Published†', 'Conditional-Training-LSR', '2019', '5', '500-study official test', '0.929', 'Stanford leaderboard'),
        ('Published†', 'Stanford baseline ensemble', '2019', '5', '500-study official test', '0.907', 'Stanford leaderboard'),
        ('Ours', 'DenseNet201 v6/v12 flip TTA', '2026', '5', '500-study test, repeated', fmt(own_five), 'Saved local result'),
    ]
    v6 = load('checkpoints/chexpert_plus_densenet201_v6/best_metrics.json')['development']
    v12 = load('checkpoints/chexpert_plus_impression_320_v12/best_metrics.json')['development']
    all14_path = ROOT / 'reports/selected_ensemble_all14_development.json'
    all14 = load('reports/selected_ensemble_all14_development.json') if all14_path.exists() else None
    fourteen = [
        ('Ours', 'DenseNet201 v6', 'CheXpert Plus Impression development', str(v6['evaluable_labels']), fmt(v6['macro_auroc']), fmt(v6['macro_ap'])),
        ('Ours', 'DenseNet201 v12', 'CheXpert Plus Impression development', str(v12['evaluable_labels']), fmt(v12['macro_auroc']), fmt(v12['macro_ap'])),
        ('Ours', 'Selected v6/v12 ensemble', 'CheXpert Plus Impression development',
         str(all14['evaluable_labels']) if all14 else '14 planned',
         fmt(all14['macro_auroc']) if all14 else 'Pending',
         fmt(all14['macro_ap']) if all14 else 'Pending'),
    ]
    # IU rows are transcribed from the cited CliQ-RRG Table 1. A star in that
    # paper means its authors reproduced that method; we have not reproduced it.
    iu = [
        ('Paper-reproduced⋆', 'R2GenRL', 'ACL 2022', .487,.177,.205,.377,.452,.835),
        ('Paper-reproduced⋆', 'DCL', 'CVPR 2023', .483,.171,.201,.391,.461,.838),
        ('Paper-reproduced⋆', 'CoFE', 'ECCV 2024', .406,.177,.199,.423,.508,.849),
        ('Paper-reproduced⋆', 'R2-LLM', 'AAAI 2024', .487,.178,.215,.401,.485,.846),
        ('Paper-reproduced⋆', 'HC-LLM', 'AAAI 2025', .532,.199,.221,.423,.503,.839),
        ('Published†', 'MPO', 'AAAI 2025', .548,.209,.224,.415,None,None),
        ('Published†', 'DAD-RG', 'AAAI 2026', .495,.187,None,.384,.634,None),
        ('CliQ-RRG', 'Graph-guided', 'EMNLP 2026', .565,.221,.225,.438,.589,.885),
    ]
    chex_vlm = [
        ('Benchmark reproduction', 'R2Gen', 'EMNLP 2020',None,.081,.113,.246,.077,None,.181),
        ('Benchmark reproduction', 'R2GenCMN', 'ACL 2021',None,.087,.127,.256,.102,None,.231),
        ('Benchmark reproduction', 'XProNet', 'ECCV 2022',None,.100,.146,.265,.121,None,.259),
        ('Benchmark reproduction', 'ORGan', 'ACL 2023',None,.086,.135,.261,.107,None,.277),
        ('Benchmark reproduction', 'R2GenGPT', '2023',None,.101,.145,.266,.123,None,.260),
        ('Benchmark reproduction', 'PromptMRG', 'AAAI 2024',None,.095,.121,.222,.044,None,.281),
        ('Benchmark reproduction', 'MambaXray-VL-L', 'CVPR 2025',None,.112,.157,.276,.139,None,.335),
    ]
    for dataset, filename, rows in [
        ('IU X-Ray', 'checkpoints/report_lora_selected_cnn_v1/standard_validation_metrics.json', iu),
        ('CheXpert Plus', 'checkpoints/report_lora_chexpert_plus_v1/standard_validation_metrics.json', chex_vlm),
    ]:
        path = ROOT / filename
        if path.exists():
            measured = json.loads(path.read_text())['metrics']
            for name in ('base', 'adapter'):
                item = measured[name]
                rows.append(('Local validation', f'Qwen2.5-VL-7B {name}', f"n={item['cases']}",
                             item['BLEU_1'], item['BLEU_4'], item['METEOR'],
                             item['ROUGE_L'], item['CIDEr'], item['BERTScore_F1_roberta_large'], None))
        else:
            rows.extend([('Local validation', 'Qwen2.5-VL-7B base', 'Pending',None,None,None,None,None,None,None),
                         ('Local validation', 'Qwen2.5-VL-7B LoRA', 'Pending',None,None,None,None,None,None,None)])
    test_result = ROOT / 'checkpoints/report_lora_chexpert_plus_v1/reserved_test_results_rescored.json'
    if test_result.exists():
        heldout = json.loads(test_result.read_text())
        values = heldout['metrics']
        chex_vlm.append(('Local reserved test', 'Qwen2.5-VL-7B adapter (4-bit)',
                         f"n={heldout['cases']}; {heldout['schema_valid']} valid",
                         values['BLEU_1'], values['BLEU_4'], values['METEOR'],
                         values['ROUGE_L'], values['CIDEr'],
                         values['BERTScore_F1_roberta_large'], None))
    fp16_result = ROOT / 'checkpoints/report_lora_chexpert_plus_v1/reserved_test_fp16_followup.json'
    if fp16_result.exists():
        followup = json.loads(fp16_result.read_text())
        values = followup['metrics']
        chex_vlm.append(('Exploratory follow-up', 'Qwen2.5-VL-7B adapter (FP16)',
                         f"n={followup['cases']}; {followup['schema_valid']} valid",
                         values['BLEU_1'], values['BLEU_4'], values['METEOR'],
                         values['ROUGE_L'], values['CIDEr'],
                         values['BERTScore_F1_roberta_large'], None))
    vlm_iu = [[kind, method, year, *(fmt(x) for x in (values + [None] * (7-len(values))))]
              for kind, method, year, *values in iu]
    vlm_chex = [[kind, method, year, *(fmt(x) for x in values)]
                for kind, method, year, *values in chex_vlm]
    lines = [
        '# CNN and VLM quantitative comparisons', '',
        'The published rows are context. Local and published rows have different test splits and often different scorers; do not rank them as a single leaderboard. † means published as reported in the cited paper; ⋆ means reproduced by the CliQ-RRG authors, not by this project.', '',
        '## CNN: five expert-labeled CheXpert findings', '',
        markdown(['Source','Method','Year','Labels','Cohort','Mean AUROC','Provenance'], five),
        'Our 0.903 is measured but exploratory because the same 500-study test set was inspected during model development. The official leaderboard rows target five findings; no 14-label test score is inferred.', '',
        '## CNN: all 14 report-derived CheXpert Plus labels', '',
        markdown(['Source','Method','Cohort','Evaluable labels','Macro AUROC','Macro AP'], fourteen),
        'These 14-label numbers are from the frozen patient-disjoint development split using Impression-derived weak labels, not the five-finding expert test.', '',
        '## VLM: IU X-Ray', '',
        markdown(['Source','Method','Year / cohort','B1','B4','METEOR','ROUGE-L','CIDEr','BERTScore','CE-F1'], vlm_iu),
        'The published IU scores are from the CliQ-RRG paper. Local scores use our separate held-out IU validation cases and strict JSON report output.', '',
        '## VLM: CheXpert Plus', '',
        markdown(['Source','Method','Year / cohort','B1','B4','METEOR','ROUGE-L','CIDEr','BERTScore','CE-F1'], vlm_chex),
        'The published CheXpert Plus scores are from CXPMRG-Bench. Its paper did not report BERTScore in this table. Our local split differs from that benchmark. The FP16 row reuses the same 96 cases after inspection of the 4-bit result and is exploratory. Invalid reports score zero. Clinical F1 requires a verified clinical labeler and is not inferred from word overlap.', '',
        '## Sources', '',
        *[f'- {name}: {url}' for name, url in SOURCES.items()],
    ]
    (OUT / 'comparison.md').write_text('\n'.join(lines) + '\n')
    tex = '\n'.join([
        '% Published numbers are contextual; only matched local rows may be ranked against each other.',
        tex_table('CheXpert expert-test five-finding CNN AUROC. The selected local result is a repeat evaluation.',
                  'tab:cnn_five_2026', ['Source','Method','Year','Labels','Cohort','AUROC','Provenance'], five),
        tex_table('Fourteen-label CheXpert Plus development results. Report-derived labels differ from the expert five-finding test.',
                  'tab:cnn_fourteen_2026', ['Source','Method','Cohort','Labels','Macro AUROC','Macro AP'], fourteen),
        tex_table('IU X-Ray report-generation scores. Published and local validation rows use different splits.',
                  'tab:vlm_iu_2026', ['Source','Method','Year/cohort','B1','B4','MTR','RG-L','C','BS','CE-F1'], vlm_iu),
        tex_table('CheXpert Plus report-generation scores. Published and local validation rows use different splits.',
                  'tab:vlm_chex_2026', ['Source','Method','Year/cohort','B1','B4','MTR','RG-L','C','BS','CE-F1'], vlm_chex),
    ])
    (OUT / 'comparison_tables.tex').write_text(tex)
    (OUT / 'source_manifest.json').write_text(json.dumps({'sources': SOURCES,
        'local_cnn_five': 'reports/chexpert_densenet201_v12_tta_blends_test_exploratory.json',
        'local_cnn_fourteen': 'checkpoints/chexpert_plus_densenet201_v6/best_metrics.json and checkpoints/chexpert_plus_impression_320_v12/best_metrics.json',
        'local_vlm_metrics': ['checkpoints/report_lora_selected_cnn_v1/standard_validation_metrics.json',
                              'checkpoints/report_lora_chexpert_plus_v1/standard_validation_metrics.json'],
        'comparison_status': 'contextual only until common test cohorts and scorer implementations are established'}, indent=2) + '\n')
    print(OUT / 'comparison.md')


if __name__ == '__main__':
    main()
