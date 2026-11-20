"""Explicit local model selection; original artifacts remain available."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def selection():
    path=ROOT/'checkpoints/active_models.json'
    return json.loads(path.read_text()) if path.is_file() else {}


def selected_cnn():
    override=os.environ.get('CHESTXPERT_CNN_CHECKPOINT')
    if override:return Path(override)
    chosen=selection().get('cnn')
    if chosen:
        if chosen.get('status')!='selected_experimental':raise ValueError('CNN selection is not approved for experimental use')
        if chosen.get('architecture') == 'densenet201_v6_v12_tta':
            raise ValueError('Selected CNN is a two-checkpoint ensemble; use SelectedEnsembleEngine')
        return ROOT/chosen['path']
    return ROOT/'checkpoints/cnn_model.pth'


def selected_cnn_checkpoints():
    chosen=selection().get('cnn') or {}
    if chosen.get('architecture') == 'densenet201_v6_v12_tta':
        return [ROOT/path for path in chosen['paths']]
    return [selected_cnn()]


def selected_adapter():
    chosen=selection().get('report_adapter')
    if chosen and chosen.get('status')=='selected_experimental':return str(ROOT/chosen['path'])
    return ''
