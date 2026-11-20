import numpy as np
import torch
from PIL import Image
from chest_xpert.disease_targets import DISEASES,disease_targets
from chest_xpert.checkpoint_engine import prepare_checkpoint_image
from cliq_rrg.stage1 import PrAM


def row(text,mesh=None):
    return {'mesh_terms':mesh or [],'target':{'sections':[{'heading':'Findings','text':text}]}}


def test_negation_unknown_and_uncertainty():
    targets,mask=disease_targets(row('No pleural effusion or pneumothorax. Possible pneumonia.'))
    assert targets[DISEASES.index('Effusion')]==0 and mask[DISEASES.index('Effusion')]==1
    assert targets[DISEASES.index('Pneumothorax')]==0 and mask[DISEASES.index('Pneumothorax')]==1
    assert mask[DISEASES.index('Pneumonia')]==0
    assert mask[DISEASES.index('Hernia')]==0


def test_positive_mesh_and_normal():
    targets,mask=disease_targets(row('Small pleural fluid.', ['Pleural Effusion']))
    assert targets[DISEASES.index('Effusion')]==1 and mask[DISEASES.index('Effusion')]==1
    targets,mask=disease_targets(row('Lungs are clear.', ['normal']))
    assert targets==[0]*14 and mask==[1]*14


def test_no_change_is_not_absence():
    targets,mask=disease_targets(row('No change in cardiomegaly.'))
    assert targets[DISEASES.index('Cardiomegaly')]==1


def test_vectorized_attention_matches_individual_queries_and_gradients():
    torch.manual_seed(1)
    module=PrAM(16,8,4)
    visual=torch.randn(2,9,16,requires_grad=True)
    queries=torch.randn(2,5,8,requires_grad=True)
    separate=torch.stack([module(visual,queries[:,i]) for i in range(5)],dim=1)
    batched=module(visual,queries)
    assert torch.allclose(separate,batched,atol=1e-6)
    g1=torch.autograd.grad(separate.sum(),visual,retain_graph=True)[0]
    g2=torch.autograd.grad(batched.sum(),visual)[0]
    assert torch.allclose(g1,g2,atol=1e-6)


def test_checkpoint_pixel_scale_and_crop(tmp_path):
    path=tmp_path/'image.png';Image.fromarray(np.tile(np.arange(255,dtype=np.uint8),(100,1))).save(path)
    legacy=prepare_checkpoint_image(path,224,1,'resize')
    corrected=prepare_checkpoint_image(path,224,1024,'center')
    assert legacy.shape==corrected.shape==(1,224,224)
    assert -1<=legacy.min()<=legacy.max()<=1
    assert -1024<=corrected.min()<=corrected.max()<=1024
    assert not torch.allclose(legacy,corrected/1024)


def test_legacy_checkpoint_can_be_selected_explicitly(monkeypatch,tmp_path):
    from chest_xpert import model_selection
    path=tmp_path/'original.pth'
    monkeypatch.setenv('CHESTXPERT_CNN_CHECKPOINT',str(path))
    assert model_selection.selected_cnn()==path


def test_corrupt_images_excluded_from_manifests(tmp_path):
    from chest_xpert.training_data import prepare
    root=tmp_path/'data';(root/'images').mkdir(parents=True);(root/'reports').mkdir()
    (root/'images/bad.png').write_bytes(b'not a PNG')
    audit=prepare(root,tmp_path/'out')
    assert audit['counts']['excluded_corrupt_images']==1
    assert audit['excluded_corrupt_image_files']==['images/bad.png']


def test_gradcam_works_with_frozen_encoder_and_head():
    from chest_xpert.explainability import GradCAM
    class Head(torch.nn.Module):
        def forward(self,features):return {'disease_logits':features.mean((2,3))}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=torch.nn.Conv2d(1,4,3);self.alignment=Head()
    model=Model().requires_grad_(False)
    feats,grads=GradCAM(model)._activations_and_gradients(torch.randn(1,1,8,8),1)
    assert feats.shape==grads.shape==(1,4,6,6)
    assert torch.isfinite(grads).all() and grads.abs().sum()>0


def test_clear_of_is_negation_not_positive_disease():
    targets,mask=disease_targets(row('The lungs are clear of focal airspace disease, pneumothorax, or pleural effusion.', ['normal']))
    assert targets==[0.0]*14
    assert mask==[1.0]*14


def test_conditional_recommendation_is_not_a_diagnosis():
    targets,mask=disease_targets(row('If there is concern for pneumonia, additional imaging may be considered.'))
    assert mask[DISEASES.index('Pneumonia')]==0


def test_negation_does_not_cross_sentence_into_a_positive_finding():
    targets,mask=disease_targets(row('Lungs are clear of effusion. Mild cardiomegaly is present.'))
    assert targets[DISEASES.index('Effusion')]==0
    assert targets[DISEASES.index('Cardiomegaly')]==1


def test_per_finding_queries_use_the_matching_classifier_row():
    from cliq_rrg.stage1 import DiseaseAwareAlignment
    head=DiseaseAwareAlignment(num_diseases=2,visual_dim=4,disease_dim=4,pram_heads=1)
    class FixedAttention(torch.nn.Module):
        def forward(self,*args,**kwargs):return torch.tensor([[[1.,2.,3.,4.],[4.,3.,2.,1.]]])
    head.pram=FixedAttention()
    head.disease_classifier.weight.data.copy_(torch.tensor([[1.,0.,0.,0.],[0.,0.,0.,1.]]))
    head.disease_classifier.bias.data.zero_()
    visual=torch.zeros(1,4,2,2)
    assert torch.allclose(head(visual)['disease_logits'],torch.tensor([[2.5,2.5]]))
    head.disease_specific_queries=True
    assert torch.allclose(head(visual)['disease_logits'],torch.tensor([[1.,1.]]))
