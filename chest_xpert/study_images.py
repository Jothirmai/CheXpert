"""Resolve every original view referenced by a dataset study, without filename view guesses."""
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

class IncompleteStudyError(ValueError):
    pass


def index_study_images(images_dir):
    return {p.stem:p.resolve() for p in Path(images_dir).rglob('*.png')}


def resolve_study_images(report_xml, images_dir, image_index=None):
    root=ET.parse(report_xml).getroot()
    expected=[node.get('id') for node in root.findall('parentImage') if node.get('id')]
    if not expected:raise IncompleteStudyError('Study report does not identify its original images')
    paths=image_index if image_index is not None else index_study_images(images_dir)
    resolved=[];missing=[]
    for identifier in expected:
        path=paths.get(identifier)
        try:
            if path is None:raise FileNotFoundError(identifier)
            with Image.open(path) as image:image.verify()
            resolved.append(path)
        except (OSError,ValueError):missing.append(identifier)
    if missing:
        raise IncompleteStudyError('Cannot compare a full-study reference with incomplete input views. Missing/corrupt: '+', '.join(missing))
    return resolved
