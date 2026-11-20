"""IU X-Ray (Demner-Fushman et al., 2015) dataset downloader and loader.

Downloads images and reports from OpenI (NLM), extracts them, and builds a
PyTorch Dataset with 7:2:1 train/val/test splits.
"""

from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import requests
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# IU X-Ray dataset URLs (OpenI / National Library of Medicine)
IU_XRAY_IMAGES_URL = (
    "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"
)
IU_XRAY_REPORTS_URL = (
    "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
)

# 7:2:1 train/val/test split following prior work
SPLIT_TRAIN = 0.7
SPLIT_VAL = 0.2
SPLIT_TEST = 0.1

# IU X-Ray finding labels (14 common thoracic findings)
IU_FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural Thickening", "Pneumonia", "Pneumothorax",
]


@dataclass
class IUSample:
    image_path: str
    report_text: str
    findings: list[str] = field(default_factory=list)
    view: str = ""
    patient_id: str = ""
    study_id: str = ""


class IUXRayDownloader:
    """Download and extract IU X-Ray dataset from OpenI."""

    def __init__(self, root: str | Path = "./data/iu_xray"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.root / "images"
        self.reports_dir = self.root / "reports"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def download_images(self, force: bool = False) -> Path:
        zip_path = self.root / "NLMCXR_png.tgz"
        # A nonempty folder may be an interrupted extraction; verify report coverage.
        import xml.etree.ElementTree as ET
        expected=set()
        for report in self.reports_dir.rglob("*.xml"):
            try:expected.update(node.get("id") for node in ET.parse(report).getroot().findall("parentImage") if node.get("id"))
            except ET.ParseError:continue
        valid={}
        for image_path in self.images_dir.rglob("*.png"):
            try:
                with Image.open(image_path) as image:image.verify()
                valid[image_path.stem]=image_path
            except (OSError,ValueError):continue
        if not force and expected and expected.issubset(valid):
            print(f"All {len(expected)} referenced image files verified")
            return self.images_dir
        cached=Path.home()/".cache/chestxpert/NLMCXR_png.tgz"
        if not zip_path.exists() and cached.exists():zip_path=cached
        if not force and zip_path.exists():
            print(f"Images archive already exists: {zip_path}")
        else:
            print(f"Downloading IU X-Ray images from {IU_XRAY_IMAGES_URL} ...")
            r = requests.get(IU_XRAY_IMAGES_URL, stream=True, timeout=600)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

        import io,json
        archive_ids=set()
        with tarfile.open(zip_path, "r|gz") as tf:
            for member in tf:
                name=Path(member.name).name
                if not member.isfile() or not re.match(r"^CXR\d.*\.png$",name):continue
                identifier=Path(name).stem;archive_ids.add(identifier)
                if not force and identifier in valid:continue
                data=tf.extractfile(member).read()
                try:
                    with Image.open(io.BytesIO(data)) as image:image.verify()
                except (OSError,ValueError):continue
                target=valid.get(identifier,self.images_dir/name)
                temporary=target.with_suffix('.downloadtmp');temporary.write_bytes(data);temporary.replace(target)
                valid[identifier]=target
        missing=sorted(expected-set(valid))
        (self.root/'image_inventory.json').write_text(json.dumps({'referenced_images':len(expected),'archive_images':len(archive_ids),'verified_images':len(valid),'missing_referenced_images':missing,'complete_for_reports':not missing},indent=2))
        print(f"Verified images: {len(valid)}; missing report-referenced images: {len(missing)}")
        return self.images_dir

    def download_reports(self, force: bool = False) -> Path:
        zip_path = self.root / "NLMCXR_reports.tgz"
        # Check if reports already exist directly in reports_dir (from prior download)
        if not force and any(self.reports_dir.iterdir()):
            xml_count = len(list(self.reports_dir.rglob("*.xml")))
            print(f"Reports already exist: {self.reports_dir} ({xml_count} XML files)")
            return self.reports_dir
        if not force and zip_path.exists():
            print(f"Reports archive already exists: {zip_path}")
        else:
            print(f"Downloading IU X-Ray reports from {IU_XRAY_REPORTS_URL} ...")
            r = requests.get(IU_XRAY_REPORTS_URL, stream=True, timeout=600)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

        with tarfile.open(zip_path, "r:gz") as tf:
            tf.extractall(self.reports_dir)
        print(f"Extracted reports to {self.reports_dir}")
        return self.reports_dir

    def download_all(self, force: bool = False):
        self.download_images(force=force)
        self.download_reports(force=force)


def parse_reports(reports_dir: str | Path) -> dict[str, str]:
    """Parse IU X-Ray report XML/text files into {study_id: report_text}."""
    from chest_xpert.training_data import parse_clinical_xml, clinical_report_text
    import xml.etree.ElementTree as ET
    reports = {}
    for path in Path(reports_dir).rglob("*.xml"):
        try:
            record = parse_clinical_xml(path)
        except (ET.ParseError, OSError):
            continue
        text = clinical_report_text(record)
        if text:
            reports[path.stem] = text
    return reports


def build_samples(images_dir: str | Path, reports: dict[str, str]) -> list[IUSample]:
    """Match images to reports by study ID."""
    samples: list[IUSample] = []
    idir = Path(images_dir)
    for img_path in sorted(idir.rglob("*.png")):
        # Skip empty/corrupted files
        if img_path.stat().st_size == 0:
            continue
        # Verify image is openable
        try:
            with Image.open(img_path) as test_img:
                test_img.verify()
        except Exception:
            continue
        stem = img_path.stem
        m = re.match(r"(CXR\d+)", stem)
        study_id = m.group(1) if m else stem
        # IU X-Ray report XML files are named by number only (e.g. 2845.xml),
        # while image filenames include a "CXR" prefix (e.g. CXR2845_IM-...png).
        # Try both the full ID and the number-only variant.
        report_text = reports.get(study_id, "")
        if not report_text and study_id.startswith("CXR"):
            report_text = reports.get(study_id[3:], "")
        view = "PA" if "PA" in stem.upper() else ("Lateral" if "LAT" in stem.upper() else "")
        samples.append(IUSample(
            image_path=str(img_path),
            report_text=report_text,
            view=view,
            study_id=study_id,
        ))
    return samples


def train_val_test_split(samples: list[IUSample], seed: int = 42):
    """7:2:1 split following prior work."""
    import random
    groups = {}
    for sample in samples:
        groups.setdefault(sample.study_id, []).append(sample)
    identifiers = sorted(groups)
    random.Random(seed).shuffle(identifiers)
    n = len(identifiers)
    boundaries = (int(n * SPLIT_TRAIN), int(n * (SPLIT_TRAIN + SPLIT_VAL)))
    return tuple([sample for key in keys for sample in groups[key]] for keys in
        (identifiers[:boundaries[0]], identifiers[boundaries[0]:boundaries[1]], identifiers[boundaries[1]:]))


class IURadiologyDataset(Dataset):
    """PyTorch Dataset for IU X-Ray images + reports."""

    def __init__(self, samples: list[IUSample], transform=None, resolution: int = 224):
        self.samples = samples
        self.transform = transform or transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        img = self.transform(img)
        return {
            "image": img,
            "report": s.report_text,
            "findings": s.findings,
            "view": s.view,
            "study_id": s.study_id,
            "image_path": s.image_path,
        }
