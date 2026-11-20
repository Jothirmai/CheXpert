import io,tarfile
from pathlib import Path
import pytest
from PIL import Image
from chest_xpert.study_images import resolve_study_images,IncompleteStudyError


def png(value):
    stream=io.BytesIO();Image.new('L',(8,8),value).save(stream,format='PNG');return stream.getvalue()


def test_full_study_resolution_rejects_missing_view(tmp_path):
    report=tmp_path/'study.xml';images=tmp_path/'images';images.mkdir()
    report.write_text('<eCitation><parentImage id="CXR1_IM-1-1001"/><parentImage id="CXR1_IM-1-2001"/></eCitation>')
    (images/'CXR1_IM-1-1001.png').write_bytes(png(20))
    with pytest.raises(IncompleteStudyError,match='2001'):
        resolve_study_images(report,images)
    (images/'CXR1_IM-1-2001.png').write_bytes(png(80))
    assert [p.name for p in resolve_study_images(report,images)]==['CXR1_IM-1-1001.png','CXR1_IM-1-2001.png']


def test_downloader_repairs_partial_directory_without_replacing_valid_images(tmp_path):
    from cliq_rrg.dataset import IUXRayDownloader
    dataset=IUXRayDownloader(tmp_path)
    (dataset.reports_dir/'1.xml').write_text('<eCitation><parentImage id="CXR1_IM-1-1001"/><parentImage id="CXR1_IM-1-2001"/></eCitation>')
    kept=dataset.images_dir/'CXR1_IM-1-1001.png';kept.write_bytes(png(20))
    with tarfile.open(tmp_path/'NLMCXR_png.tgz','w:gz') as archive:
        for name,data in [('CXR1_IM-1-1001.png',png(200)),('CXR1_IM-1-2001.png',png(80))]:
            info=tarfile.TarInfo('images/'+name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    dataset.download_images()
    assert Image.open(kept).getpixel((0,0))==20
    assert Image.open(dataset.images_dir/'CXR1_IM-1-2001.png').getpixel((0,0))==80
