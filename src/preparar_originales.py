"""Recupera y verifica los originales de Garbage Classification V2."""
import argparse
import csv
import hashlib
import io
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
from zipfile import ZipFile

from PIL import Image, ImageOps

from common import ROOT, original_image_path


DOWNLOAD = 'https://www.kaggle.com/api/v1/datasets/download/sumn2u/garbage-classification-v2'


def verify(content, row):
    with Image.open(io.BytesIO(content)) as raw:
        image = ImageOps.exif_transpose(raw).convert('RGB')
        fingerprint = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    if fingerprint != row['pixel_sha']:
        raise ValueError(f"La imagen no coincide con el manifiesto: {row['original']}")


def recover(rows, origin, verify_only):
    missing = [row for row in rows if original_image_path(ROOT, row['original']) is None]
    if verify_only and missing:
        raise FileNotFoundError(f"Faltan {len(missing)} originales. Primer archivo: {missing[0]['original']}")
    if missing and not origin:
        with tempfile.TemporaryDirectory(prefix='dataset_residuos_') as folder:
            archive = Path(folder)/'kaggle.zip'
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                            '--retry', '2', '--max-time', '1800', '--output', str(archive),
                            DOWNLOAD], check=True)
            copy_missing(missing, archive)
    elif missing:
        copy_missing(missing, origin)
    for row in rows:
        path = original_image_path(ROOT, row['original'])
        if path is None:
            raise FileNotFoundError(row['original'])
        verify(path.read_bytes(), row)


def copy_missing(rows, origin):
    if origin.is_dir():
        for row in rows:
            rel = Path(row['original'])
            source = origin/rel.parent.name/rel.name
            write_original(source.read_bytes(), row)
        return
    with ZipFile(origin) as archive:
        originals = {}
        for name in archive.namelist():
            parts = PurePosixPath(name).parts
            if len(parts) >= 3 and parts[-3] == 'original' and not name.endswith('/'):
                key = parts[-2:]
                if key in originals:
                    raise ValueError(f'Ruta de imagen duplicada en el ZIP: {name}')
                originals[key] = name
        for row in rows:
            key = PurePosixPath(row['original']).parts[-2:]
            if key not in originals:
                raise FileNotFoundError(f"El ZIP de Kaggle no incluye {row['original']}")
            write_original(archive.read(originals[key]), row)


def write_original(content, row):
    verify(content, row)
    target = ROOT/row['original']
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix='.tmp') as temp:
        temp.write(content)
    Path(temp.name).replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--origen', type=Path, help='ZIP de Kaggle o carpeta original con las diez clases.')
    options.add_argument('--verificar', action='store_true', help='Verifica la copia local sin descargar nada.')
    args = parser.parse_args()
    with (ROOT/'data/manifest.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    recover(rows, args.origen, args.verificar)
    print(f'Originales verificados contra el manifiesto: {len(rows)}')


if __name__ == '__main__':
    main()
