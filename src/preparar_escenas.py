"""Descarga ejemplos anotados de TACO. Sólo usa lotes distintos al de la demostración."""
import argparse
import hashlib
import io
import json
import random
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/taco'
INDEX = 'https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json'
# Identificadores de las categorías en el índice original de TACO.
CATEGORIES = {**dict.fromkeys([4,5,7,21,24,27,29,36,40,47],7),
              **dict.fromkeys([6,9,23,26],4), **dict.fromkeys([0,8,10,11,12,28,52],5),
              **dict.fromkeys([13,14,15,17,18,19],2), **dict.fromkeys([30,32,33,34],6),
              25:1, 1:0, 53:8}


def collect(index):
    annotations = defaultdict(list)
    for ann in index['annotations']:
        annotations[ann['image_id']].append(ann)
    pools = defaultdict(list)
    for image in index['images']:
        batch = image['file_name'].split('/')[0]
        url = image.get('flickr_url')
        if batch == 'batch_8' or not url or urlparse(url).hostname != 'img.tacodataset.org':
            continue
        split = 'val' if batch=='batch_5' else 'test' if batch in ('batch_9','batch_10') else 'train'
        w,h=image['width'],image['height']
        found = []
        significant = set()
        for ann in annotations[image['id']]:
            x,y,bw,bh = ann['bbox']
            frac = bw*bh/(w*h)
            label = CATEGORIES.get(ann['category_id'])
            if frac>.013:
                significant.add(label)
            if label is not None and frac>=.027 and min(bw/w,bh/h)>.04:
                found.append((frac,label,[x/w,y/h,(x+bw)/w,(y+bh)/h]))
        if not found:
            continue
        found.sort(reverse=True)
        _,label,box=found[0]
        pools[split,label].append(dict(image_id=image['id'],source=image['file_name'],url=url,
                label=label,box=box,whole=significant=={label},license=image['license'],split=split))
    rng=random.Random(42)
    selected=[]
    for (split,label),items in sorted(pools.items()):
        rng.shuffle(items)
        cap={'train':100,'val':22,'test':25}[split]
        selected.extend(items[:cap])
    return selected


def download(row):
    folder = DATA/'images'
    path = folder/(str(row['image_id'])+'.jpg')
    try:
        if not path.exists():
            response = subprocess.run(['curl','--fail','--silent','--show-error','--location',
                    '--max-time','45','--retry','1','--max-filesize','32000000',row['url']],
                    capture_output=True,check=True)
            content=response.stdout
            with Image.open(io.BytesIO(content)) as im:
                converted=ImageOps.exif_transpose(im).convert('RGB')
                converted.thumbnail((800,800),Image.Resampling.LANCZOS)
                converted.save(path.with_suffix('.tmp'),format='JPEG',quality=95)
            path.with_suffix('.tmp').replace(path)
        row['path']=str(path.relative_to(ROOT).as_posix())
        return row
    except (OSError,ValueError,subprocess.CalledProcessError) as err:
        print(row['image_id'],str(err),flush=True)
        return None


def main():
    parser=argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--desde-manifiesto', action='store_true',
                      help='Descarga las fotos versionadas sin modificar el manifiesto (opción predeterminada).')
    mode.add_argument('--nuevo-subconjunto', action='store_true',
                      help='Selecciona otro subconjunto y reemplaza el manifiesto.')
    mode.add_argument('--index', type=Path, help='Genera otro manifiesto desde anotaciones locales.')
    args=parser.parse_args()
    if not (args.nuevo_subconjunto or args.index):
        rows = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))['rows']
        if not rows or any(
            row['path'] != f"data/taco/images/{row['image_id']}.jpg"
            or urlparse(row['url']).scheme != 'https'
            or urlparse(row['url']).hostname != 'img.tacodataset.org'
            for row in rows
        ):
            parser.error('El manifiesto contiene rutas o direcciones inesperadas.')
        (DATA/'images').mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=10) as pool:
            saved = list(pool.map(download, rows))
        missing = [row['path'] for row, result in zip(rows, saved) if result is None]
        if missing:
            parser.error(f'Faltan {len(missing)} fotos de TACO. Copia desde el respaldo o reintenta: '
                         + ', '.join(missing[:8]))
        print('Fotos disponibles del manifiesto:', len(rows))
        return
    content = args.index.read_bytes() if args.index else subprocess.run(
        ['curl','--fail','--silent','--show-error','--location','--max-time','60',INDEX],
        capture_output=True,check=True).stdout
    rows=collect(json.loads(content))
    print('Fotos previstas:',len(rows),flush=True)
    (DATA/'images').mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=10) as pool:
        saved=[r for r in pool.map(download,rows) if r]
    if len(saved)<.85*len(rows):
        raise RuntimeError('Descarga incompleta; se puede reintentar usando las fotos guardadas.')
    (DATA/'manifest.json').write_text(json.dumps(dict(index=INDEX,index_sha256=hashlib.sha256(content).hexdigest(),
            excluded_batch='batch_8',rows=saved),ensure_ascii=False,indent=2),encoding='utf8')
    print('Fotos disponibles:',len(saved),flush=True)


if __name__=='__main__': main()

