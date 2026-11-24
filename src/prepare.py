"""Auditoría de las diez categorías originales y división por grupos visuales."""
import csv, hashlib, json, random
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from common import ROOT, CLASSES, SIZE, save_json

KEYS=['battery','biological','cardboard','clothes','glass','metal','paper','plastic','shoes','trash']

def contact_sheets(rows):
    for index,label in enumerate(CLASSES):
        subset=[r for r in rows if r['label']==label]
        chosen=random.Random(42).sample(subset,min(20,len(subset)))
        canvas=Image.new('RGB',(600,550),'white');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(chosen):
            x=(i%5)*120;y=(i//5)*135
            with Image.open(ROOT/r['path']) as im:canvas.paste(im,(x,y))
            draw.text((x,y+97),r['split']+' '+r['group'],fill='black')
            draw.text((x,y+112),label,fill='black')
        canvas.save(ROOT/f'results/contacto_{index+1:02d}.jpg')

def main():
    source=ROOT/'data/archive/original'
    if not all((source/k).is_dir() for k in KEYS):raise SystemExit('Faltan categorías en data/archive/original')
    rows=[];invalid=[];seen={};duplicates=[];conflicts=set()
    for key,label in zip(KEYS,CLASSES):
        for file in sorted((source/key).iterdir()):
            if not file.is_file():continue
            try:
                with Image.open(file) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB');im.load()
                    sha=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
                    if sha in seen:
                        duplicates.append(str(file.relative_to(ROOT)))
                        if seen[sha]['label']!=label:conflicts.add(sha)
                        continue
                    a=np.asarray(im.convert('L').resize((9,8)))
                    dh=sum(int(v)<<i for i,v in enumerate((a[:,1:]>a[:,:-1]).ravel()))
                    target=ROOT/'data/processed'/key/(file.stem+'_'+sha[:12]+'.png')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    ImageOps.pad(im,(SIZE,SIZE),method=Image.Resampling.BILINEAR,color=(128,128,128)).save(target)
                row=dict(path=target.relative_to(ROOT).as_posix(),original=file.relative_to(ROOT).as_posix(),label=label,pixel_sha=sha,dhash=dh)
                seen[sha]=row;rows.append(row)
            except Exception as e:invalid.append(dict(path=str(file),error=str(e)))
    rows=[r for r in rows if r['pixel_sha'] not in conflicts]
    rng=random.Random(42);groups_count={}
    for label in CLASSES:
        subset=[r for r in rows if r['label']==label];parent=list(range(len(subset)))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for i,a in enumerate(subset):
            for j in range(i):
                if (a['dhash']^subset[j]['dhash']).bit_count()<=6:parent[find(i)]=find(j)
        groups={}
        for i,r in enumerate(subset):
            r['group']=f'{CLASSES.index(label)}_{find(i)}';groups.setdefault(r['group'],[]).append(r)
        items=list(groups.values());rng.shuffle(items);items.sort(key=len,reverse=True)
        if len(items)<3:raise SystemExit('Insuficientes grupos independientes en '+label)
        counts=dict(train=0,val=0,test=0);targets={s:p*len(subset) for s,p in [('train',.7),('val',.15),('test',.15)]}
        for i,group in enumerate(items):
            empty=[s for s in counts if counts[s]==0]
            split=empty[0] if len(items)-i==len(empty) else max(counts,key=lambda s:(targets[s]-counts[s])/targets[s])
            for r in group:r['split']=split
            counts[split]+=len(group)
        groups_count[label]=len(items)
    manifest=ROOT/'data/manifest.csv'
    with manifest.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    audit=dict(input_images=sum(1 for k in KEYS for f in (source/k).iterdir() if f.is_file()),usable_images=len(rows),
        exact_duplicates=len(duplicates),invalid=invalid,conflicting_pixel_hashes=sorted(conflicts),
        classes=CLASSES,counts={c:dict(Counter(r['split'] for r in rows if r['label']==c)) for c in CLASSES},
        groups=groups_count,seed=42,dhash_distance=6,source='https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2',
        preprocessing='EXIF, RGB, pad 96x96 bilineal gris. Variaciones aleatorias solo al entrenar.',
        independence='Agrupos dHash heurísticos; identidad física sin verificación exhaustiva.',
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    save_json(ROOT/'results/auditoria.json',audit);save_json(ROOT/'results/duplicados.json',duplicates)
    contact_sheets(rows)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
