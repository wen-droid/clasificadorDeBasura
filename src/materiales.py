"""CNN de cinco materiales y una salida auxiliar para ejemplos fuera de alcance."""
import argparse
import csv
import copy
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

from common import ROOT, CLASSES as ORIGINAL_CLASSES, load_model, save_json

LABELS = ['Orgánico', 'Papel y cartón', 'Vidrio', 'Metal', 'Plástico', 'Fuera de las cinco categorías']
# Índices en el mismo orden que ORIGINAL_CLASSES.
GROUP = [5, 0, 1, 5, 2, 3, 1, 4, 5, 5]
CHECKPOINT = ROOT / 'models/materiales.pt'
REPORT = ROOT / 'results/materiales.json'


class MaterialCNN(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [3, 16, 32, 48, 64]
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU())
            for a, b in zip(channels, channels[1:])
        ])
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.2), nn.Linear(64 * 9, len(LABELS)))

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        for block in self.blocks:
            x = F.max_pool2d(block(x), 2)
        return self.head(x)


def read_images():
    partitions = {s: ([], [], []) for s in ('train', 'val', 'test')}
    missing = []
    groups = {}
    with (ROOT / 'data/manifest.csv').open(encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            split = row['split']
            if groups.setdefault(row['group'], split) != split:
                raise ValueError('Grupo compartido entre particiones')
            original = Path(row['original'])
            candidates = [ROOT / original, ROOT / 'data/archive/data' / original.parent.name / original.name]
            file = next((p for p in candidates if p.is_file()), None)
            if file is None:
                missing.append(row['original'])
                continue
            with Image.open(file) as raw:
                image = ImageOps.exif_transpose(raw).convert('RGB')
                if hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest() != row['pixel_sha']:
                    raise ValueError(f'Imagen diferente del manifiesto: {file}')
                image = ImageOps.pad(image, (96, 96), Image.Resampling.BILINEAR, color=(128, 128, 128))
                images, labels, originals = partitions[split]
                images.append(np.asarray(image))
                label = ORIGINAL_CLASSES.index(row['label'])
                labels.append(GROUP[label])
                originals.append(label)
    data = {}
    for split, (imgs, labels, original) in partitions.items():
        if set(labels) != set(range(len(LABELS))):
            raise ValueError(f'Faltan categorías en la partición {split}. Revisa las fotos originales.')
        data[split] = (torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2), torch.tensor(labels), torch.tensor(original))
        imgs.clear()
    return data, missing


def augment(x):
    n = len(x)
    def uniform(low, high):
        return torch.empty(n).uniform_(low, high)
    angle = uniform(-.22, .22)
    scale = uniform(.85, 1.12)
    flip = torch.where(torch.rand(n) < .5, -1., 1.)
    transform = torch.stack((angle.cos()*scale*flip, -angle.sin()*scale, uniform(-.09,.09),
                             angle.sin()*scale*flip, angle.cos()*scale, uniform(-.09,.09)), 1).reshape(n,2,3)
    grid = F.affine_grid(transform, x.size(), align_corners=False)
    x = F.grid_sample(x - .5, grid, align_corners=False) + .5
    return (x * uniform(.85,1.15)[:,None,None,None] + uniform(-.08,.08)[:,None,None,None]).clamp_(0,1)


@torch.inference_mode()
def scores(model, images):
    model.eval()
    return torch.cat([model(batch.float().div(255)).softmax(1) for batch in images.split(64)])


def metrics(pred, y):
    matrix = torch.bincount(y*6 + pred, minlength=36).reshape(6,6)
    return dict(n=len(y), accuracy=(pred==y).double().mean().item(),
                macro_f1=(2*matrix.diag()/(matrix.sum(0)+matrix.sum(1)).clamp_min(1)).mean().item(),
                confusion=matrix.tolist())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=30)
    args=parser.parse_args()
    if args.epochs <= 0:
        parser.error('--epochs debe ser positivo')
    torch.manual_seed(42)
    torch.set_num_threads(4)
    data, missing = read_images()
    train, labels, _ = data['train']
    val, truth, _ = data['val']
    print('Particiones:', {k:len(v[1]) for k,v in data.items()}, flush=True)
    model=MaterialCNN().to(memory_format=torch.channels_last)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.005)
    schedule=torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.002, epochs=args.epochs,
                                                  steps_per_epoch=math.ceil(len(train)/96),pct_start=.15)
    # Atenuar el predominio de la salida auxiliar y de papel/cartón.
    weights=(labels.bincount().float().mean()/labels.bincount()).sqrt()
    best=-1
    history=[]
    for epoch in range(1,args.epochs+1):
        model.train()
        loss_total=0.
        for idx in torch.randperm(len(train)).split(96):
            batch=augment(train[idx].float().div(255)).contiguous(memory_format=torch.channels_last)
            loss=F.cross_entropy(model(batch),labels[idx],weight=weights,label_smoothing=.03)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            loss_total+=loss.item()*len(idx)
        p=scores(model,val)
        m=metrics(p.argmax(1),truth)
        history.append(dict(epoch=epoch,train_loss=loss_total/len(train),val_macro_f1=m['macro_f1'],val_accuracy=m['accuracy']))
        print(f"Época {epoch}: F1 validación {m['macro_f1']:.3f}",flush=True)
        if m['macro_f1']>best:
            best=m['macro_f1']
            checkpoint=dict(state_dict=copy.deepcopy(model.state_dict()),epoch=epoch,classes=LABELS)
    model.load_state_dict(checkpoint['state_dict'])
    p=scores(model,val)
    conf, pred=p.max(1)
    # Máxima cobertura entre umbrales con al menos 85 % de acierto en materiales aceptados.
    threshold=1.
    for t in range(0,101):
        accept=(pred<5)&(conf>=t/100)
        if accept.any() and (pred[accept]==truth[accept]).float().mean()>=.85:
            threshold=t/100
            break
    checkpoint['threshold']=threshold
    images,y,source=data['test']
    p=scores(model,images)
    conf,pred=p.max(1)
    accept=(pred<5)&(conf>=threshold)
    old,_=load_model()
    baseline=torch.tensor(GROUP)[scores(old,images).argmax(1)]
    report=dict(classes=LABELS,seed=42,epochs=args.epochs,selected_epoch=checkpoint['epoch'],
                splits={k:len(v[1]) for k,v in data.items()},missing=missing,history=history,
                manifest_sha256=hashlib.sha256((ROOT/'data/manifest.csv').read_bytes()).hexdigest(),
                baseline=metrics(baseline,y),test=metrics(pred,y),threshold=threshold,
                accepted_materials=int(accept.sum()),correct_accepted=int(((pred==y)&accept).sum()),
                known_total=int((y<5).sum()),known_accepted=int(((y<5)&accept).sum()),
                actual_shoes=int((source==8).sum()),shoes_accepted_as_material=int(((source==8)&accept).sum()))
    temporary = CHECKPOINT.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(CHECKPOINT)
    report['checkpoint_sha256'] = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    save_json(REPORT,report)
    print(report['baseline'],report['test'],flush=True)


if __name__=='__main__':
    main()

