"""Entrena desde inicialización aleatoria. Selección por validación; prueba se evalúa después."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import csv
import hashlib
import json
import platform
import random
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from common import ROOT, CLASSES, SIZE, ResiduoCNN, preprocess, save_json


class Images(Dataset):
    def __init__(self, rows, augment=False):
        self.rows, self.augment = rows, augment
        def read(r):
            with Image.open(ROOT/r['path']) as im:
                return im.convert('RGB').copy()
        print(f'Leyendo {len(rows)} imágenes de la partición...',flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.cache = list(pool.map(read, rows))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image = self.cache[index].copy()
        if self.augment:
            if random.random()<.5:
                image = ImageOps.mirror(image)
            image = image.rotate(random.uniform(-18,18),resample=Image.Resampling.BILINEAR,
                                 fillcolor=(128,128,128))
            image = ImageEnhance.Brightness(image).enhance(random.uniform(.85,1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(.9,1.1))
        return preprocess(image), CLASSES.index(self.rows[index]['label'])


def pass_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, correct, total = 0.,0,0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits=model(x); loss=criterion(logits,y)
            if optimizer is not None:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss+=loss.item()*len(y); correct+=(logits.argmax(1)==y).sum().item(); total+=len(y)
    return total_loss/total,correct/total


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    p.add_argument('--overwrite',action='store_true',help='Reemplaza una ejecución anterior explícitamente')
    args=p.parse_args()
    checkpoint_path=ROOT/'models/mejor_modelo.pt'
    if checkpoint_path.exists() and not args.overwrite:
        raise SystemExit('Ya existe un modelo. Copia sus resultados antes de usar --overwrite.')
    if (ROOT/'results/metricas.json').exists():
        raise SystemExit('Prueba ya evaluada. Conserva esta ejecución; usa otra carpeta para un experimento nuevo.')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    rows=list(csv.DictReader((ROOT/'data/manifest.csv').open(encoding='utf-8')))
    # La prueba no participa ni en el entrenamiento ni en la selección.
    train=[r for r in rows if r['split']=='train']; val=[r for r in rows if r['split']=='val']
    generator=torch.Generator().manual_seed(args.seed)
    trainloader=DataLoader(Images(train,True),batch_size=args.batch_size,shuffle=True,generator=generator,num_workers=0)
    valloader=DataLoader(Images(val),batch_size=args.batch_size,num_workers=0)
    model=ResiduoCNN().to(device)  # únicamente inicialización aleatoria
    counts=Counter(r['label'] for r in train)
    if any(counts[c]==0 for c in CLASSES):
        raise SystemExit('Faltan clases en entrenamiento.')
    weights=torch.tensor([len(train)/(len(CLASSES)*counts[c]) for c in CLASSES],device=device)
    criterion=nn.CrossEntropyLoss(weight=weights)
    validation_criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4)
    config=vars(args)|dict(device_used=device,classes=CLASSES,size=SIZE,framework='PyTorch',
             torch_version=str(torch.__version__),python=platform.python_version(),platform=platform.platform(),
             gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
             pretrained=False,initialization='Aleatoria de PyTorch; ninguna carga de pesos externos',
             parameters=sum(p.numel() for p in model.parameters()),class_weights=weights.cpu().tolist(),
             manifest_sha256=hashlib.sha256((ROOT/'data/manifest.csv').read_bytes()).hexdigest(),
             selection='Mínima entropía cruzada de validación, no ponderada',
             preprocessing='EXIF, RGB, pad gris 96x96 bilineal, CHW float32 /255',
             train_augmentation='Reflejo horizontal p=.5; rotación ±18°; brillo .85–1.15; contraste .9–1.1')
    save_json(ROOT/'results/configuracion.json',config)
    (ROOT/'models').mkdir(exist_ok=True)
    (ROOT/'results/arquitectura.txt').write_text(str(model),encoding='utf-8')
    history=[]; best=float('inf'); wait=0; start=time.perf_counter()
    print(f'Entrenando {config["parameters"]:,} parámetros desde cero en {device}',flush=True)
    for epoch in range(1,args.epochs+1):
        loss,acc=pass_epoch(model,trainloader,criterion,device,optimizer)
        vl,va=pass_epoch(model,valloader,validation_criterion,device)
        history.append(dict(epoch=epoch,train_loss=loss,train_accuracy=acc,val_loss=vl,val_accuracy=va))
        print(f'Época {epoch:03d}: train={acc:.3f}, val={va:.3f}, val_loss={vl:.4f}',flush=True)
        if vl < best-1e-5:
            best=vl; wait=0
            torch.save(dict(state_dict=model.state_dict(),classes=CLASSES,size=SIZE,
                            epoch=epoch,val_loss=vl,pretrained=False,
                            manifest_sha256=config['manifest_sha256']),checkpoint_path)
        else:
            wait+=1
        save_json(ROOT/'results/historial.json',history)
        if wait>=args.patience:
            break
    config['elapsed_seconds']=time.perf_counter()-start
    config['epochs_executed']=len(history)
    config['best_epoch']=min(history,key=lambda r:r['val_loss'])['epoch']
    config['checkpoint_sha256']=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    save_json(ROOT/'results/configuracion.json',config)
    with (ROOT/'results/historial.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    print('Modelo guardado. Ejecutar evaluate.py una vez terminada la selección con validación.')


if __name__=='__main__':
    main()
