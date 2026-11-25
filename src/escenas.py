"""Continúa el entrenamiento propio con fotos reales y recortes anotados de TACO."""
import copy
import hashlib
import json

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

from common import ROOT, CLASSES, save_json
from materiales import read_images, augment
from objetos import CHECKPOINT as BASE, ObjectCNN, calibrate, predict, measure

CHECKPOINT = ROOT/'models/escenas.pt'
REPORT = ROOT/'results/escenas.json'


def taco_data():
    rows=json.loads((ROOT/'data/taco/manifest.json').read_text(encoding='utf8'))['rows']
    missing = [row['path'] for row in rows if not (ROOT/row['path']).is_file()]
    if missing:
        raise FileNotFoundError(f'Faltan {len(missing)} fotos de TACO. Copia el subconjunto o ejecuta '
                                'python src/preparar_escenas.py --desde-manifiesto.')
    parts={}
    for split in ('train','val','test'):
        images,labels,views=[],[],[]
        for row in rows:
            if row['split'] != split:
                continue
            if row['source'].startswith('batch_8/'):
                raise ValueError('El lote de demostración debe permanecer reservado.')
            with Image.open(ROOT/row['path']) as im:
                w,h=im.size
                x,y,r,b=row['box']
                margin=.12*max((r-x)*w,(b-y)*h)
                box=(max(0,int(x*w-margin)),max(0,int(y*h-margin)),
                     min(w,int(r*w+margin+1)),min(h,int(b*h+margin+1)))
                samples=[('recorte', im.crop(box))]
                if row['whole']:
                    samples.append(('completa',im))
                for view,sample in samples:
                    img=ImageOps.pad(sample,(96,96),Image.Resampling.BILINEAR,color=(128,128,128))
                    images.append(np.array(img.convert('RGB')))
                    labels.append(row['label'])
                    views.append(view)
        parts[split]=(torch.from_numpy(np.stack(images)).permute(0,3,1,2),torch.tensor(labels),views)
    return parts


def exterior_scores(p,y,views):
    result={}
    for name in ['recorte','completa']:
        idx=[i for i,v in enumerate(views) if v==name]
        accuracy = (p[idx].argmax(1)==y[idx]).float().mean().item() if idx else None
        result[name]=dict(n=len(idx),accuracy=accuracy)
    result['accuracy']=(p.argmax(1)==y).float().mean().item()
    return result


def main():
    torch.manual_seed(43)
    torch.set_num_threads(4)
    exterior=taco_data()
    original,_=read_images()
    train,_,labels=original['train']
    val,_,truth=original['val']
    add,targets,_=exterior['train']
    scene_val,scene_y,views=exterior['val']
    model=ObjectCNN().to(memory_format=torch.channels_last)
    start=torch.load(BASE,map_location='cpu',weights_only=True)
    model.load_state_dict(start['state_dict'])
    model.eval()
    val_floor=measure(predict(val,model.probabilities),truth)['macro_f1']
    initial_ext=exterior_scores(predict(scene_val,model.probabilities),scene_y,views)
    best=val_floor+.45*initial_ext['accuracy']
    best_epoch=0
    best_state=copy.deepcopy(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.005)
    epochs=12
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.000015)
    history=[]
    for epoch in range(1,epochs+1):
        model.train()
        loss_sum=0.
        sample=torch.randperm(len(train))[:4608].split(72)
        for idx in sample:
            j=torch.randint(len(add),(24,))
            x=torch.cat([train[idx],add[j]]).float()/255
            y=torch.cat([labels[idx],targets[j]])
            x=augment(x).contiguous(memory_format=torch.channels_last)
            loss=F.cross_entropy(model(x),y,label_smoothing=.025)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum+=loss.item()
        schedule.step()
        model.eval()
        f1=measure(predict(val,model.probabilities),truth)['macro_f1']
        ext=exterior_scores(predict(scene_val,model.probabilities),scene_y,views)
        score=f1+.45*ext['accuracy']
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(sample),val_macro_f1=f1,external_val=ext))
        print(epoch,round(f1,3),round(ext['accuracy'],3),flush=True)
        if f1 >= val_floor-.018 and score>best:
            best=score
            best_epoch=epoch
            best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    val_probs=predict(val,model.probabilities)
    ext_probs=predict(scene_val,model.probabilities)
    thresholds=calibrate(torch.cat([val_probs,ext_probs]),torch.cat([truth,scene_y])).clamp(min=.5)
    imgs,_,y=original['test']
    test_probs=predict(imgs,model.probabilities)
    ext_imgs,ext_y,views=exterior['test']
    ext_new=exterior_scores(predict(ext_imgs,model.probabilities),ext_y,views)
    model.load_state_dict(start['state_dict'])
    ext_base=exterior_scores(predict(ext_imgs,model.probabilities),ext_y,views)
    tmp=CHECKPOINT.with_suffix('.pt.tmp')
    torch.save(dict(state_dict=best_state,classes=CLASSES,thresholds=thresholds.tolist(),epoch=best_epoch),tmp)
    tmp.replace(CHECKPOINT)
    save_json(REPORT,dict(classes=CLASSES,seed=43,base_checkpoint_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
              checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),selected_epoch=best_epoch,
              external_views={s:len(v[1]) for s,v in exterior.items()}, history=history,thresholds=thresholds.tolist(),
              initial_external_val=initial_ext,initial_internal_val_macro_f1=val_floor,
              test=measure(test_probs,y),selective=measure(test_probs,y,thresholds),
              external_test_before=ext_base,external_test_after=ext_new))


if __name__=='__main__': main()
