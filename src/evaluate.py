"""Evaluación final independiente del ajuste y creación de evidencias gráficas."""
import argparse
import csv
import hashlib
import json
import time
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, CLASSES, load_model, save_json
from train import Images
from torch.utils.data import DataLoader


def classification_metrics(y, predicted):
    cm=np.zeros((len(CLASSES),len(CLASSES)),dtype=int)
    for a,b in zip(y,predicted):
        cm[a,b]+=1
    per_class=[]
    for i,label in enumerate(CLASSES):
        tp=int(cm[i,i]); support=int(cm[i].sum()); guesses=int(cm[:,i].sum())
        precision=tp/guesses if guesses else 0.
        recall=tp/support if support else 0.
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        per_class.append(dict(class_name=label,precision=precision,recall=recall,f1=f1,support=support))
    return dict(accuracy=float(np.trace(cm)/cm.sum()),macro_f1=float(np.mean([r['f1'] for r in per_class])),
                confusion_matrix=cm.tolist(),per_class=per_class,classes=CLASSES)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repeat-same-model',action='store_true');args=p.parse_args()
    model,checkpoint=load_model()
    fingerprint=hashlib.sha256((ROOT/'models/mejor_modelo.pt').read_bytes()).hexdigest()
    previous=ROOT/'results/metricas.json'
    if previous.exists():
        old=json.loads(previous.read_text(encoding='utf-8'))
        if not args.repeat_same_model or old['checkpoint_sha256']!=fingerprint:
            raise SystemExit('La prueba ya fue evaluada. No usarla para seleccionar otro modelo.')
    manifest_hash=hashlib.sha256((ROOT/'data/manifest.csv').read_bytes()).hexdigest()
    if manifest_hash!=checkpoint['manifest_sha256']:
        raise SystemExit('El manifiesto cambió desde entrenamiento. Restaurar la partición original.')
    torch.set_num_threads(4)
    rows=list(csv.DictReader((ROOT/'data/manifest.csv').open(encoding='utf-8')))
    test=[r for r in rows if r['split']=='test']
    if set(r['group'] for r in test)&set(r['group'] for r in rows if r['split']!='test'):
        raise SystemExit('Fuga: hay grupos compartidos entre prueba y desarrollo.')
    loader=DataLoader(Images(test),batch_size=32,num_workers=0)
    probs=[];labels=[];started=time.perf_counter()
    with torch.inference_mode():
        for x,y in loader:
            probs.extend(model(x).softmax(1).tolist());labels.extend(y.tolist())
    elapsed=time.perf_counter()-started
    predicted=np.asarray(probs).argmax(1).tolist()
    metrics=classification_metrics(labels,predicted)
    majority=max(range(len(CLASSES)),key=lambda i:sum(r['label']==CLASSES[i] for r in rows if r['split']=='train'))
    metrics.update(n_test=len(test),checkpoint_sha256=fingerprint,manifest_sha256=manifest_hash,
                   selected_epoch=checkpoint['epoch'],
                   baseline_class=CLASSES[majority],baseline_accuracy=sum(y==majority for y in labels)/len(labels),
                   evaluation_cpu_batch_seconds=elapsed,scope='Exploratorio: agrupación visual automática; independencia de objetos pendiente de revisión humana.',
                   no_external_test=True)
    records=[]
    for r,y,p,prob in zip(test,labels,predicted,probs):
        records.append(dict(path=r['path'],group=r['group'],true=CLASSES[y],predicted=CLASSES[p],
                            score=max(prob),correct=y==p,probabilities=prob))
    save_json(ROOT/'results/predicciones_prueba.json',records)
    save_json(previous,metrics)
    plt.rcParams.update({'font.size':10,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    cm=np.asarray(metrics['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(11,8));im=ax.imshow(cm,cmap='YlGnBu')
    ax.set(xticks=range(len(CLASSES)),yticks=range(len(CLASSES)),xticklabels=CLASSES,yticklabels=CLASSES,
           xlabel='Predicción',ylabel='Clase real',title=f'Matriz de confusión · prueba (n={len(test)})')
    ax.tick_params(axis='x', labelrotation=40)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else '#123')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(ROOT/'results/matriz_confusion.png',dpi=160);plt.close(fig)
    history=json.loads((ROOT/'results/historial.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric,title in zip(axes,['loss','accuracy'],['Pérdida','Exactitud']):
        ax.plot([r['epoch'] for r in history],[r['train_'+metric] for r in history],label='Entrenamiento')
        ax.plot([r['epoch'] for r in history],[r['val_'+metric] for r in history],label='Validación')
        ax.axvline(checkpoint['epoch'],ls=':',color='#586');ax.set(xlabel='Época',title=title);ax.legend()
    axes[0].set_ylabel('Train ponderada / validación sin ponderar')
    axes[1].set_ylim(0,1);fig.tight_layout();fig.savefig(ROOT/'results/curvas.png',dpi=160);plt.close(fig)
    errors=sorted([r for r in records if not r['correct']],key=lambda r:-r['score'])[:12]
    if errors:
        canvas=Image.new('RGB',(800,((len(errors)+3)//4)*180),'#f5f4ef');draw=ImageDraw.Draw(canvas)
        for k,r in enumerate(errors):
            x,y=(k%4)*200,(k//4)*180
            with Image.open(ROOT/r['path']) as image:
                canvas.paste(ImageOps.pad(image,(190,132)),(x,y))
            draw.text((x+3,y+134),'Real: '+r['true'],fill='black')
            draw.text((x+3,y+149),'Pred.: '+r['predicted'],fill='black')
            draw.text((x+3,y+164),f"Puntuación: {r['score']:.1%}",fill='black')
        canvas.save(ROOT/'results/errores.jpg')
    print(json.dumps(metrics,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
