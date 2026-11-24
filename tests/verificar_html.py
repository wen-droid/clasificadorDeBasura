"""Verifica que el HTML autónomo ejecuta el mismo modelo que PyTorch."""
import hashlib
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image, ImageOps
import torch
from common import ROOT, CLASSES, load_model, preprocess, save_json
from materiales import MaterialCNN, CHECKPOINT, GROUP
from objetos import ObjectCNN, CHECKPOINT as OBJECT_CHECKPOINT
from escenas import CHECKPOINT as SCENE_CHECKPOINT


def main():
    html=(ROOT/'CLASIFICADOR_RESIDUOS.html').read_text(encoding='utf-8')
    model_data=json.loads(re.search(r'<script id="cnn-model" type="application/json">(.*?)</script>',html,re.S).group(1))
    assert '<script src=' not in html and 'fetch(' not in html
    assert 'href="/informe"' not in html
    materials = 'threshold' in model_data
    objects = 'thresholds' in model_data
    if objects:
        path = SCENE_CHECKPOINT if SCENE_CHECKPOINT.exists() else OBJECT_CHECKPOINT
        checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        model = ObjectCNN().eval()
        model.load_state_dict(checkpoint['state_dict'])
        assert model_data['thresholds'] == checkpoint['thresholds']
    elif materials:
        checkpoint = torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
        model = MaterialCNN().eval()
        model.load_state_dict(checkpoint['state_dict'])
        path = CHECKPOINT
    else:
        model,checkpoint = load_model()
        path = ROOT/'models/mejor_modelo.pt'
    assert model_data['classes'] == checkpoint['classes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == model_data['checkpoint_sha256']
    torch.set_num_threads(2)
    rows=[r for r in csv.DictReader((ROOT/'data/manifest.csv').open(encoding='utf-8')) if r['split']=='test']
    for r in rows:
        original = Path(r['original'])
        candidates = (ROOT/r['path'], ROOT/original, ROOT/'data/archive/data'/original.parent.name/original.name)
        r['available'] = next((p for p in candidates if p.is_file()),None)
    rows=[r for r in rows if r['available']]
    assert set(r['label'] for r in rows)==set(CLASSES)
    rows=[r for c in CLASSES for r in [x for x in rows if x['label']==c][:2]]
    def infer(tensor):
        x = tensor.unsqueeze(0)
        probs = model.probabilities(x) if objects else model(x).softmax(1)
        return probs[0].tolist()

    inputs=[];expected=[];true=[]
    for row in rows:
        with Image.open(row['available']) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');rgba=np.asarray(im.convert('RGBA'))
            tensor=preprocess(im)
        with torch.inference_mode():
            scores=infer(tensor)
        inputs.append(dict(width=im.width,height=im.height,rgba=rgba.flatten().tolist()))
        expected.append(dict(tensor=tensor.numpy().flatten(),scores=np.array(scores)))
        label = CLASSES.index(row['label'])
        true.append(GROUP[label] if materials else label)
    # Proporciones no cuadradas y redondeo al par en padding.
    for w,h in [(97,43),(43,97),(96,96),(301,200),(5,7)]:
        rng=np.random.default_rng(w+h)
        im=Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8))
        tensor=preprocess(im)
        with torch.inference_mode():
            scores=infer(tensor)
        inputs.append(dict(width=w,height=h,rgba=np.asarray(im.convert('RGBA')).flatten().tolist()))
        expected.append(dict(tensor=tensor.numpy().flatten(),scores=np.array(scores)))
    with tempfile.TemporaryDirectory(prefix='residuos_html_') as td:
        tmp=Path(td)
        engine = re.search(r'<script id="cnn-engine"[^>]*>(.*?)</script>', html, re.S).group(1)
        (tmp/'engine.cjs').write_text(engine, encoding='utf-8')
        (tmp/'model.json').write_text(json.dumps(model_data),encoding='utf-8')
        (tmp/'inputs.json').write_text(json.dumps(inputs),encoding='utf-8')
        runner='''const fs=require('fs');const {decodeWeights,predictCNN,preprocessRGBA}=require(process.argv[2]);
const model=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const samples=JSON.parse(fs.readFileSync(process.argv[4],'utf8'));
const weights=Object.fromEntries(Object.entries(model.weights).map(([k,v])=>[k,decodeWeights(v)]));
const out=samples.map(s=>{const x=preprocessRGBA(new Uint8ClampedArray(s.rgba),s.width,s.height);return {tensor:Array.from(x),scores:predictCNN(x,weights)}});
fs.writeFileSync(process.argv[5],JSON.stringify(out));'''
        (tmp/'run.cjs').write_text(runner,encoding='utf-8')
        subprocess.run(['node',str(tmp/'run.cjs'),str(tmp/'engine.cjs'),str(tmp/'model.json'),str(tmp/'inputs.json'),str(tmp/'out.json')],check=True)
        outputs=json.loads((tmp/'out.json').read_text())
    max_pixels=max(float(np.max(np.abs(e['tensor']-np.array(o['tensor'])))) for e,o in zip(expected,outputs))
    max_scores=max(float(np.max(np.abs(e['scores']-np.array(o['scores'])))) for e,o in zip(expected,outputs))
    agree=sum(int(np.argmax(e['scores'])==np.argmax(o['scores'])) for e,o in zip(expected,outputs))
    assert max_pixels<1e-6, f'Preprocesamiento distinto: {max_pixels}'
    assert max_scores<2e-5, f'Inferencia distinta: {max_scores}'
    assert agree==len(inputs)
    html_accuracy=sum(int(np.argmax(o['scores'])==y) for o,y in zip(outputs,true))/len(true)
    if materials or objects:
        def decision(scores):
            winner = int(np.argmax(scores))
            if objects:
                return winner if max(scores) >= model_data['thresholds'][winner] else -1
            return winner if winner == 5 or max(scores) >= model_data['threshold'] else -1
        assert all(decision(e['scores']) == decision(o['scores']) for e,o in zip(expected,outputs))
    # La paridad se comprueba en una muestra estratificada; las métricas completas se calculan con PyTorch.
    check=dict(samples=len(inputs),test_images=len(rows),same_prediction=agree,max_preprocess_difference=max_pixels,
               max_probability_difference=max_scores,sample_accuracy=html_accuracy,no_external_scripts=True,
               no_server_required=True,model_sha256=model_data['checkpoint_sha256'])
    save_json(ROOT/'results'/('verificacion_escenas.json' if path == SCENE_CHECKPOINT else 'verificacion_objetos.json' if objects else 'verificacion_materiales.json' if materials else 'verificacion_html.json'),check)
    print(json.dumps(check,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
