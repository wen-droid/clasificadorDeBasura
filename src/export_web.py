"""Exporta el modelo de escenas o una variante histórica indicada explícitamente."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import torch
from common import ROOT, load_model
from materiales import CHECKPOINT, MaterialCNN
from objetos import CHECKPOINT as OBJECT_CHECKPOINT, ObjectCNN
from escenas import CHECKPOINT as SCENE_CHECKPOINT

MODELS = {
    'escenas': SCENE_CHECKPOINT,
    'objetos': OBJECT_CHECKPOINT,
    'materiales': CHECKPOINT,
    'inicial': ROOT / 'models/mejor_modelo.pt',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modelo', choices=MODELS, default='escenas')
    parser.add_argument('--salida', type=Path, default=ROOT / 'CLASIFICADOR_RESIDUOS.html')
    args = parser.parse_args()
    path = MODELS[args.modelo]
    if not path.is_file():
        parser.error(f'Falta {path.relative_to(ROOT)}. Obtén los pesos o entrena el modelo seleccionado.')
    if args.modelo != 'inicial':
        objects = path != CHECKPOINT
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        model = (ObjectCNN() if objects else MaterialCNN()).eval()
        model.load_state_dict(checkpoint['state_dict'])
        tensors = {}
        for index, block in enumerate(model.blocks):
            prefix = ('stage' if objects else 'block') + str(index)
            for start in range(0, len(block), 3):
                conv = torch.nn.utils.fusion.fuse_conv_bn_eval(block[start], block[start+1])
                key = prefix if start == 0 else prefix + '.extra'
                tensors[key + '.weight'] = conv.weight
                tensors[key + '.bias'] = conv.bias
        tensors['head.weight'] = model.head[-1].weight
        tensors['head.bias'] = model.head[-1].bias
        field = 'thresholds' if objects else 'threshold'
        metadata = {field: checkpoint[field]}
        template = 'materiales.html'
    else:
        model, checkpoint = load_model()
        tensors = model.state_dict()
        metadata = {}
        template = 'web_template.html'
    weights = {name: base64.b64encode(t.detach().cpu().numpy().astype('<f4').tobytes()).decode('ascii')
               for name, t in tensors.items()}
    payload = dict(classes=checkpoint['classes'], weights=weights, **metadata,
                   checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    engine = (ROOT / 'src/browser_engine.js').read_text(encoding='utf-8')
    html = (ROOT / 'src' / template).read_text(encoding='utf-8')
    if template == 'materiales.html':
        classes = checkpoint['classes'] if objects else checkpoint['classes'][:5]
        names = ''.join('<div class="material"><span class="icon"></span><div><b>' + name +
                        '</b></div></div>' for name in classes)
        html = html.replace('/*__CARDS__*/', names).replace('/*__COUNT__*/', str(len(classes)))
    output = html.replace('/*__ENGINE__*/', engine).replace('/*__MODEL__*/', json.dumps(payload,ensure_ascii=False))
    target = args.salida
    temporary = target.with_suffix('.html.tmp')
    temporary.write_text(output, encoding='utf-8')
    os.replace(temporary, target)
    print('HTML generado a partir de', path.name)


if __name__ == '__main__':
    main()
