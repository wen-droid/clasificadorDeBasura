"""Exporta el último clasificador disponible a un HTML autónomo."""
import base64
import hashlib
import json
import os
import torch
from common import ROOT, load_model
from materiales import CHECKPOINT, MaterialCNN
from objetos import CHECKPOINT as OBJECT_CHECKPOINT, ObjectCNN
from escenas import CHECKPOINT as SCENE_CHECKPOINT


def main():
    path = next((p for p in (SCENE_CHECKPOINT, OBJECT_CHECKPOINT, CHECKPOINT) if p.is_file()), None)
    if path is not None:
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
        path = ROOT / 'models/mejor_modelo.pt'
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
    target = ROOT / 'CLASIFICADOR_RESIDUOS.html'
    temporary = target.with_suffix('.html.tmp')
    temporary.write_text(output, encoding='utf-8')
    os.replace(temporary, target)
    print('HTML generado a partir de', path.name)


if __name__ == '__main__':
    main()
