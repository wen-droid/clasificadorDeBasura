"""Entrena y evalúa las diez categorías originales, incluidas ropa y zapatos."""
import copy
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from common import CLASSES, ROOT, load_model, save_json
from materiales import augment, read_images

CHECKPOINT = ROOT / 'models/objetos.pt'
REPORT = ROOT / 'results/objetos.json'
EPOCHS = 40
SHOES = CLASSES.index('Zapatos')


class ObjectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [3, 24, 40, 64, 96]
        self.blocks = nn.ModuleList()
        for i, (before, after) in enumerate(zip(channels, channels[1:])):
            layers = [nn.Conv2d(before, after, 3, stride=2 if i == 0 else 1, padding=1),
                      nn.BatchNorm2d(after), nn.ReLU()]
            if i >= 2:
                layers.extend([nn.Conv2d(after, after, 3, padding=1), nn.BatchNorm2d(after), nn.ReLU()])
            self.blocks.append(nn.Sequential(*layers))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(.25), nn.Linear(96*9, len(CLASSES)))

    def forward(self, x):
        for block in self.blocks:
            x = F.max_pool2d(block(x), 2)
        return self.head(x)

    def probabilities(self, x):
        return (self(x).softmax(1) + self(x.flip([-1])).softmax(1)) / 2


@torch.inference_mode()
def predict(images, inference):
    return torch.cat([inference(batch.float()/255) for batch in images.split(64)])


def measure(probabilities, labels, thresholds=None):
    score, pred = probabilities.max(1)
    n = len(CLASSES)
    matrix = torch.bincount(n*labels + pred, minlength=n*n).reshape(n,n)
    f1 = 2*matrix.diag() / (matrix.sum(0)+matrix.sum(1)).clamp_min(1)
    accept = torch.ones_like(pred, dtype=torch.bool) if thresholds is None else score >= thresholds[pred]
    shoe = (pred == SHOES) & accept
    return dict(n=len(labels), accuracy=(pred == labels).float().mean().item(),
                macro_f1=f1.mean().item(), confusion=matrix.tolist(),
                accepted=int(accept.sum()), correct_accepted=int(((pred==labels)&accept).sum()),
                true_shoes=int((labels==SHOES).sum()), correct_shoes=int((shoe&(labels==SHOES)).sum()),
                false_shoes=int((shoe&(labels!=SHOES)).sum()))


def calibrate(probs, truth):
    confidence, predicted = probs.max(1)
    thresholds = []
    for i in range(len(CLASSES)):
        target = .85 if i == SHOES else .80
        # Si la validación aporta pocos aciertos, presentar sólo una hipótesis.
        cutoff = 1.
        for step in range(101):
            keep = (predicted==i) & (confidence>=step/100)
            if keep.sum() >= 20 and (truth[keep]==i).float().mean() >= target:
                cutoff = step/100
                break
        thresholds.append(cutoff)
    return torch.tensor(thresholds)


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    previous, _ = load_model()
    data, excluded = read_images()
    train, _, labels = data['train']
    val, _, truth = data['val']
    model = ObjectCNN().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.005)
    batch_size = 96
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.002, epochs=EPOCHS,
                steps_per_epoch=math.ceil(len(train)/batch_size), pct_start=.12)
    weights = (labels.bincount().float().mean()/labels.bincount()).sqrt()
    history = []
    best = -1
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.
        for idx in torch.randperm(len(train)).split(batch_size):
            x = augment(train[idx].float()/255).contiguous(memory_format=torch.channels_last)
            loss = F.cross_entropy(model(x), labels[idx], weight=weights, label_smoothing=.025)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            total_loss += loss.item()*len(idx)
        model.eval()
        metrics = measure(predict(val, model.probabilities), truth)
        history.append(dict(epoch=epoch, train_loss=total_loss/len(train), val_accuracy=metrics['accuracy'],
                            val_macro_f1=metrics['macro_f1']))
        print(epoch, 'F1 validación', round(metrics['macro_f1'],3), flush=True)
        if metrics['macro_f1'] > best:
            best = metrics['macro_f1']
            checkpoint = dict(state_dict=copy.deepcopy(model.state_dict()), epoch=epoch, classes=CLASSES)
    model.load_state_dict(checkpoint['state_dict'])
    thresholds = calibrate(predict(val, model.probabilities), truth)
    images, _, target = data['test']
    probs = predict(images, model.probabilities)
    baseline = predict(images, lambda x: previous(x).softmax(1))
    checkpoint['thresholds'] = thresholds.tolist()
    temp = CHECKPOINT.with_suffix('.pt.tmp')
    torch.save(checkpoint,temp)
    temp.replace(CHECKPOINT)
    report = dict(classes=CLASSES, seed=42, epochs=EPOCHS, selected_epoch=checkpoint['epoch'],
                  thresholds=thresholds.tolist(), splits={s:len(v[2]) for s,v in data.items()}, excluded=excluded,
                  history=history, original=measure(baseline,target), test=measure(probs,target),
                  selective=measure(probs,target,thresholds),
                  checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest())
    save_json(REPORT,report)
    print('Prueba:',report['selective'],flush=True)


if __name__=='__main__':
    main()
