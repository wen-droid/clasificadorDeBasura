"""Preparación de ejecuciones aisladas y guardado en Drive, sin paquete ZIP."""
import hashlib
import json
import shutil
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = ['src/common.py', 'src/prepare.py', 'src/train.py', 'src/evaluate.py',
            'src/export_web.py', 'src/browser_engine.js', 'src/web_template.html',
            'src/crear_word.py', 'requirements.txt']


def _label(name):
    return ''.join(c for c in unicodedata.normalize('NFKD', name.lower())
                   if not unicodedata.combining(c)).split('-samples')[0]


def validate_source(source):
    source = Path(source).expanduser().resolve()
    missing = [name for name in REQUIRED if not (source/name).is_file()]
    if missing:
        raise FileNotFoundError('La ruta debe apuntar a primerAvanceOficial. Faltan: '+', '.join(missing))
    originals = source/'data/archive/original'
    expected = {'battery','biological','cardboard','clothes','glass','metal','paper','plastic','shoes','trash'}
    if not originals.is_dir() or {p.name for p in originals.iterdir() if p.is_dir()} != expected:
        raise FileNotFoundError('Se requieren las diez categorías en data/archive/original.')
    archives = sorted(p for p in originals.rglob('*') if p.is_file())
    if not archives: raise FileNotFoundError('No hay imágenes originales.')
    for file in [*(source/name for name in REQUIRED), *archives]:
        if not file.resolve().is_relative_to(source):
            raise ValueError('Un archivo del proyecto apunta fuera de primerAvanceOficial: '+str(file))
    return source, archives


def prepare_run(source, workspace):
    source, archives = validate_source(source)
    workspace = Path(workspace).resolve()
    if workspace == source or workspace.is_relative_to(source):
        raise ValueError('La ejecución temporal debe estar fuera de la carpeta compartida.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    local = workspace/(stamp+'_'+uuid.uuid4().hex[:8])
    local.mkdir(parents=True, exist_ok=False)
    inputs = []
    for folder in ['src', 'tests']:
        for file in sorted((source/folder).rglob('*')):
            if file.is_file() and '__pycache__' not in file.parts and file.suffix in {'.py', '.js', '.html'}:
                inputs.append(file)
    inputs.extend(archives)
    for name in ['requirements.txt', 'Clasificador_residuos_Colab.ipynb',
                 'data/fuente_dataset.json', 'data/grupos_manuales.csv']:
        if (source/name).is_file(): inputs.append(source/name)
    hashes = {}
    for file in inputs:
        if not file.resolve().is_relative_to(source):
            raise ValueError('Archivo fuera del proyecto: '+str(file))
        relative = file.relative_to(source)
        target = local/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
        hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    trace = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=str(source),
                 run_name=local.name, inputs_sha256=hashes, initialization='Desde cero',
                 copied_previous_models=False, copied_previous_results=False)
    (local/'ejecucion_drive.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding='utf-8')
    return local


def save_run(local, source):
    local = Path(local).resolve()
    source, _ = validate_source(source)
    if local == source or local.is_relative_to(source):
        raise ValueError('Solo se guardan ejecuciones temporales aisladas.')
    trace = json.loads((local/'ejecucion_drive.json').read_text(encoding='utf-8'))
    parent = source/'Ejecuciones_equipo'
    parent.mkdir(exist_ok=True)
    # Cada guardado tiene un identificador nuevo y no reemplaza resultados oficiales.
    destination = parent/(trace['run_name']+'_guardado_'+uuid.uuid4().hex[:8])
    destination.mkdir(exist_ok=False)
    for folder in ['src', 'tests', 'models', 'results', 'informe']:
        if (local/folder).is_dir():
            shutil.copytree(local/folder, destination/folder,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['ejecucion_drive.json', 'requirements.txt', 'Clasificador_residuos_Colab.ipynb',
                 'CLASIFICADOR_RESIDUOS.html', 'data/manifest.csv', 'data/grupos_manuales.csv',
                 'data/fuente_dataset.json']:
        file = local/name
        if file.is_file():
            target = destination/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
    (destination/'GUARDADO_COMPLETO.txt').write_text(
        'Copia completa de los artefactos disponibles al guardar. Consultar results/configuracion.json y metricas.json para el estado del experimento.\n', encoding='utf-8')
    return destination
