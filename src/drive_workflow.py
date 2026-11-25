"""Preparación de ejecuciones aisladas y guardado en Drive, sin paquete ZIP."""
import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

INPUTS = ['requirements.txt', 'data/manifest.csv', 'data/exclusiones_primer_avance.txt',
          'data/fuente_dataset.json', 'data/taco/manifest.json', 'models/mejor_modelo.pt']
REQUIRED = INPUTS + ['src/common.py', 'src/materiales.py', 'src/objetos.py', 'src/escenas.py',
                     'src/preparar_escenas.py', 'src/export_web.py', 'src/browser_engine.js',
                     'src/materiales.html', 'tests/verificar_html.py']


def validate_source(source):
    source = Path(source).expanduser().resolve()
    missing = [name for name in REQUIRED if not (source/name).is_file()]
    if missing:
        raise FileNotFoundError('La ruta debe apuntar al repositorio actualizado. Faltan: '+', '.join(missing))
    expected = {'battery','biological','cardboard','clothes','glass','metal','paper','plastic','shoes','trash'}
    roots = [source/'data/archive'/name for name in ('original', 'data')]
    categories = [root/label for root in roots for label in sorted(expected) if (root/label).is_dir()]
    if {path.name for path in categories} != expected:
        raise FileNotFoundError('Se requieren las diez categorías en data/archive/original o data/archive/data.')
    archives = sorted(p for folder in categories for p in folder.iterdir() if p.is_file())
    if not archives: raise FileNotFoundError('No hay imágenes originales.')
    for file in [*(source/name for name in REQUIRED), *archives]:
        if not file.resolve().is_relative_to(source):
            raise ValueError('Un archivo apunta fuera del repositorio: '+str(file))
    return source, archives


def prepare_run(source, workspace):
    source, archives = validate_source(source)
    workspace = Path(workspace).resolve()
    if workspace == source or workspace.is_relative_to(source):
        raise ValueError('La ejecución temporal debe estar fuera de la carpeta compartida.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    local = workspace/(stamp+'_'+uuid.uuid4().hex[:8])
    local.mkdir(parents=True, exist_ok=False)
    inputs = [source/name for name in INPUTS]
    for folder in ['src', 'tests']:
        for file in sorted((source/folder).rglob('*')):
            if file.is_file() and '__pycache__' not in file.parts and file.suffix in {'.py', '.js', '.html'}:
                inputs.append(file)
    inputs.extend(archives)
    inputs.extend(sorted((source/'data/taco/images').glob('*.jpg')))
    notebook = source/'Clasificador_residuos_Colab.ipynb'
    if notebook.is_file():
        inputs.append(notebook)
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
                 run_name=local.name, inputs_sha256=hashes,
                 initialization='ObjectCNN desde cero, ajuste TACO desde objetos.pt',
                 reference_checkpoint='models/mejor_modelo.pt', copied_previous_results=False)
    (local/'ejecucion_drive.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding='utf-8')
    return local


def save_run(local, source):
    local = Path(local).resolve()
    source = Path(source).expanduser().resolve()
    if local == source or local.is_relative_to(source):
        raise ValueError('Solo se guardan ejecuciones temporales aisladas.')
    trace = json.loads((local/'ejecucion_drive.json').read_text(encoding='utf-8'))
    if Path(trace['source']) != source:
        raise ValueError('La carpeta de origen no coincide con la registrada en la ejecución.')
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
                 'CLASIFICADOR_RESIDUOS.html', 'data/manifest.csv', 'data/exclusiones_primer_avance.txt',
                 'data/fuente_dataset.json', 'data/taco/manifest.json']:
        file = local/name
        if file.is_file():
            target = destination/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
    (destination/'GUARDADO_COMPLETO.txt').write_text(
        'Copia de los artefactos disponibles. Consulta results/objetos.json y results/escenas.json.\n',
        encoding='utf-8')
    return destination
