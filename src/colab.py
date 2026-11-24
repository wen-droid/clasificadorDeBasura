"""Genera el cuaderno de Drive sin paquetes de entrada o salida."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []
    def add(kind, text):
        cell = dict(cell_type=kind, id=f'residuos-{len(cells):02d}', metadata={}, source=text.splitlines(keepends=True))
        if kind == 'code': cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''# Clasificador de residuos desde la carpeta compartida
Primer avance de visión por computadora. CNN propia entrenada desde cero.

El cuaderno lee código y datos de `primerAvanceOficial` en Drive, sin subir un paquete de archivos. La carpeta compartida debe incluir `src/`, `tests/`, `requirements.txt` y los imágenes de las diez categorías en `data/archive/original/`. Estas imágenes corresponden al dataset de Kaggle.

Si la carpeta está en «Compartido conmigo», añade un acceso directo a ella en «Mi unidad». Cada integrante abre el cuaderno con su propia cuenta. Necesita acceso de lectura al proyecto y permiso de edición para guardar resultados en él.

Referencias: [Colab](https://research.google.com/colaboratory/faq.html) y [accesos directos de Drive](https://support.google.com/drive/answer/9700156?hl=es).
''')
    add('code', '''# 1 Conectar Drive con tu cuenta. Acepta la autorización cuando Colab la solicite.
from google.colab import drive
drive.mount('/content/drive')
''')
    add('code', '''# 2 CAMBIA esta ruta si primerAvanceOficial está dentro de otras carpetas.
# Usa la ruta del panel de archivos de Colab, no un enlace web de Drive.
from pathlib import Path
RUTA_PROYECTO = '/content/drive/MyDrive/primerAvanceOficial'
SOURCE = Path(RUTA_PROYECTO).expanduser().resolve()
if not (SOURCE/'src/drive_workflow.py').is_file():
    raise FileNotFoundError('Revisa la ruta y que src/drive_workflow.py esté en la carpeta compartida actualizada.')
print('Proyecto compartido:', SOURCE)
''')
    add('markdown', '''## Preparar el trabajo de cada integrante
El cuaderno copia código y datos de entrada a una ejecución temporal nueva en Colab. **No copia modelos entrenados ni métricas anteriores.** Así se entrena desde cero y no se modifica la versión oficial.

Cada ejecución y cada guardado tienen un identificador distinto. Los resultados se guardan en `primerAvanceOficial/Ejecuciones_equipo/`. No uses una carpeta de resultados como ruta de entrada del proyecto.
''')
    add('code', '''# 3 Comprobar archivos y preparar una ejecución aislada.
import importlib.util, os, subprocess, sys
spec = importlib.util.spec_from_file_location('drive_workflow', SOURCE/'src/drive_workflow.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
PROJECT = workflow.prepare_run(SOURCE, Path('/content/residuos_ejecuciones'))
os.chdir(PROJECT)
def run(script, *args):
    subprocess.run([sys.executable, str(PROJECT/script), *args], cwd=PROJECT, check=True)
def guardar_avance():
    destino = workflow.save_run(PROJECT, SOURCE)
    print('Guardado en Drive:', destino)
    return destino
print('Ejecución temporal:', PROJECT)
print('Se copiaron código e imágenes; no se copiaron pesos anteriores.')
''')
    add('code', '''# 4 Dependencias. GPU es opcional; puedes habilitarla en Entorno de ejecución.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(PROJECT/'requirements.txt')], check=True)
import torch
print('PyTorch:', torch.__version__, '| GPU disponible:', torch.cuda.is_available())
''')
    add('markdown', '''## A Datos y preprocesamiento
La auditoría comprueba archivos ilegibles, elimina duplicados exactos y agrupa imágenes visualmente próximas. Separa entrenamiento, validación y prueba **antes del aumento de datos**.

Revisa etiquetas y hojas de contacto. Si reconoces el mismo objeto en distintas fotos, detén el entrenamiento y revisa su agrupamiento antes de continuar. dHash no garantiza identificar todos los objetos físicos. Cualquier cambio de partición debe documentarse como otra versión del conjunto.

La procedencia es Kaggle; completa la publicación específica en la bibliografía del equipo. Orgánico incluye alimentos enteros y puede existir sesgo de fondo.
''')
    add('code', '''# 5 Auditar y revisar las imágenes y sus particiones.
run('src/prepare.py')
import json
audit = json.loads((PROJECT/'results/auditoria.json').read_text(encoding='utf-8'))
print('Originales:', audit['input_images'], '| Utilizables:', audit['usable_images'])
print('Duplicados exactos retirados:', audit['exact_duplicates'])
for label, counts in audit['counts'].items(): print(label, counts)
from IPython.display import display, Image
for path in sorted((PROJECT/'results').glob('contacto_*.jpg')):
    print(path.name)
    display(Image(filename=str(path)))
guardar_avance()
''')
    add('code', '''# 6 Confirma la revisión antes de entrenar.
DATOS_REVISADOS = False  # Cambia a True después de revisar imágenes, etiquetas y particiones.
if not DATOS_REVISADOS:
    raise RuntimeError('Revisa los datos y cambia DATOS_REVISADOS a True para continuar.')
print('Revisión confirmada por el integrante.')
''')
    add('markdown', '''## B Entrenamiento desde cero
La CNN inicializa sus parámetros aleatoriamente. El aumento y dropout se aplican en entrenamiento; validación selecciona la época por menor pérdida. Prueba no participa en el ajuste.

Una técnica profunda cumple el mínimo del primer avance. Comparar variantes es opcional y debe realizarse usando validación. No se promete una exactitud ni una nota determinadas.
''')
    add('code', '''# 7 Entrenar y conservar una copia de los artefactos en Drive.
if not DATOS_REVISADOS: raise RuntimeError('Revisa los datos antes de entrenar.')
try:
    run('src/train.py', '--epochs', '50', '--patience', '10', '--seed', '42')
finally:
    guardar_avance()  # También guarda archivos parciales si el proceso termina con error.
''')
    add('code', '''# 8 Curvas de desarrollo, sin consultar prueba.
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT/'.cache/matplotlib'))
import matplotlib.pyplot as plt
history = json.loads((PROJECT/'results/historial.json').read_text(encoding='utf-8'))
plt.plot([r['epoch'] for r in history], [r['train_accuracy'] for r in history], label='Entrenamiento')
plt.plot([r['epoch'] for r in history], [r['val_accuracy'] for r in history], label='Validación')
plt.xlabel('Época'); plt.ylabel('Exactitud'); plt.legend(); plt.show()
''')
    add('markdown', '''## C Evaluación final y prototipo
Consulta prueba una vez terminada la selección por validación. No ajustes de nuevo el modelo mirando esa prueba. Se calculan métricas y matriz de confusión; se exportan un HTML autónomo y un Word separado. La evaluación interna no sustituye las fotografías externas nuevas.
''')
    add('code', '''# 9 Evaluar y generar la aplicación y el informe de esta ejecución.
import shutil
try:
    run('src/evaluate.py')
    run('src/export_web.py')
    if shutil.which('node'):
        run('tests/verificar_html.py')
    else:
        print('Comprobación JavaScript-PyTorch no ejecutada: Node no está instalado en esta sesión.')
    run('src/crear_word.py')
finally:
    RESULTADO_DRIVE = guardar_avance()
metrics = json.loads((PROJECT/'results/metricas.json').read_text(encoding='utf-8'))
print(f"Exactitud: {metrics['accuracy']:.1%} | Macro F1: {metrics['macro_f1']:.3f} | n={metrics['n_test']}")
display(Image(filename=str(PROJECT/'results/curvas.png')))
display(Image(filename=str(PROJECT/'results/matriz_confusion.png')))
print('HTML:', RESULTADO_DRIVE/'CLASIFICADOR_RESIDUOS.html')
print('Word:', RESULTADO_DRIVE/'informe/Informe_clasificacion_residuos.docx')
''')
    add('markdown', '''## Qué encuentra el equipo en Drive
Las subcarpetas de `Ejecuciones_equipo` guardan el código, manifiesto y artefactos disponibles en ese momento: configuración, historial, modelo y, cuando estén generados, métricas, gráficas, HTML y Word. `GUARDADO_COMPLETO.txt` indica que terminó la copia; no certifica que el modelo ya esté validado. Revisa sus resultados.

Las imágenes originales siguen en `data/archive/original` de la carpeta compartida principal. Las imágenes extraídas quedan en la sesión temporal y pueden reconstruirse a partir de esos originales. Los hashes de entradas quedan en `ejecucion_drive.json`.

La sesión de Colab es temporal: ejecuta `guardar_avance()` después de cambios importantes y antes de cerrarla. Al acabar, descarga el HTML de tu ejecución para demostrar la aplicación. El código, Word y evidencias también forman parte del trabajo para defenderlo. Completa grupo e integrantes, bibliografía y pruebas externas. La cámara física se prueba por separado.

El responsable del equipo decide qué ejecución será oficial. El cuaderno no la sobrescribe automáticamente. El segundo modelo no supervisado corresponde al siguiente avance.
''')
    nb = dict(cells=cells, metadata=dict(colab=dict(provenance=[]),
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.11')), nbformat=4, nbformat_minor=5)
    (ROOT/'Clasificador_residuos_Colab.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Cuaderno actualizado: lee Drive y guarda resultados sin paquetes ZIP.')


if __name__ == '__main__': main()
