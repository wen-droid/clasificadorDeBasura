"""Genera el cuaderno de reproducción del modelo de escenas en Colab."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []

    def add(kind, text):
        cell = dict(cell_type=kind, id=f'residuos-{len(cells):02d}', metadata={},
                    source=text.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''# Clasificador de residuos en Google Colab
El cuaderno reproduce la CNN de diez categorías y su ajuste con TACO. Coloca el repositorio actualizado en Drive y agrega las imágenes internas indicadas en el README. Si la carpeta es compartida, añade un acceso directo en Mi unidad y verifica que puedas guardar en ella.
''')
    add('code', '''from google.colab import drive
drive.mount('/content/drive')
''')
    add('markdown', '''## Preparar la ejecución
Se conserva el manifiesto versionado y su lista de exclusiones. La ejecución copia el checkpoint inicial como referencia para la comparación. El entrenamiento de ObjectCNN parte de una inicialización aleatoria y la etapa TACO parte del nuevo checkpoint de objetos. Las fotos de TACO se copian desde Drive si están disponibles o se descargan según el manifiesto.
''')
    add('code', '''import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path('/content/drive/MyDrive/clasificadorDeBasura')  # Ajusta la ruta a tu copia.
spec = importlib.util.spec_from_file_location('drive_workflow', SOURCE/'src/drive_workflow.py')
if not (SOURCE/'src/drive_workflow.py').is_file():
    raise FileNotFoundError('Ajusta SOURCE a la carpeta que contiene src/drive_workflow.py.')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)
PROJECT = workflow.prepare_run(SOURCE, Path('/content/residuos_ejecuciones'))

def run(script, *args):
    subprocess.run([sys.executable, script, *args], cwd=PROJECT, check=True)

def guardar_avance():
    print('Copia en Drive', workflow.save_run(PROJECT, SOURCE))

run('-m', 'pip', 'install', '-r', 'requirements.txt')
run('src/preparar_escenas.py', '--desde-manifiesto')
''')
    add('markdown', '''## Entrenar y evaluar
Primero se entrena ObjectCNN con el conjunto interno. Después se ajusta con TACO y se recalculan los umbrales. Los dos scripts seleccionan la época con validación y registran las métricas de prueba al final. El código actual entrena en CPU.
''')
    add('code', '''run('src/objetos.py')
guardar_avance()
''')
    add('code', '''run('src/escenas.py')
guardar_avance()
''')
    add('markdown', '''## Exportar y comprobar el prototipo
El HTML que produce esta celda trae los pesos de la ejecución actual. La comprobación de equivalencia requiere Node.js. La carpeta de Drive contiene el HTML, los pesos, las métricas y los archivos de código de esta ejecución.
''')
    add('code', '''run('src/export_web.py')
if shutil.which('node'):
    run('tests/verificar_html.py', '--guardar')
else:
    print('Node.js no está disponible. Ejecuta la comprobación local indicada en el README.')
guardar_avance()
report = json.loads((PROJECT/'results/escenas.json').read_text(encoding='utf-8'))
print('Exactitud interna', report['test']['accuracy'])
print('F1 macro interno', report['test']['macro_f1'])
print('Prueba externa', report['external_test_after'])
''')
    nb = dict(cells=cells, metadata=dict(colab=dict(provenance=[]),
              kernelspec=dict(display_name='Python 3', language='python', name='python3'),
              language_info=dict(name='python', version='3.11')), nbformat=4, nbformat_minor=5)
    (ROOT/'Clasificador_residuos_Colab.ipynb').write_text(
        json.dumps(nb, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
