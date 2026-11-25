# Clasificador de residuos

Primer avance de Machine Learning de la Universidad Mayor de San Simón. El prototipo clasifica imágenes en diez categorías, entre ellas zapatos, ropa, vidrio y plástico.

## Probar el prototipo

Descarga [CLASIFICADOR_RESIDUOS.html](CLASIFICADOR_RESIDUOS.html), ábrelo en Chrome, Edge o Firefox y elige **Subir imagen**. La clasificación se ejecuta sin conexión. Puedes usar el recorte manual para encuadrar un objeto. Cuando la puntuación ganadora queda por debajo del umbral de su clase, la interfaz muestra «Posible» seguido de la categoría.

## Regenerar y verificar el HTML

Desde la raíz del repositorio, con Python 3.11 o superior y Node.js 18 o superior:

```text
python -m pip install -r requirements.txt
python src/export_web.py
python tests/verificar_html.py --solo-sinteticas
```

El exportador usa [models/escenas.pt](models/escenas.pt) y se detiene si falta ese archivo. La verificación compara las puntuaciones en PyTorch y JavaScript y comprueba el SHA-256 de los pesos cargados por el HTML. Para verificar también con 20 fotos del conjunto interno, prepara los originales y ejecuta `python tests/verificar_html.py`. Si las fotos están en otra copia del proyecto, pasa `--datos RUTA_DE_ESE_PROYECTO`. Añade `--guardar` para escribir el resultado con fotos en `results/verificacion_escenas.json`.

## Repetir el entrenamiento

El ZIP de entrega académica incluye las **12.259 imágenes originales** en `data/archive/original/` y las **204 fotos seleccionadas de TACO** en `data/taco/images/`. Extrae todo el ZIP antes de abrir la carpeta en VS Code. La [lista de exclusiones](data/exclusiones_primer_avance.txt) conserva las 19 ausencias históricas, por lo que el flujo de objetos y escenas usa 12.240 imágenes internas.

Si obtuviste el código desde GitHub, ejecuta los siguientes comandos para reunir los datos. Necesitan curl y acceso a Kaggle y al servidor de TACO. El primero valida los píxeles con [data/manifest.csv](data/manifest.csv) y acepta también un ZIP descargado manualmente con `--origen RUTA.zip`. El segundo usa las URL de [data/taco/manifest.json](data/taco/manifest.json) y conserva las particiones.

```text
python -m pip install -r requirements.txt
python src/preparar_originales.py
python src/preparar_escenas.py --desde-manifiesto
```

Con el ZIP académico ya extraído puedes ejecutar directamente la preparación local y el entrenamiento. La instalación de dependencias puede requerir Internet. Los scripts de entrenamiento vuelven a generar los pesos de objetos y escenas, sus métricas y el HTML. Trabaja en una copia si quieres conservar los artefactos entregados. El checkpoint `models/mejor_modelo.pt` se usa para la comparación.

```text
python -m pip install -r requirements.txt
python src/preparar_originales.py --verificar
python src/objetos.py
python src/escenas.py
python src/export_web.py
python tests/verificar_html.py
```

En Google Colab, abre [el cuaderno](Clasificador_residuos_Colab.ipynb), cambia la ruta `SOURCE` a tu copia en Drive y ejecuta las celdas en orden. Los resultados se guardan en `Ejecuciones_equipo/`. Los scripts de objetos y escenas entrenan en CPU. Los manifiestos fijan las particiones y el subconjunto. Las versiones de las bibliotecas y el entorno pueden afectar los pesos resultantes.

## Resultados y archivos del equipo

- El informe final acompaña al ZIP como documento separado.
- [Métricas de la CNN de objetos](results/objetos.json) y [métricas del modelo final](results/escenas.json). En las mismas 1.840 fotos del conjunto interno, el modelo inicial y el final obtuvieron 65,6 % y 75,8 % de exactitud, respectivamente. El final acertó 2 de 13 fotos completas en la prueba externa de TACO.
- [Pruebas funcionales en Microsoft Edge](results/prueba_capturas.json) y [verificación de paridad del modelo final](results/verificacion_escenas.json).
- Los scripts `src/prepare.py`, `src/train.py`, `src/evaluate.py` y `src/crear_word.py` documentan la CNN inicial. El ensayo de cinco materiales está en `src/materiales.py`. Para exportar una variante histórica, usa `--modelo objetos`, `--modelo materiales` o `--modelo inicial` junto con `--salida ARCHIVO.html`.

## Fuentes de las imágenes

- [Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2), publicado por sumn2u. Kaggle identifica la colección con licencia MIT. La ruta y la variante empleadas se registran en [data/fuente_dataset.json](data/fuente_dataset.json).
- [TACO](https://github.com/pedropro/TACO), de Pedro F. Proença y Pedro Simões. El [manifiesto](data/taco/manifest.json) registra las URL, identificadores y los datos de licencia disponibles para cada foto. El paquete conserva la versión redimensionada usada para entrenar.
