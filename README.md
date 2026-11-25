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

1. Obtén las diez carpetas de la variante original de **Garbage Classification V2**, indicada en [data/fuente_dataset.json](data/fuente_dataset.json), y colócalas en `data/archive/original/` o `data/archive/data/`. El cargador valida los hashes de [data/manifest.csv](data/manifest.csv). La [lista de exclusiones](data/exclusiones_primer_avance.txt) fija las 19 fotos que faltaron en la ejecución histórica.
2. Copia las fotos de **TACO** a `data/taco/images/` o recupéralas con el comando de abajo, que requiere curl y acceso a su servidor. Se usa [el manifiesto versionado](data/taco/manifest.json). Si alguna URL falla, reintenta o recupera la foto del respaldo del equipo. El código requiere todas las fotos registradas para repetir el ajuste.
3. Trabaja en una copia del proyecto si quieres conservar los artefactos entregados. Estos comandos vuelven a generar los pesos de objetos y escenas, sus métricas y el HTML. El checkpoint original `models/mejor_modelo.pt` se usa para la comparación.

```text
python src/preparar_escenas.py --desde-manifiesto
python src/objetos.py
python src/escenas.py
python src/export_web.py
python tests/verificar_html.py
```

En Google Colab, abre [el cuaderno](Clasificador_residuos_Colab.ipynb), cambia la ruta `SOURCE` a tu copia en Drive y ejecuta las celdas en orden. Los resultados se guardan en `Ejecuciones_equipo/`. Los scripts de objetos y escenas entrenan en CPU. Los manifiestos fijan las particiones y el subconjunto. Las versiones de las bibliotecas y el entorno pueden afectar los pesos resultantes.

## Resultados y archivos del equipo

- [Informe actualizado con los scripts en cada etapa](informe/Informe_primer_avance_APA.docx).
- [Métricas de la CNN de objetos](results/objetos.json) y [métricas del modelo final](results/escenas.json). En las mismas 1.840 fotos del conjunto interno, el modelo inicial y el final obtuvieron 65,6 % y 75,8 % de exactitud, respectivamente. El final acertó 2 de 13 fotos completas en la prueba externa de TACO.
- [Pruebas funcionales en Microsoft Edge](results/prueba_capturas.json) y [verificación de paridad del modelo final](results/verificacion_escenas.json).
- [Paquete descargable del equipo](Clasificador_residuos_equipo.zip). Las colecciones de imágenes se obtienen por separado.
- Los scripts `src/prepare.py`, `src/train.py`, `src/evaluate.py` y `src/crear_word.py` documentan la CNN inicial. El ensayo de cinco materiales está en `src/materiales.py`. Para exportar una variante histórica, usa `--modelo objetos`, `--modelo materiales` o `--modelo inicial` junto con `--salida ARCHIVO.html`.
