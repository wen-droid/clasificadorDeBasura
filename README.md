# Clasificador de residuos

Primer avance de Machine Learning de la Universidad Mayor de San Simón. El prototipo clasifica imágenes en diez categorías, entre ellas zapatos, ropa, vidrio y plástico.

## Probar el prototipo

Descarga [CLASIFICADOR_RESIDUOS.html](CLASIFICADOR_RESIDUOS.html), ábrelo en Chrome, Edge o Firefox y selecciona **Subir imagen**. El archivo incluye los pesos y ejecuta la clasificación en el navegador.

## Revisar el trabajo

- [Informe del primer avance en APA 7](informe/Informe_primer_avance_APA.docx)
- [Código de entrenamiento y exportación](src/)
- [Pesos entrenados](models/)
- [Métricas del modelo final](results/escenas.json) y [resultados de las cuatro pruebas en Edge](results/prueba_capturas.json)
- [Paquete de trabajo del equipo](Clasificador_residuos_equipo.zip)

## Repetir el entrenamiento

Coloca las imágenes de **Garbage Classification V2** bajo `data/archive/data/` y las de **TACO** bajo `data/taco/images/`. Los manifiestos versionados registran las rutas esperadas y las particiones de los experimentos. Obtén las carpetas de imágenes del equipo para usar el mismo subconjunto. El archivo `data/fuente_dataset.json` registra la procedencia del dataset principal.

Ejecuta estos comandos desde la raíz del repositorio. La verificación del HTML requiere Node.js.

```powershell
python -m pip install -r requirements.txt
python src/objetos.py
python src/escenas.py
python src/export_web.py
python tests/verificar_html.py
```

Para regenerar el HTML con el modelo guardado, instala las dependencias y ejecuta sólo `python src/export_web.py`.
