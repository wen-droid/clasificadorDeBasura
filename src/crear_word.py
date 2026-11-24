"""Informe académico autónomo basado exclusivamente en la ejecución guardada."""
import json
from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
ROOT=Path(__file__).resolve().parents[1]

def main():
    read=lambda n:json.loads((ROOT/'results'/n).read_text(encoding='utf-8'))
    a=read('auditoria.json');c=read('configuracion.json');m=read('metricas.json')
    d=Document();s=d.sections[0];s.page_width=Cm(21);s.page_height=Cm(29.7)
    s.top_margin=s.bottom_margin=Cm(2);s.left_margin=s.right_margin=Cm(2.2)
    for name in ['Normal','Title','Subtitle','Heading 1','Heading 2']:
        style=d.styles[name];style.font.name='Times New Roman';style.font.color.rgb=RGBColor(0,0,0)
        for border in list(style.element.xpath('.//w:pBdr')):border.getparent().remove(border)
    d.styles['Normal'].font.size=Pt(12);d.styles['Normal'].paragraph_format.line_spacing=1.15
    d.styles['Normal'].paragraph_format.space_after=Pt(7)
    d.styles['Title'].font.size=Pt(25)
    def p(text):return d.add_paragraph(text)
    def h(text):d.add_heading(text,1)
    def table(title,headers,rows):
        cap=p(title);cap.paragraph_format.keep_with_next=True
        t=d.add_table(rows=1,cols=len(headers));t.style='Table Grid'
        for cell,text in zip(t.rows[0].cells,headers):cell.text=str(text)
        tr=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(tr)
        for row in rows:
            for cell,text in zip(t.add_row().cells,row):cell.text=str(text)
        for i,row in enumerate(t.rows):
            for cell in row.cells:
                for pp in cell.paragraphs:
                    pp.paragraph_format.space_after=Pt(3)
                    if i==0:pp.paragraph_format.keep_with_next=True
                    for r in pp.runs:r.font.size=Pt(9);r.bold=i==0
        p('Fuente elaboración propia')
    def image(file,caption,width=15):
        pp=d.add_paragraph();pp.paragraph_format.keep_with_next=True
        pp.add_run().add_picture(str(ROOT/'results'/file),width=Cm(width))
        pp=p(caption+'\nFuente elaboración propia');pp.alignment=WD_ALIGN_PARAGRAPH.CENTER
        for r in pp.runs:r.font.size=Pt(9)
    p('UNIVERSIDAD MAYOR DE SAN SIMÓN')
    d.add_paragraph('Clasificación visual de residuos domésticos en diez categorías','Title')
    p('Primer avance de aprendizaje profundo en visión por computadora')
    p('Docente Lic Patricia Rodríguez Bilbao')
    p('Grupo e integrantes __________________________________________')
    p('Materia ____________________________________________________')
    p('Cochabamba 22 de noviembre de 2026')
    p(f'Se desarrolló una CNN entrenada desde cero sobre imágenes de Garbage Classification V2 de Kaggle. La auditoría consideró {a["input_images"]} archivos y conservó {a["usable_images"]} imágenes para entrenamiento, validación y prueba. El modelo obtuvo {m["accuracy"]:.1%} de exactitud y {m["macro_f1"]:.3f} de macro F1 sobre {m["n_test"]} imágenes reservadas. Se implementó una aplicación HTML con inferencia local y un cuaderno de Colab conectado a Drive.')
    d.add_page_break()
    h('I Introducción')
    p('La clasificación visual de residuos domésticos permite estudiar el reconocimiento de tipos de desechos mediante imágenes. Este proyecto propone un prototipo educativo que recibe una fotografía con un residuo predominante y asigna una de diez categorías. La clasificación apoya la identificación inicial; la disposición final requiere considerar condiciones y normas que no pueden inferirse únicamente de los píxeles.')
    p('Se emplea una red neuronal convolucional propia en PyTorch con parámetros inicializados aleatoriamente. El trabajo comprende selección y auditoría de datos, preprocesamiento, entrenamiento, selección del checkpoint por validación y evaluación sobre una partición reservada. La interfaz ejecuta los pesos aprendidos en el navegador.')
    h('II Objetivos')
    p('Objetivo general: desarrollar y evaluar un prototipo de visión por computadora que clasifique imágenes de residuos domésticos en diez categorías mediante una CNN entrenada desde cero.')
    for text in ['Auditar el dataset y registrar imágenes válidas, duplicados y conflictos de etiqueta.', 'Separar grupos visualmente próximos entre entrenamiento, validación y prueba.', 'Entrenar una CNN con inicialización aleatoria y documentar su configuración.', 'Evaluar métricas globales y por categoría, curvas y matriz de confusión.', 'Integrar el checkpoint en un HTML autónomo y facilitar la reproducción mediante Drive y Colab.']:d.add_paragraph(text,'List Bullet')
    h('III Marco teórico')
    p('La visión por computadora comprende métodos de extracción de información a partir de imágenes. Clasificar una imagen asigna una etiqueta a su contenido global; detectar objetos implica además localizar instancias, y segmentar asigna etiquetas a regiones o píxeles. Este estudio aborda clasificación de imágenes [1].')
    p('Las convoluciones aprenden filtros compartidos espacialmente. Las activaciones ReLU introducen no linealidad y MaxPooling reduce la resolución espacial. Una representación final alimenta capas densas que generan logits para las diez clases. Softmax convierte esos valores en puntuaciones que suman uno, sin garantizar que estén calibradas como probabilidades de acierto [1, 2].')
    p('La entropía cruzada compara los logits con las etiquetas. La retropropagación calcula gradientes y Adam actualiza los pesos. Dropout y weight decay regularizan el ajuste; las variaciones de orientación, brillo y contraste se aplican solo en entrenamiento. El aumento no crea objetos independientes ni reemplaza la diversidad del dataset [1, 2].')
    p('La validación permite seleccionar la época sin utilizar la prueba. La exactitud es la proporción de aciertos. Precision mide la proporción de predicciones de una clase que son correctas; recall mide qué fracción de sus ejemplos reales se recupera. F1 es la media armónica de precision y recall; macro F1 promedia las clases con igual peso. La matriz de confusión presenta clases reales por filas y predicciones por columnas.')
    h('IV Marco referencial')
    p('Problema: ¿qué desempeño alcanza una CNN entrenada desde cero para distinguir diez tipos de residuos domésticos en imágenes y qué errores presenta entre categorías? El uso propuesto es educativo, como apoyo a la identificación visual en la separación inicial de residuos en Cochabamba.')
    p('El conjunto Garbage Classification V2, publicado en Kaggle por sumn2u, contiene diez categorías [3]. Se empleó la carpeta original de la versión descargada. Las etiquetas se traducen como baterías, biológico, cartón, ropa, vidrio, metal, papel, plástico, zapatos y otros residuos. Esta taxonomía combina materiales y tipos de objetos; por ello el problema se define como clasificación de tipos de residuos.')
    p('El alcance comprende fotografías con un objeto predominante. No se realiza detección, conteo ni clasificación multietiqueta de basura acumulada. La evaluación experimental no establece eficacia municipal ni determina peligrosidad o reciclabilidad.')
    h('V Ingeniería')
    p(f'Auditoría: se inspeccionaron {a["input_images"]} archivos, se identificaron {a["exact_duplicates"]} repeticiones exactas, {len(a["invalid"])} archivos inválidos y {len(a["conflicting_pixel_hashes"])} hashes con etiquetas contradictorias. Las reglas de depuración excluyen repeticiones y conflictos de etiqueta; se conservaron {a["usable_images"]} imágenes. El manifiesto registra ruta original, etiqueta, hash de píxeles, dHash, grupo y partición.')
    p('Se corrigió orientación EXIF, se convirtió a RGB y se ajustó a 96 por 96 píxeles con interpolación bilineal y relleno gris conservando proporción. Esa transformación determinista se guardó en PNG antes de entrenar. Las variaciones aleatorias se aplicaron después de separar las particiones y solo al entrenamiento. La entrada se normaliza al intervalo de cero a uno.')
    p('El agrupamiento heurístico une imágenes de la misma categoría con distancia Hamming dHash menor o igual a seis mediante componentes conectados. Los grupos se distribuyen con objetivo 70 por ciento entrenamiento, 15 validación y 15 prueba y semilla 42. No se comparten grupos automáticos entre particiones; ello no garantiza identidad física independiente de todos los objetos.')
    table('Tabla 1 Distribución utilizada',['Categoría','Entrena','Valida','Prueba'],[[x,*[a['counts'][x].get(k,0) for k in ['train','val','test']]] for x in a['classes']])
    p('Arquitectura: tres bloques Conv2D de kernel 3 por 3 con 24, 48 y 96 filtros, ReLU y MaxPooling 2 por 2; AveragePooling 3 por 3; aplanado de 1536 valores; Dense de 64 unidades; ReLU, dropout 0.4 y Dense de diez logits. Todos los parámetros se ajustan desde inicialización aleatoria, sin cargar pesos externos.')
    table('Tabla 2 Configuración ejecutada',['Parámetro','Valor'],[['Python',c['python']],['PyTorch',c['torch_version']],['Dispositivo',c['gpu'] or c['device_used']],['Parámetros',c['parameters']],['Épocas máximas',c['epochs']],['Épocas ejecutadas',c['epochs_executed']],['Época seleccionada',c['best_epoch']],['Optimizador','Adam, lr .001, weight decay .0001'],['Lote',c['batch_size']],['Semilla',c['seed']],['Tiempo segundos',round(c['elapsed_seconds'],1)]])
    p('La pérdida de entrenamiento pondera las clases mediante sus frecuencias en entrenamiento. La validación usa entropía cruzada sin ponderación. Se conserva el checkpoint de menor pérdida de validación, con parada temprana según la paciencia registrada. Las pérdidas ponderada y no ponderada no son magnitudes directamente comparables.')
    image('curvas.png','Figura 1 Curvas de entrenamiento y validación')
    p(f'El modelo seleccionado obtuvo {m["accuracy"]:.1%} de exactitud y {m["macro_f1"]:.3f} de macro F1 en {m["n_test"]} imágenes de prueba. La referencia constante de clase mayoritaria alcanza {m["baseline_accuracy"]:.1%}. La selección utilizó únicamente validación; la prueba se evaluó al finalizar el ajuste.')
    table('Tabla 3 Métricas por clase',['Categoría','Precision','Recall','F1','Soporte'],[[r['class_name'],f"{r['precision']:.3f}",f"{r['recall']:.3f}",f"{r['f1']:.3f}",r['support']] for r in m['per_class']])
    image('matriz_confusion.png','Figura 2 Matriz de confusión de prueba',13)
    p('El prototipo incluye los pesos aprendidos y un motor JavaScript que reproduce el preprocesamiento y las operaciones de la CNN en un Web Worker. El HTML permite carga de fotografías y presenta las diez puntuaciones. La captura de cámara depende del permiso del navegador y su prueba física queda pendiente.')
    if (ROOT/'results/verificacion_html.json').exists():
        v=read('verificacion_html.json');p(f'La comprobación numérica comparó {v["samples"]} casos entre PyTorch y JavaScript y obtuvo coincidencia en las {v["same_prediction"]} predicciones. La diferencia máxima de puntuaciones fue {v["max_probability_difference"]:.3g}. Es una prueba de equivalencia de implementación y no una evaluación externa.')
    p('Limitaciones: el agrupamiento por dHash no identifica todos los objetos repetidos; las categorías presentan desbalance y condiciones de captura heterogéneas, incluyendo fotografías de productos y residuos. No se dispone de una prueba externa nueva. La red puede asignar una categoría conocida a una imagen ajena al problema. Estas limitaciones restringen la interpretación de las métricas internas.')
    h('VI Conclusiones')
    p(f'Se desarrolló una CNN para las diez categorías del dataset y se completó el ciclo de auditoría, preparación, entrenamiento y evaluación. El modelo obtuvo {m["accuracy"]:.1%} de exactitud y macro F1 de {m["macro_f1"]:.3f}; estos resultados corresponden a la partición reservada de la versión utilizada.')
    p('La aplicación ejecuta el modelo localmente y el cuaderno permite reproducir el procedimiento desde una carpeta compartida de Drive, guardando cada ejecución por separado. Las mejoras propuestas incluyen verificar grupos por objeto y medir fotografías externas. El segundo avance podrá estudiar representaciones de un autoencoder y agrupamiento sin etiquetas.')
    h('VII Bibliografía')
    for ref in ['[1] Goodfellow I, Bengio Y y Courville A. Deep Learning. MIT Press, 2016. https://www.deeplearningbook.org/.','[2] PyTorch. Neural Networks y Reproducibility. https://docs.pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html y https://docs.pytorch.org/docs/stable/notes/randomness.html.','[3] sumn2u. Garbage Classification V2. Kaggle. https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2. Versión descargada con 12259 imágenes en la carpeta original. Consulta 22 de noviembre de 2026.']:p(ref)
    footer=s.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    out=ROOT/'informe/Informe_clasificacion_residuos.docx';out.parent.mkdir(exist_ok=True);d.save(out);print(out)

if __name__=='__main__':main()
