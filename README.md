# TFM_MIOT_YZG
Trabajo de Fin de Master IOT Universidad Complutense de Madrid
Análisis de la madurez del ecosistema de datos abiertos en España

Autor: Yulisme Zambrano
Universidad Complutense de Madrid
Master en Internet de las Cosas

Resumen
Análisis de la madurez del ecosistema de datos abiertos en España
Los portales de datos abiertos desempeñan un papel fundamental en la transparencia y en la promoción de la reutilización de la información pública. No obstante, evaluar de forma objetiva su nivel de madurez y calidad sigue siendo un desafío, especialmente cuando se requiere un análisis comparable entre distintos portales y conjuntos de datos. En este contexto, el presente Trabajo Fin de Máster propone un enfoque metodológico para analizar la madurez de los portales de datos abiertos a partir de los metadatos publicados y de un conjunto de métricas internas orientadas a la calidad, accesibilidad, trazabilidad e interoperabilidad de los datasets.
El trabajo se basa en la definición e implementación de un pipeline compuesto por varias fases, que incluye la selección de conjuntos de datos a partir de criterios homogéneos, la extracción y normalización de metadatos, la construcción de métricas y el análisis comparativo de resultados. La muestra analizada está compuesta por varios portales de datos abiertos en español, seleccionando un volumen representativo de datasets pertenecientes a categorías temáticas comparables como transporte, salud, educación, medio ambiente y demografía.
A partir de los metadatos recopilados, se construyen indicadores que permiten evaluar aspectos como la disponibilidad de formatos reutilizables, la presencia de licencias abiertas, la antigüedad y frecuencia de actualización, la existencia de identificadores trazables y el grado de interoperabilidad técnica y semántica. Estos indicadores se agregan en métricas de madurez que facilitan la comparación entre portales y conjuntos de datos.
Como resultado, el trabajo ofrece una visión estructurada del estado de los portales analizados desde la perspectiva de sus metadatos y capacidades internas, proporcionando una base objetiva para el análisis comparativo y la identificación de buenas prácticas. La metodología propuesta es reproducible y extensible, y puede servir como apoyo para futuras evaluaciones de calidad y madurez en iniciativas de datos abiertos.


Palabras clave: datos abiertos, madurez en portales, reutilización de datos, open data, interoperabilidad, accesibilidad, estándar dcat, portales de datos, análisis empírico.

Estructura
Punto 1 Extraccion_MetadatosDatasets
    Script_Punto1 Extraccion_MetadatosDatasets_V1.ipynb
Punto 2 Reutilizacion_Extraccion_Huellas
    Script_Punto 2 Reutilizacion_Extraccion_Huellas_V1.ipynb
Punto 3 Construccion_MetricasMadurezPortal
    Script_Punto 3 Construccion_MetricasMadurezPortal.ipynb
Punto 4 Visualizacion_GraficasyResultados
    Script_Punto 4 Visualizacion_GraficasyResultados_V2.ipynb

Lenguaje y enfoque técnico

Lenguaje principal: Python 3
Paradigma: Análisis de datos y procesamiento de metadatos
Fuentes de datos: APIs REST (CKAN), DCAT-AP (JSON/RDF), SPARQL endpoints
Entorno recomendado: Jupyter Notebook
Todos los scripts están diseñados para ser reproducibles, auditables y extensibles, utilizando únicamente columnas reales obtenidas en el Punto 1 como base para métricas posteriores.

Requisitos
Python 3.8 o superior
Sistema operativo: Windows, macOS o Linux
Conexión a Internet (para acceder a APIs públicas)
Se recomienda ejecutar los scripts en Jupyter Notebook, ya que:
    Permite ejecución paso a paso
    Facilita depuración y validación de resultados
    Permite visualizar resultados intermedios y finales
pip install notebook
conda install notebook
Librerias necesarias: pip install pandas numpy requests openpyxl rdflib
