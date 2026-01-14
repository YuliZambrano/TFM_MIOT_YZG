# TFM_MIOT_YZG
Trabajo de Fin de Master IOT Universidad Complutense de Madrid
Análisis de la madurez del ecosistema de datos abiertos en España

Autor: Yulisme Zambrano
Universidad Complutense de Madrid
Master en Internet de las Cosas

Descripción general
Este repositorio contiene el código y los recursos desarrollados para el Trabajo de Fin de Máster (TFM) del Máster en Internet de las Cosas (IoT) de la Universidad Complutense de Madrid.
El proyecto tiene como objetivo evaluar la madurez del ecosistema de datos abiertos en España, analizando portales de datos abiertos desde una perspectiva técnica, estructural y de calidad de metadatos, más allá del simple volumen de datasets publicados.
Para ello, se diseña e implementa un pipeline reproducible en Python que permite analizar datasets a nivel individual y construir un índice de madurez basado en métricas objetivas y comparables.
Objetivo del proyecto
Diseñar y evaluar un marco metodológico para analizar  y comparar el grado de madurez de los portales de datos abiertos en España a partir de:
La calidad y completitud de los metadatos.
El uso de formatos abiertos y reutilizables.
El grado de interoperabilidad técnica y semántica.
La accesibilidad real de los recursos publicados.
La trazabilidad temporal y estructural de los datasets.

Marco metodológico
La metodología se basa en un pipeline automatizado de cuatro fases, alineado con estándares de datos abiertos y buenas prácticas internacionales.

Estructura del repositorio
TFM_MIOT_YZG/
│
├── Punto 1 Extraccion_MetadatosDatasets/
│   ├── Update_Script_Punto1_Extraccion_MetadatosDatasets_CKAN_V1.ipynb
│   └── (scripts auxiliares de extracción por portal)
│
├── Punto 2 Reutilizacion_Extraccion_Huellas/
│   └── (scripts experimentales – no forman parte del índice final de madurez)
│
├── Punto 3 Construccion_MetricasMadurezPortal/
│   └── (scripts de cálculo de métricas e índice de madurez)
│
├── Punto 4 Visualizacion_GraficasResultados/
│   └── (notebooks y scripts de visualización de resultados)
│
├── LICENSE
└── README.md

Requisitos técnicos
Entorno recomendado: Jupyter Notebook
Lenguaje principal: Python 3 o superior 
Conexión a Internet (para acceder a APIs públicas)
Se recomienda ejecutar los scripts en Jupyter Notebook, ya que:
    Permite ejecución paso a paso
    Facilita depuración y validación de resultados
    Permite visualizar resultados intermedios y finales
pip install notebook
conda install notebook
Librerías necesarias: pandas, numpy, requests, plotly, matplotlib

Sistema operativo: Windows, macOS o Linux Conexión a Internet (para acceder a APIs públicas) 
Acceso a APIs públicas de portales Open Data.
Paradigma: Análisis de datos y procesamiento de metadatos
Fuentes de datos: APIs REST (CKAN), DCAT-AP (JSON/RDF), SPARQL endpoints

Todos los scripts están diseñados para ser reproducibles, auditables y extensibles, utilizando únicamente columnas reales obtenidas en el Punto 1 como base para métricas posteriores.
Instalación:
git clone https://github.com/YuliZambrano/TFM_MIOT_YZG.git
cd TFM_MIOT_YZG
Para ejecutar cada script el punto 1 genera un resultado que luego será usado como archivo de entrada del siguiente punto.

Reutilización del repositorio
Este repositorio está diseñado para ser reutilizado por:
Investigadores en datos abiertos y gobierno digital.
Administraciones públicas que deseen evaluar sus portales.
Estudiantes de ciencia de datos e IoT.
Proyectos comparativos entre países o regiones.
La metodología es replicable, transparente y extensible.

Punto 1. Extracción, selección y normalización de metadatos
Extracción automática de metadatos desde APIs de portales de datos abiertos (CKAN, Socrata, Opendatasoft)
Aplicación de criterios homogéneos de selección:
-Licencia abierta declarada.
-Accesibilidad de URLs y recursos.
-Identificador o URL estable del dataset.
-Antigüedad mínima y fechas de publicación/modificación.
-Categorías temáticas comparables entre portales.
Análisis automático de:
Disponibilidad y validez de URLs de acceso y descarga.
Tipos y número de formatos publicados.
Licencias declaradas y su normalización.
Existencia de campos clave DCAT (title, description, issued, modified, publisher, theme, distribution).
Resultado: una tabla normalizada de datasets seleccionados por portal.(csv, xlsx)

Punto 2. Extracción, citación de huellas de reutilización. (Aporte metodológico) 

Punto 3. Cálculo de métricas de madurez
A partir de los resultados anteriores se construyen métricas agrupadas en cinco dimensiones:
Accesibilidad
 Disponibilidad efectiva de URLs, recursos descargables y acceso sin restricciones.
Interoperabilidad
 Uso de formatos abiertos, estándares y consistencia semántica.
Calidad de metadatos
 Completitud, coherencia y riqueza descriptiva.
Trazabilidad
 Presencia de fechas, frecuencia de actualización, estabilidad de identificadores y seguimiento temporal.
Licenciamiento
 Claridad, apertura y estandarización de licencias.
Estas métricas se combinan para generar un índice global de madurez por portal.

Punto 4. Generación de resultados
Rankings comparativos de madurez por portal.
Análisis por dimensiones (accesibilidad, interoperabilidad, calidad, trazabilidad).
Distribución de formatos y licencias.
Análisis temporal de publicación y actualización de datasets.
Visualizaciones generadas en Python (Plotly / Matplotlib).

Portales analizados
El pipeline se ha aplicado a múltiples portales de datos abiertos en España, incluyendo:
Portal Nacional: Portal nacional de datos abiertos.
Portales autonómicos y municipales: Junta de Andalucía, Aragón, Castilla y León, Comunidad de Madrid, Región de Murcia
Portales Municipales:  Datos abiertos Barcelona
Se analizan aproximadamente 400-600 datasets por portal, manteniendo equilibrio temático.

