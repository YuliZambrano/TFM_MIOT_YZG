# Analysis of the Maturity of the Open Data Ecosystem in Spain

## Repository Description

This repository contains the code used to develop the Master’s Thesis in Internet of Things at the Complutense University of Madrid.

**Author:** Yulisme Zambrano

---

## Project Summary

The objective of this project is to evaluate the maturity of the open data ecosystem in Spain by analyzing open data portals from a technical, structural, and metadata quality perspective, going beyond the simple volume of published datasets.

A reproducible Python-based pipeline is designed and implemented to analyze datasets at an individual level and to construct a comparative maturity index based on objective and measurable criteria.

---

## Project Objective

To design and evaluate a methodological framework to analyze and compare the maturity level of open data portals in Spain based on:

- The quality and completeness of metadata  
- The use of open and reusable formats  
- The degree of technical and semantic interoperability  
- The real accessibility of published resources  
- The temporal and structural traceability of datasets  

---

## Selected Open Data Portals

For metadata collection, the following portals were selected for analysis:

- Opendata Barcelona  
- Ayuntamiento Junta Andalucia  
- Datos Portal Nacional  
- Datos Comunidad Madrid  
- Datos abiertos Región Murcia  
- Datos abiertos Castilla y León  
- Opendata Aragon  

These portals were selected using homogeneous criteria:

- Declared open license  
- Accessibility of URLs and resources  
- Stable dataset identifier or URL  
- Minimum dataset age and available publication/modification dates  
- Comparable thematic categories across portals  

Approximately 400–600 datasets per portal were analyzed, maintaining thematic balance.

---

## Repository Structure and Instructions

### Step 1: Dataset Metadata Extraction

**Folder:** Extraccion_MetadatosDatasets  

Scripts:

- Script_Punto1_Extraccion_MetadatosDatasets_APIDATA+SPARQL_V1.ipynb  
- Script_Punto1_Extraccion_MetadatosDatasets_OPENDATASOFT_V1.ipynb  
- Script_Punto1_Extraccion_MetadatosDatasets_CKAN_V1.ipynb  
- Script_Punto1_Extraccion_MetadatosDatasets_PaisesFueraUE.ipynb  

Each portal uses different technologies; therefore, the script structure varies mainly in the API connection logic (e.g., CKAN, OpenDataSoft, SPARQL).

When executed, the scripts extract dataset metadata such as:  
title, description, category, identifier, DOI, download_url, license, issued, modified, format, among others.

**Output:** CSV and XLSX files.

Additionally, a specific version for non-EU countries is included, designed to extract empirical reuse traces. This contribution is intended for future research projects.

---

### Step 2: Reuse Trace Extraction

**Folder:** Reutilizacion_Extraccion_Huellas  

Scripts:

- Script_Punto2_Reutilizacion_Extraccion_Huellas_V1.ipynb  
- Script_Punto2_Reutilizacion_Extraccion_Huellas_ParaFueraUE.ipynb  

This script represents a methodological contribution for future research. Multiple versions aim to identify dataset reuse traces in sources describing research projects.

**Input:** metadata files generated in Step 1  
**Output:** CSV and XLSX files  

The non-EU version successfully retrieved reuse results for the selected portal.

---

### Step 3: Portal Maturity Metrics Construction

Script:

- Script_Punto3_Construccion_MetricasMadurezPortal.ipynb  

This script constructs the metrics required for the analysis of results. It uses the Step 1 output file as input and generates two output files in CSV and XLSX formats.

---

### Step 4: Visualization and Results

Script:

- Script_Punto4_Visualizacion_GraficasyResultados_V3.ipynb  

This script generates all visualizations required for the analysis. It produces an HTML file containing:

- Individual portal charts  
- Comparative charts across portals  

The input file is the output generated in Step 3.

---

## Result Datasets

The resulting datasets can be found at the following link:  

**Proyecto zenodo**  
https://zenodo.org/records/18246926

---

## Technical Requirements

- Recommended environment: Jupyter Notebook  
- Main language: Python 3 or higher  
- Operating system: Windows, macOS, or Linux  
- Internet connection: Required to access public APIs  

It is recommended to run the scripts in Jupyter Notebook because it:

- Allows step-by-step execution  
- Facilitates debugging and validation  
- Enables visualization of intermediate and final results  

### Required Libraries

- pandas  
- numpy  
- requests  
- plotly  
- matplotlib  

### Data Sources

- REST APIs (CKAN)  
- DCAT-AP (JSON/RDF)  
- SPARQL endpoints  

---

## Installation

git clone https://github.com/YuliZambrano/TFM_MIOT_YZG.git  
pip install notebook  
# or  
conda install notebook  
cd TFM_MIOT_YZG  

---

## Repository Reuse

This repository is designed to be reused by:

- Open data and digital government researchers  
- Public administrations aiming to assess their open data portals  
- Data science and IoT students  
- Comparative projects across countries or regions  

The methodology is replicable, transparent, and extensible.

---

# **Versión en Español**

## **Análisis de la madurez del ecosistema de datos abiertos en España**

Este repositorio contiene el código utilizado para la realización del Trabajo Fin de Máster en Internet de la Cosas de la Universidad Complutense de Madrid.

**Autor:** Yulisme Zambrano

---

## **Resumen del Proyecto**

El proyecto tiene como objetivo evaluar la madurez del ecosistema de datos abiertos en España, analizando portales de datos abiertos desde una perspectiva técnica, estructural y de calidad de metadatos, más allá del simple volumen de datasets publicados.

Para ello, se diseña e implementa un pipeline reproducible en Python que permite analizar datasets a nivel individual y construir un índice de madurez basado en métricas objetivas y comparables.

---

## **Objetivo del proyecto**

Diseñar y evaluar un marco metodológico para analizar y comparar el grado de madurez de los portales de datos abiertos en España a partir de:

- La calidad y completitud de los metadatos.  
- El uso de formatos abiertos y reutilizables.  
- El grado de interoperabilidad técnica y semántica.  
- La accesibilidad real de los recursos publicados.  
- La trazabilidad temporal y estructural de los datasets.  

---

## **Portales de datos abiertos seleccionados**

Para la recopilación de los metadatos se seleccionaron, los siguientes portales para el análisis: Opendata Barcelona, Ayuntamiento Junta Andalucia, Datos Portal Nacional, Datos Comunidad Madrid, Datos abiertos Región Murcia, Datos abiertos Castilla y León, Opendata Aragon

Los cuales se diseñaron con la aplicación de criterios homogéneos de selección:

● Licencia abierta declarada.
● Accesibilidad de URLs y recursos.
● Identificador o URL estable del dataset.
● Antigüedad mínima y fechas de publicación/modificación.
● Categorías temáticas comparables entre portales.


Se analizan aproximadamente 400–600 datasets por portal, manteniendo equilibrio temático.

---

## Instrucciones y composicion del repositorio:

### Punto 1 Extraccion_MetadatosDatasets

**Carpeta:** Extraccion_MetadatosDatasets  

Scripts:

- Script_Punto1_Extraccion_MetadatosDatasets_APIDATA+SPARQL_V1.ipynb  
- Script_Punto1_Extraccion_MetadatosDatasets_OPENDATASOFT_V1.ipynb  
- Script_Punto1_Extraccion_MetadatosDatasets_CKAN_V1.ipynb  
- Script_Punto1_Extraccion_MetadatosDatasets_PaisesFueraUE.ipynb  

Cada portal usa tecnologías diferente por lo que el script cambia la estructura en alguna de sus funciones, la principal es la consulta para conectarse a la api como por ejemplo (ckan, opendatasoft, sparql) Al ejecutar el script se obtienen extraen los metadatos (title,description, category, identifier, doi, download_url, license, issued, modified, format.... entre otros). El resultado es un archivo en formato csv y xlsx.
Adicionalmente se incluye una versión distinta para PaisesFueraUe que se ha diseñado para la extracción de huellas de reutilizacion empírica. Esto ha quedado como aporte para futuros proyectos.


---

### Punto 2 Reutilizacion_Extraccion_Huellas

**Carpeta:** Reutilizacion_Extraccion_Huellas  

Scripts:

Aca se muestran 4 versiones del script de reutilizacion, para usar con el resultado del punto 1

- Script_Punto2_Reutilizacion_Extraccion_Huellas_V1.ipynb  
- Script_Punto2_Reutilizacion_Extraccion_Huellas_V2.ipynb  
- Script_Punto2_Reutilizacion_Extraccion_Huellas_V3.ipynb  
- Script_Punto2_Reutilizacion_Extraccion_Huellas_V4.ipynb  
- Script_Punto2_Reutilizacion_Extraccion_Huellas_ParaFueraUE.ipynb  

Este script es un aporte metodológico para trabajos futuros, se realizan varias versiones que buscan el resultado de huellas de reutilización de un dataset, en fuentes que describen proyectos de investigación, para este script se usa como punto de entrada el archivo resultado del Punto 1. Está la version de ParaFueraUE que para el portal usado si trajo resultados. Se coloca el nombre del archivo de entrada y se ejecuta. Los resultados traeran archivos en csv y xlsx.


---

### Punto 3 Construccion_MetricasMadurezPortal

Script:

- Script_Punto3_Construccion_MetricasMadurezPortal.ipynb  

Este script construye las métricas para poder realizar el análisis de resultados. Este script usa el archivo resultado del Punto 1 como entrada, al ejecutar el script trae dos archivos en csv y xlsx.

---

### Punto 4 Visualizacion_GraficasyResultados

Script:

- Script_Punto 4 Visualizacion_GraficasyResultados_PorPortal.ipynb
- Script_Punto 4 Visualizacion GraficasyResultados_TodoslosPortales.ipynb
- Script_Punto4_Boxplot_Desviacion.ipynb
- Script_Punto4_Test_ANOVA_Kruskal.ipynb


Este script construye todas las gráficas necesarias del análisis de resultados, el script imprime en un archivo html, todas las gráficas de cada portal y las de comparaciones entre portales. Usa el archivo resultado del Punto 3 como archivo de entrada.
Adicionalmente se agregan 2 script que complementan el analisis de resultados Boxplot_Desviacion y Test_ANOVA_Kruskal



---

## Resultados Datasets

Estos conjuntos de datos de resultados se pueden encontrar en el siguiente enlace:   

**Proyecto zenodo**  
https://zenodo.org/records/18246926

---

## Requisitos técnicos:

Entorno recomendado: Jupyter Notebook
Lenguaje principal: Python 3 o superior 
Conexión a Internet (para acceder a APIs públicas)
Se recomienda ejecutar los scripts en Jupyter Notebook, ya que:
    Permite ejecución paso a paso
    Facilita depuración y validación de resultados
    Permite visualizar resultados intermedios y finales

### Librerías necesarias

- pandas  
- numpy  
- requests  
- plotly  
- matplotlib  

---

### Sistema operativo: 

Windows, macOS o Linux Conexión a Internet (para acceder a APIs públicas) 
Acceso a APIs públicas de portales Open Data.
Paradigma: Análisis de datos y procesamiento de metadatos

---

### Fuentes de datos

- REST APIs (CKAN)  
- DCAT-AP (JSON/RDF)  
- SPARQL endpoints  

---

## Installation

git clone https://github.com/YuliZambrano/TFM_MIOT_YZG.git  
pip install notebook  
# or 
conda install notebook  
cd TFM_MIOT_YZG  

---

## **Reutilización del repositorio**

Este repositorio está diseñado para ser reutilizado por:

- Investigadores en datos abiertos y gobierno digital.  
- Administraciones públicas que deseen evaluar sus portales.  
- Estudiantes de ciencia de datos e IoT.  
- Proyectos comparativos entre países o regiones.  

La metodología es replicable, transparente y extensible.
