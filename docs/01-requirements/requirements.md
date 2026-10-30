# Especificación de Requerimientos de Software y Ciencia de Datos (IEEE 830 / ISO 29148)
**Proyecto**: Análisis Relacional y Causal de Exportaciones e Importaciones de Colombia en la Alianza del Pacífico  
**Versión**: 2.0.0  
**Fecha**: 2026-10-29  
**Fase PDCO**: `PLAN`  
**Skill Activa**: `01-requirements`  

---

## 1. Definición y Segmentación del Problema

### ¿Qué problema resuelve?
Determinar con rigor econométrico y estadístico si la diversificación de la canasta exportadora (margen extensivo $N_{i,t}$ e índice HHI invertido $D_{i,t}$) y la posición competitiva frente a las importaciones ($\theta_{i,t}, SCR_{i,t}, GL_{i,t}$) han impulsado un **crecimiento real** en las exportaciones de Colombia hacia los socios de la **Alianza del Pacífico** (Chile, México y Perú) a través de tres hitos históricos (`2013–2014`, `2018–2020` y `2024–2025`).

### Stakeholders
| Rol | Interés Analítico | Nivel de Influencia |
| :--- | :--- | :---: |
| Investigador Principal / Analista Económico | Contrastar $H_0$ vs $H_1$ con pruebas paramétricas, no paramétricas, econometría de panel y ML causal. | Alto |
| Ingeniero de Datos / Arquitecto | Garantizar trazabilidad DAMA-BOK sobre los 168 cortes mensuales del DANE (`Expo_*` e `Impo_*`). | Alto |
| Tomadores de Decisión en Política Comercial | Identificar productos clave con ventaja comparativa y potencial de comercio intraindustrial. | Medio |

---

## 2. Requerimientos Funcionales (RF)

| ID | Descripción | Prioridad | Artefacto Asociado |
| :--- | :--- | :---: | :--- |
| **RF-001** | El sistema debe ingerir y normalizar los **84 cortes mensuales de Exportaciones** (`Expo_2013` a `Expo_2025`) y los **84 cortes mensuales de Importaciones** (`Impo_2013` a `Impo_2025`) desde microdatos oficiales del DANE. | Alta | `unzip_exporter.py`, `dane_impo_downloader.py`, `unzip_importer.py` |
| **RF-002** | El sistema debe homologar la codificación de países entre el régimen DIAN (`2013–2024`) y el estándar ONU M49 (`2025`) hacia ISO-3 (`COD_PAI4`) y normalizar las subpartidas arancelarias (`POSAR` / `NABAN`) a 10 dígitos. | Alta | `impo_loader.py`, `crispdm_pipeline.py` |
| **RF-003** | El sistema debe clasificar tanto exportaciones como importaciones en **Canasta Ofertada** (8 complejos clave), **Canasta Demandada** (8 complejos clave) y **Resto Arancelario**, generando análisis desagregado por producto (`HS4`). | Alta | `crispdm_pipeline.py`, `analisis_desagregado_productos_ap.csv` |
| **RF-004** | El sistema debe ejecutar un Análisis Exploratorio de Datos (EDA) riguroso que integre perfilamiento automatizado (`ydata-profiling`), ajuste paramétrico de distribuciones (`distfit`) y visualizaciones interactivas (`plotly`). | Alta | `Analisis_exportaciones.ipynb`, `update_notebook.py` |
| **RF-005** | El sistema debe contrastar $H_0$ vs $H_1$ mediante pruebas paramétricas ($t$-Student pareada, ANOVA) y no paramétricas (Wilcoxon, Prueba de los Signos, Kruskal-Wallis), econometría de panel (FE-OLS) y Machine Learning Causal (*Double Machine Learning* — DML + Random Forest / Gradient Boosting) evaluando métricas predictivas ($R^2, \text{RMSE}, \text{MAE}, \text{MAPE}$). | Alta | `crispdm_pipeline.py`, `Analisis_exportaciones.ipynb` |

---

## 3. Requerimientos No Funcionales (RNF)

| ID | Tipo | Descripción | Métrica de Verificación |
| :--- | :--- | :--- | :--- |
| **RNF-001** | Rendimiento | La recarga del panel y ejecución del Notebook desde los cubos pre-agregados debe completarse en menos de 30 segundos. | Tiempo de ejecución `< 30s` |
| **RNF-002** | Mantenibilidad | Cumplimiento estricto de PEP 8, Clean Code y principios SOLID en todos los módulos bajo `src/`. | `0` violaciones SOLID |
| **RNF-003** | Reproducibilidad Científica | Semillas aleatorias fijas (`random_state=42`) en todos los estimadores de Machine Learning y *cross-fitting* causal. | Resultados 100% deterministas |
