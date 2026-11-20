# predict_future_sales

Pronóstico de ventas sobre el dataset **Predict Future Sales** de Kaggle: ventas diarias de la
cadena de tiendas rusa 1C Company, entre enero de 2013 y octubre de 2015.

El objetivo es predecir `item_cnt_month`, las unidades que venderá cada par tienda-producto en
**noviembre de 2015**. El repositorio cubre el recorrido completo: del análisis exploratorio al
archivo de predicciones listo para enviar.

- Maestría · 4T 2026 · Product Management · Clase Cloud
- **Autor:** Danny García
- **Bitácora de decisiones:** [`ESTADO_PROYECTO.md`](ESTADO_PROYECTO.md)

## Los notebooks, en orden

| Notebook | Qué hace | Qué produce |
|---|---|---|
| [`01_exploratory_data_analysis`](notebooks/01_exploratory_data_analysis.ipynb) | Análisis exploratorio: calidad, estacionalidad, ACF/PACF | Los hallazgos que orientan todo lo demás |
| [`02_feature_exploration`](notebooks/02_feature_exploration.ipynb) | Decide qué variables crear y qué transformación recibe cada una | La lista escrita del proceso |
| [`03_feature_creation`](notebooks/03_feature_creation.ipynb) | Convierte esa lista en un `Pipeline` de scikit-learn | `models/pipeline_features.pkl` |
| [`04_model_training_n_tunning`](notebooks/04_model_training_n_tunning.ipynb) | Enfrenta 15 modelos y elige uno por RMSE | `models/pipeline_completo.pkl` |
| [`05_inference_calculation`](notebooks/05_inference_calculation.ipynb) | Predice `test.csv` con el artefacto | `data/processed/submission.csv` |

Cada notebook consume la salida del anterior y no se adelanta. El orden importa: la fuga de
información se cuela cuando una fase usa datos que la siguiente debería desconocer.

---

## Puesta en marcha

### 1. Clonar

```bash
git clone https://github.com/dannygarcia-hue/predict_future_sales.git
cd predict_future_sales
```

### 2. Entorno

```bash
conda create -n env_predict_future_sales python=3.11 -y
conda activate env_predict_future_sales
pip install -r requirements.txt
```

`requirements.txt` es la **fuente de verdad** del entorno: toda librería que entre se agrega ahí
en el mismo paso, con la versión fijada.

### 3. Datos

Los datos **no están versionados** (`sales_train.csv` pesa 94.6 MB). Hay que descargarlos de
[la página de la competencia](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data)
—requiere cuenta gratuita de Kaggle— y descomprimirlos en `data/raw/`:

```
data/raw/
├── item_categories.csv
├── items.csv
├── sales_train.csv
├── sample_submission.csv
├── shops.csv
└── test.csv
```

### 4. Ejecutar el notebook

```bash
jupyter lab notebooks/01_exploratory_data_analysis.ipynb
```

> **Nota para Windows.** Los catálogos de tiendas y productos están en cirílico. Al ejecutar
> scripts desde la consola hay que forzar UTF-8 con `python -X utf8 ...`; la consola usa cp1252
> y de lo contrario falla con `UnicodeEncodeError`.

---

## Los datos

| Archivo | Contenido | Filas |
|---|---|---|
| `sales_train.csv` | Ventas diarias históricas (una fila por producto vendido en una tienda un día) | 2 935 849 |
| `test.csv` | Pares tienda-producto a predecir (nov 2015) | 214 200 |
| `sample_submission.csv` | Formato de envío; confirma el target `item_cnt_month` | 214 200 |
| `items.csv` | Catálogo de productos | 22 170 |
| `item_categories.csv` | Catálogo de categorías | 84 |
| `shops.csv` | Catálogo de tiendas | 60 |

`date_block_num` es el número de mes consecutivo (0 = enero 2013, 33 = octubre 2015).
`item_cnt_day` es la cantidad vendida en el día; los valores negativos son devoluciones.
El dataset **no trae** una columna de ingresos: se construye como `item_price × item_cnt_day`.

---

## Contenido del EDA

1. Comprensión inicial del dataset
2. Validación de la estructura de los datos
3. Valores faltantes y duplicados
4. Análisis descriptivo general
5. Exploración temporal
6. Análisis comparativo entre sucursales y productos
7. Visualizaciones clave
8. Autocorrelación (ACF) y autocorrelación parcial (PACF)
9. Insights y conclusiones

---

## El pipeline

El objeto que se guarda en `models/pipeline_completo.pkl` va de **datos crudos a predicciones**.
Recibe únicamente `shop_id` e `item_id`, y se encarga de todo lo demás:

```
UnirCatalogos          -> pega categoría y nombres desde los catálogos que lleva dentro
CaracteristicasCalendario -> mes, año, diciembre, días y fines de semana del mes
LagFeatures            -> rezagos 1, 2, 3 y 12; medias móviles; agregados por tienda y producto
Mapper                 -> extrae la ciudad del nombre de la tienda y el tipo de la categoría
ColumnTransformer      -> imputa, recorta extremos, transforma el sesgo, escala y codifica
<modelo ganador>       -> predice item_cnt_month
```

Los operadores propios viven en [`predict_future_sales/features.py`](predict_future_sales/features.py),
no dentro de los notebooks. Un `.pkl` no guarda el código de las clases, solo dónde encontrarlas:
si vivieran en un notebook, el artefacto no se podría abrir desde ningún otro sitio.

### Verificación de que no hay fuga de información

El riesgo central de un modelo de pronóstico es que un rezago mire hacia adelante. El síntoma es
un resultado excelente en validación y un fracaso en producción, sin ninguna señal por el camino.

Aquí se verifica con una prueba automatizada sobre datos de juguete, donde la respuesta correcta
se conoce de memoria:

```bash
pytest tests/ -v
```

---

## Estructura del proyecto

Generada con [Cookiecutter Data Science v2.3.0](https://cookiecutter-data-science.drivendata.org/).

```
├── data
│   ├── external        <- Datos de terceros
│   ├── interim         <- Datos intermedios ya transformados
│   ├── processed       <- Conjuntos finales para modelado
│   └── raw             <- Datos originales de Kaggle (no versionados)
├── docs                <- Documentación mkdocs
├── models              <- Modelos entrenados y serializados
├── notebooks           <- Notebooks de Jupyter
├── references          <- Enunciado, método de referencia y diccionarios de datos
├── reports
│   └── figures         <- Gráficas generadas
├── predict_future_sales  <- Código fuente: operadores de ingeniería de características
├── tests               <- Pruebas automatizadas (la de no-fuga es la importante)
├── ESTADO_PROYECTO.md  <- Bitácora de decisiones y su justificación
├── pyproject.toml      <- Configuración del paquete y de ruff
└── requirements.txt    <- Dependencias con versiones fijadas
```

---

## Licencia

MIT. Ver [`LICENSE`](LICENSE).
