# predict_future_sales

Análisis exploratorio de datos (EDA) sobre el dataset **Predict Future Sales** de Kaggle:
ventas diarias de la cadena de tiendas rusa 1C Company, entre enero de 2013 y octubre de 2015.

El objetivo final de la competencia es predecir las ventas **mensuales** de cada par
tienda-producto para noviembre de 2015. Este repositorio cubre el EDA que prepara esa etapa.

- **Proyecto #1** — Maestría · 4T 2026 · Product Management · Clase Cloud
- **Autor:** Danny García
- **Entregable principal:** [`notebooks/01_exploratory_data_analysis.ipynb`](notebooks/01_exploratory_data_analysis.ipynb)
- **Bitácora de decisiones:** [`ESTADO_PROYECTO.md`](ESTADO_PROYECTO.md)

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
├── predict_future_sales  <- Código fuente del proyecto
├── ESTADO_PROYECTO.md  <- Bitácora de decisiones y su justificación
├── pyproject.toml      <- Configuración del paquete y de ruff
└── requirements.txt    <- Dependencias con versiones fijadas
```

---

## Licencia

MIT. Ver [`LICENSE`](LICENSE).
