# Estado del proyecto — `predict_future_sales`

Documento de continuidad. Registra **las decisiones y su justificación**, no solo lo que se
hizo. Se actualiza al cerrar cada fase.

- **Proyecto:** #1 Exploratory Data Analysis (cerrado) · #2 Pipeline de Machine Learning (en curso)
- **Dataset:** [Predict Future Sales](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales) (Kaggle, 1C Company)
- **Autor:** Danny García — Maestría · 4T 2026 · Product Management · Clase Cloud
- **Tipo de problema:** pronóstico de series de tiempo jerárquico (tienda × producto), mensual
- **Objetivo final de la competencia:** predecir `item_cnt_month` de noviembre de 2015 (mes 34)
- **Método de referencia:** [`references/TRANSFERENCIA_METODO.md`](references/TRANSFERENCIA_METODO.md) (proyecto `house_pricing`)
- **Entorno:** conda `env_predict_future_sales`, Python 3.11.18
- **Repositorio:** <https://github.com/dannygarcia-hue/predict_future_sales>

---

## Estado actual

| Fase | Estado | Fecha |
|---|---|---|
| 0 · Estructura y entorno | **Cerrada** — repo publicado en GitHub | 2026-11-12 |
| 1 · EDA (`01_exploratory_data_analysis.ipynb`) | **Cerrada** — 127 celdas, 18 figuras, 0 errores | 2026-11-12 |
| 2 · Ingeniería de características (`02_feature_exploration`) | **Cerrada** | 2026-11-19 |
| 3 · Pipeline de sklearn (`03_feature_creation`) | **Cerrada** | 2026-11-20 |
| 4 · Torneo de modelos (`04_model_training_n_tunning`) | **Cerrada** | 2026-11-20 |
| 5 · Inferencia (`05_inference_calculation`) | **Cerrada** | 2026-11-20 |

**Alcance acordado:** Proyectos #1 y #2. El #2 (`references/Proyecto_2_Pipeline_Predict_Future_Sales.pdf`)
pide cuatro notebooks encadenados, de la ingeniería de características a la inferencia sobre
`test.csv`, con dos artefactos `.pkl` y un `.csv` de predicciones.

---

## 1. Decisiones de la Fase 0

### 1.1 El método de `house_pricing` se adapta, no se copia

Su §1 obliga a preguntar si es el mismo tipo de problema. **No lo es.** El origen era regresión
tabular de 1460 × 81 con filas independientes; aquí hay 2 935 849 transacciones, estructura
temporal explícita, jerarquía tienda × producto y un target que **no existe como columna**: hay
que construirlo agregando a mes.

| Del método | Decisión | Razón |
|---|---|---|
| Estructura Cookiecutter DS v2.3.0 | **Se replica** | El enunciado la exige (punto 1a) |
| Entorno conda dedicado + `requirements.txt` con versiones fijadas | **Se replica** | Reproducibilidad; misma máquina, mismo riesgo de DLLs |
| `ESTADO_PROYECTO.md` de continuidad | **Se replica** | Este documento |
| `.gitignore` con `data/**` + `!data/**/` + `!data/**/.gitkeep` | **Se replica** | Trampa ya conocida; ver §1.2 |
| `ruff` (línea 99, `extend-select=["I"]`) | **Se replica** | Viene en la plantilla, coincide con el método |
| Las 4 reglas transversales de `prompts.docx` | **Se replican** | Ver §1.4 |
| "No eliminemos nada todavía: solo documentemos" | **Se replica** | Este proyecto es diagnóstico puro: el enunciado 4b pide *definir* la estrategia, no ejecutarla |
| Cada gráfica lleva su métrica en el título | **Se replica** | Hay 12+ figuras obligatorias |
| Convención de notebooks `<orden>-<iniciales>-<desc>` | **Se descarta** | El enunciado fija el nombre exacto `01_exploratory_data_analysis.ipynb`, y es un entregable evaluado |
| Pearson + η² contra el target | **Se sustituye** por ACF/PACF | No hay predictoras que correlacionar: solo IDs de alta cardinalidad y una fecha. El análogo estructural de "dependencia con el target" aquí son los rezagos (punto 9) |
| Regla `nunique <= 15` discretas/continuas | **Se descarta** el umbral | Carece de sentido con IDs de alta cardinalidad. Se conserva la exigencia de una regla escrita |
| "Faltantes que solo aparecen en test" | **Se traduce** | `test.csv` no tiene columnas de features. La pregunta equivalente —"¿qué trae test que train no tenía?"— es el *cold start* del punto 10b |
| `log1p` sobre el target | **Se descarta** | `item_cnt_day` tiene ceros y negativos (devoluciones): el logaritmo no está definido. §4.4 del método ya advierte que tampoco arregla la inflación de ceros |
| Winsorizar p1/p99 | **Se descarta** | §7 del método: "si las colas son la señal, recortarlas borra el problema". Aquí los extremos son el hallazgo, no ruido |
| Imputación por porcentaje (5 % / 20 %) | **No aplica** | Cero nulos en las 6 tablas. El problema es de *validez*, no de ausencia |
| KFold k=10, regla de 1 SE, torneo, MLflow, Docker, FastAPI | **Diferidos** | Fases 2–10, fuera de alcance |

### 1.2 Corrección al `.gitignore` de la plantilla

La plantilla genera `/data/`, que excluye el directorio entero e **impide re-incluir los
`.gitkeep`**: git no entra a un directorio excluido, así que una negación posterior nunca se
evalúa. Se sustituyó por `data/**` + `!data/**/` + `!data/**/.gitkeep`.

No es cosmético: `sales_train.csv` pesa **94.6 MB** y el enunciado prohíbe versionar los datos.
Se aplicó **antes del primer commit**, para que nunca entraran al índice.

### 1.3 `requirements.txt` como fuente de verdad

Versiones fijadas, actualizadas en el mismo paso en que entra cada librería. Añadidos sobre la
plantilla: **`statsmodels`** (ACF/PACF, punto 9) y **`seaborn`** (heatmaps, punto 8d).
`scikit-learn` queda fijado aunque el EDA no lo use, porque la plantilla lo incluye y la fase
siguiente lo necesitará.

### 1.4 Reglas transversales adoptadas

Tomadas de `prompts.docx`, repetidas en cada fase y no una sola vez al principio:

1. "Cualquier duda pregúntame, no asumas."
2. "Evitemos a toda costa sobreajustes y fuga de datos."
3. "No hagas nada más." — acota el entregable al enunciado.
4. "Usemos el artefacto ya generado." — continuidad entre fases.

### 1.5 Advertencia para la fase de modelado

**Cuando se llegue al torneo, la validación cruzada debe ser temporal (`TimeSeriesSplit`), sin
barajar.** El §7 del método lo señala explícitamente y en estos datos barajar sería fuga directa:
el objetivo es predecir el mes 34 a partir de los meses 0–33. La regla de 1 error estándar (§4.2)
sigue siendo válida, pero se aplica sobre pliegues temporales.

---

## 2. Evaluación de los datos crudos (previa al EDA)

Verificación hecha el 2026-11-12 antes de escribir el notebook, para dimensionar el trabajo.
**Son hallazgos preliminares: el notebook los reproduce con su evidencia.**

### 2.1 Inventario — 6 archivos, no 5

La descarga de Kaggle incluye `sample_submission.csv`, que el enunciado no menciona. Se conserva
y se documenta: confirma el nombre oficial del target (**`item_cnt_month`**) y las 214 200 filas
exactas a predecir.

| Archivo | Filas | Columnas | Peso |
|---|---|---|---|
| `sales_train.csv` | 2 935 849 | 6 | 94.6 MB |
| `test.csv` | 214 200 | 3 | 3.2 MB |
| `sample_submission.csv` | 214 200 | 2 | 2.2 MB |
| `items.csv` | 22 170 | 3 | 1.5 MB |
| `item_categories.csv` | 84 | 2 | 3.5 KB |
| `shops.csv` | 60 | 2 | 3 KB |

### 2.2 Calidad: mejor de lo esperado en estructura

- **Cero nulos** en las 6 tablas.
- **Integridad referencial perfecta**: ningún `item_id`, `shop_id` ni `item_category_id` huérfano.
- **`date_block_num` 100 % consistente** con la fecha: 0 discrepancias en 2.9 M filas.
- **No hay días sin ventas**: 1 034 días distintos entre 2013-01-01 y 2015-10-31, que son
  exactamente los 1 034 días del calendario. El hueco real no está en los días, sino en los
  pares tienda-producto sin movimiento.
- 34 meses (`date_block_num` 0–33); `date` llega como texto `dd.mm.aaaa`.

### 2.3 Problemas reales (valores inválidos, no ausentes)

- **1** registro con `item_price <= 0`.
- **7 356** registros con `item_cnt_day < 0` (0.25 %): devoluciones legítimas, se documentan.
- Ningún `item_cnt_day == 0`.
- Extremos: precio máximo **307 980**, cantidad máxima **2 169**.
- **6 duplicados exactos**.

### 2.4 Hallazgo — las tiendas duplicadas no se detectan por nombre exacto

El punto 4c del enunciado pide "mismo nombre con distinto ID". **Con comparación exacta el
resultado es cero**: hay que normalizar. Entonces aparecen 4 pares con ventanas de actividad
complementarias.

| Par | Nombres | Meses activos | ¿En test? |
|---|---|---|---|
| 0 / 57 | `!Якутск Орджоникидзе, 56 фран` / `Якутск Орджоникидзе, 56` | 0–1 / 2–33 | no / **sí** |
| 1 / 58 | `!Якутск ТЦ "Центральный" фран` / `Якутск ТЦ "Центральный"` | 0–1 / 2–33 | no / **sí** |
| 10 / 11 | `Жуковский ул. Чкалова 39м?` / `...39м²` | 0–33 / solo 25 | **sí** / no |
| 39 / 40 | `РостовНаДону ТРК "Мегацентр Горизонт"` / `... Островной` | 14–33 / 14–24 | **sí** / no |

En cada par una tienda deja de registrar justo cuando la otra empieza, y solo una sobrevive en
test. **Conclusión: `shop_id` no es una identidad estable de sucursal.** El par 10/11 es
literalmente un problema de codificación (`м?` contra `м²`). Comparar sucursales (punto 7b) sin
consolidar estos pares produce 8 series truncadas artificialmente.

### 2.5 Hallazgo — el 48 % de lo que hay que predecir no tiene historial

`test.csv` es una **rejilla completa**: 42 tiendas × 5 100 productos = 214 200 pares, no una
muestra de lo observado.

- Solo el **52.0 %** de los pares de test aparece alguna vez en train.
- **363 productos** de test no existen en train (*cold start* puro).
- Las 42 tiendas de test sí están todas en train; 18 tiendas de train no están en test.

Es el hallazgo que condiciona la fase de modelado entera y debe encabezar las conclusiones del
punto 10.

### 2.6 Dato externo de la competencia

Kaggle **recorta el target real al rango [0, 20]**. No se deduce de los datos; viene de las
reglas de la competencia. Relevante para el punto 10 y decisivo para el modelado.

---

## 3. Trampas nuevas (no cubiertas por el método de referencia)

| Síntoma | Causa | Qué hacer |
|---|---|---|
| `UnicodeEncodeError: 'charmap' codec` al imprimir nombres de tienda o producto | La consola de Windows usa cp1252 y los catálogos están en cirílico | Ejecutar con `python -X utf8`. **`PYTHONIOENCODING` no sirve**: el modo aislado (`-I`) ignora las variables de entorno |
| Etiquetas de tienda/categoría ilegibles en las figuras | Fuente por defecto de matplotlib sin glifos cirílicos | Pendiente: fijar una fuente con cobertura cirílica antes de los puntos 8b y 8c |
| La plantilla v2 de cookiecutter falla con el comando `cookiecutter` | v2 exige el comando `ccds` | `pip install cookiecutter-data-science==2.3.0` y usar `ccds` |
| `ccds --no-input` rechaza `dataset_storage` como cadena JSON | Compara contra `OrderedDict`, no contra la cadena | Omitir el parámetro: toma la primera opción, que es `none` |
| Graficar 2.9 M filas | Escala 2000× mayor que el proyecto de referencia | Agregar **antes** de graficar, siempre |
| `log1p` revienta la escala de color de un heatmap | Cuatro celdas (tienda, mes) suman −1 unidad y `log1p(-1) = -inf`; un solo `-inf` colapsa el `vmin`/`vmax` del mapa entero y todo sale del mismo color | Recortar a 0 **solo para graficar** (`m.clip(lower=0)`), nunca en los datos |
| Un heatmap de magnitudes muy dispares no muestra ningún patrón temporal | Tres categorías aplastan a las otras 22 | Normalizar **fila por fila** contra el máximo de cada fila cuando la pregunta es *cuándo*, no *cuánto* |
| La barra más alta se corta en el borde del eje | matplotlib no deja aire por encima de la barra máxima | `ax.margins(y=0.14)` |

Se mantiene la trampa heredada del método: ejecutar siempre con el entorno activado
(`conda run -n env_predict_future_sales --no-capture-output python ...`) por las DLLs de BLAS/MKL,
y escribir el código a un archivo en vez de pasarlo con `python -c` cuando lleva saltos de línea.

---

## 4. Decisiones de la Fase 1

### 4.1 Fuente para el cirílico

**DejaVu Sans**, que es la de matplotlib por defecto, cubre el cirílico y el superíndice `²`. No
hizo falta instalar nada. El notebook lo **verifica con un `assert`** en la celda de
configuración en vez de darlo por hecho: si algún día la fuente cambia, falla ruidosamente en
lugar de dibujar rectángulos vacíos en las etiquetas.

### 4.2 Criterio de selección de series para ACF/PACF

> Las **5 series con mayores ingresos acumulados**, entre las que tienen observaciones en los
> **34 meses**.

Ingresos y no unidades, porque las unidades premian artículos de acompañamiento —la bolsa de
marca es el producto más vendido del dataset con 187 642 unidades y 0.03 % de los ingresos—
cuya serie no representa demanda de catálogo. Historial completo, porque la ACF sobre una serie
con huecos calcula rezagos entre observaciones que no están igualmente espaciadas.

El filtro resultó durísimo en el nivel que importa: de **424 124** pares tienda-producto solo
**102** tienen los 34 meses (0.02 %). Eso es en sí un resultado del EDA.

### 4.3 Hallazgos nuevos que la Fase 0 no había visto

- **El objetivo mensual puede ser negativo.** Cuatro celdas (tienda, mes) suman −1 unidad. Cierra
  la puerta a cualquier transformación logarítmica, ya descartada por otra vía.
- **Solo el 13.4 % de los pares de test vendió algo en el mes 33.** El 52 % tiene historial
  *alguna vez*, pero el rezago 1 —la variable más informativa según la ACF— no existe para el
  86.6 % de las filas a predecir.
- **La estructura temporal se desvanece al desagregar.** La serie total tiene 3 rezagos
  significativos; las de par tienda-producto, una o ninguna. Implica **un modelo global**, no uno
  por serie.
- **La PACF mensual se corta en el rezago 1** (0.638 → 0.099): firma de un AR(1).
- **El rezago 12 no alcanza significancia** (+0.259 contra una banda de ±0.336) y no puede
  alcanzarla: con 34 observaciones solo hay dos ciclos anuales. La estacionalidad de diciembre
  está respaldada por la evidencia descriptiva, no por el correlograma.
- **Tendencia decreciente sostenida**: −2 300 unidades/mes. Un modelo que use el promedio
  histórico sin ponderar por recencia sobreestimará noviembre de 2015.

---

## 4.4 El error que casi se entrega sin detectar

Merece sección propia porque es el tipo de fallo contra el que todo el método está construido.

El notebook de inferencia pasó **sus siete verificaciones**: número de filas correcto, valores
finitos, rango [0, 20], identificadores únicos y alineados, formato oficial idéntico al de
Kaggle, y relectura del archivo escrito. Ninguna falló. El archivo de envío era perfectamente
válido.

Y las predicciones estaban mal: media **0.0066** cuando el objetivo real promedia **0.2982**.
Cincuenta veces menos.

**La causa.** `UnirCatalogos` añade `date_block_num` al final cuando falta, que es el caso de
`test.csv`; al entrenar viene al principio. Los nombres de columna eran correctos en ambos casos,
pero como los operadores propios declaran sus nombres de salida, scikit-learn los reasigna **por
posición**: cogía la lista aprendida en el ajuste y la pegaba sobre columnas que llegaban en otro
orden. En inferencia, la columna llamada `shop_id` contenía identificadores de producto y la
llamada `item_id` contenía el número de mes.

**Lo que lo detectó** no fue ninguna de las siete comprobaciones, sino comparar la **magnitud**
de lo predicho contra la magnitud de lo real. Las verificaciones comprobaban que las predicciones
fueran *posibles*; ninguna comprobaba que fueran *sensatas*.

**Lo que se añadió:** `test_el_orden_de_columnas_no_cambia_el_resultado`, que transforma las
mismas filas con las columnas revueltas y sin `date_block_num`, y exige resultados idénticos.
Es la prueba que `TRANSFERENCIA_METODO.md` §3 ya pedía —*"una prueba de que el orden de las
columnas no cambia la predicción"*— y que no se había escrito.

---

## 5. Decisiones del Proyecto #2

### 5.0 Resultado

| | |
|---|---|
| Modelo ganador | `HistGradientBoostingRegressor` (`max_iter=200`, `learning_rate=0.05`, `max_leaf_nodes=31`) |
| RMSE en validación (meses 27-33) | **0.7595** |
| MAE | 0.2639 |
| Mejora sobre repetir el mes anterior (1.0373) | **26.8 %** |
| Predicción media del mes 34 | 0.2039 (el objetivo real del mes 33 promedia 0.2586) |
| Criterio de desempate | Regla del error estándar; entre los empatados, 6 200 números ajustados frente a 9 450 |


### 5.1 El operador `Mapper` se escribió desde cero

El enunciado permite usar *"el operador Mapper definido durante las sesiones de clase"*. Se buscó
en todo el disco —`Clase_Cloud/`, el árbol completo de `Maestria/`, `Downloads/`, `Documents/`,
y también `DataFrameMapper` de `sklearn_pandas`— y **no existe en esta máquina**. Las notas de la
sesión del 12/11 son capturas de la clase de Docker, sin código.

Se implementó uno propio siguiendo la convención de los operadores del curso que sí están
disponibles (`Downloads/Clase4-SL1.py`: `OutlierCapper`, `AgeBinning`, `DatetimeFeatures`).
Hace dos cosas que `OrdinalEncoder` no puede hacer solo: **extrae** la ciudad de dentro del nombre
de la tienda y **marca como nulo** cualquier valor no visto en el entrenamiento, en vez de
inventarle un código.

El `OutlierCapper` del material de clase se reutilizó con dos correcciones: devuelve un DataFrame
en vez de un `ndarray` (el original perdía los nombres de columna y rompía el encadenado) y usa
percentiles configurables en lugar del rango intercuartílico fijo, que sobre una distribución con
el 85 % de ceros recortaría datos legítimos.

### 5.2 Todo el código de los operadores vive en el paquete

`predict_future_sales/features.py`, no en los notebooks. Un `.pkl` no guarda el código de las
clases, solo dónde encontrarlas; si vivieran en el notebook, el artefacto fallaría con
`ModuleNotFoundError` al abrirlo desde cualquier otro sitio, y los entregables 2 y 3 quedarían
inservibles.

### 5.3 La partición es temporal, fijada por el enunciado

Meses **0–26** entrenan, **27–33** validan. Corrige la nota de §1.5, que proponía 0–32 / 33.
Barajar sería fuga directa.

### 5.4 El objetivo se recorta a [0, 20]

No es una decisión de modelado: es la regla publicada de la competencia, que evalúa contra valores
reales ya recortados. Afecta a 12 378 filas de 10.9 millones (0.11 %).

### 5.5 Submuestra documentada para dos familias

Con 8.4 GB de RAM y 4 núcleos, el bosque aleatorio y la red neuronal no terminan en tiempo
razonable sobre 9.3 millones de filas. Se entrenan con una submuestra aleatoria de 1 millón, y
**la tabla de resultados declara con cuántas filas se ajustó cada modelo**, porque comparar un
modelo entrenado con 1 M contra otro con 9.3 M no es una comparación en igualdad de condiciones.

### 5.6 Dos artefactos, uno de evidencia y otro de producción

El modelo del torneo se ajustó con los meses 0–26 y es el que produjo el RMSE reportado. El
`pipeline_completo.pkl` se reentrena con los 34 meses —los más recientes son los más valiosos con
una tendencia decreciente— **heredando los hiperparámetros**, sin volver a buscarlos: buscarlos
sobre los datos completos usaría las filas de validación para elegir.

---

## 6. Trampas nuevas del Proyecto #2

| Síntoma | Causa | Qué hacer |
|---|---|---|
| El `.pkl` funciona en el notebook y falla al cargarlo fuera con `KeyError` en una columna | `sklearn.set_config(transform_output="pandas")` es una opción **global del proceso** y no se guarda dentro del artefacto | `pipeline.set_output(transform="pandas")`, que sí se almacena en el objeto |
| Un paso propio tarda 160 s sobre 11 M de filas | Construir un `PeriodIndex` fila a fila con una comprensión de listas | Aritmética vectorizada de numpy: 160 s → 0.8 s |
| Otro paso propio tarda 117 s | Aplicar una función de Python a 11 M de cadenas | Aplicarla a los **valores únicos** (60 nombres) y repartir el resultado: 117 s → 22 s |
| El `ColumnTransformer` renombra todo a `tmp_col_name_0_1` | Los operadores propios no declaraban `get_feature_names_out` | Implementarlo; con cuidado, porque `transform` lo invoca y se forma una dependencia circular que hay que romper con un centinela |
| Dos medias móviles distintas salen idénticas | La ventana de 6 meses se calculaba con los rezagos disponibles (1, 2 y 3), los mismos que la de 3 | Calcular los rezagos auxiliares 4, 5 y 6 y descartarlos después |
| El DataFrame de características ocupa 3.35 GB | Dos columnas de texto en cirílico sobre 11 M de filas | Retirarlas en cuanto el `Mapper` las ha usado: 3.35 GB → 0.64 GB |
| `SGDRegressor` devuelve un RMSE de 18.5 | No converge con los parámetros por defecto sobre estos datos | Se descartó esa familia y se documentó, en lugar de dejarla falseando el torneo |
| **El modelo predice casi cero para todo** (media 0.0066 contra un objetivo real de 0.2982) | **Las columnas iban desplazadas.** `UnirCatalogos` añade `date_block_num` al final cuando falta —el caso de `test.csv`—, mientras que al entrenar viene al principio. Como el operador declara sus nombres de salida, scikit-learn los reasigna **por posición**: la columna llamada `shop_id` contenía identificadores de producto | Reordenar siempre a un orden canónico, y vigilarlo con `test_el_orden_de_columnas_no_cambia_el_resultado` |
| Una celda del torneo pasa de 45 minutos a más de 2 horas, con el mismo código | La máquina paginando: 0.78 GB de RAM libre y 2.7 GB ya volcados a disco | Cerrar aplicaciones antes de entrenar, y reducir el pico de memoria (ver la fila siguiente) |
| El pico de memoria dobla lo necesario al transformar | `pipe.transform(X)` devuelve un DataFrame de 1.08 GB y `.to_numpy()` crea una segunda copia antes de liberar la primera | Transformar por bloques escribiendo sobre una matriz ya reservada: pico de 2.2 a 1.2 GB |

---

## 5. Pendientes

- [x] Ejecutar `notebooks/01_exploratory_data_analysis.ipynb` completo.
- [x] Fijar fuente con cobertura cirílica para matplotlib.
- [x] Definir y escribir el criterio de selección de series representativas para ACF/PACF.
- [ ] Screenshot del `git pull` para el entregable 3.
- [ ] Fase 2 (limpieza), si se amplía el alcance: ver la lista de reservas al final del notebook.
