# Estado del proyecto — `predict_future_sales`

Documento de continuidad. Registra **las decisiones y su justificación**, no solo lo que se
hizo. Se actualiza al cerrar cada fase.

- **Proyecto:** Proyecto #1 — Exploratory Data Analysis (EDA)
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
| 1 · EDA (`01_exploratory_data_analysis.ipynb`) | Pendiente de ejecución | — |
| 2–10 · Limpieza en adelante | Fuera del alcance del Proyecto #1 | — |

**Alcance acordado:** solo el Proyecto #1. Las fases 2 a 10 del método de referencia no se
ejecutan ahora, pero la estructura y el `requirements.txt` quedan preparados para no rehacerlas.

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

Se mantiene la trampa heredada del método: ejecutar siempre con el entorno activado
(`conda run -n env_predict_future_sales --no-capture-output python ...`) por las DLLs de BLAS/MKL,
y escribir el código a un archivo en vez de pasarlo con `python -c` cuando lleva saltos de línea.

---

## 4. Pendientes

- [ ] Ejecutar `notebooks/01_exploratory_data_analysis.ipynb` completo.
- [ ] Fijar fuente con cobertura cirílica para matplotlib.
- [ ] Definir y escribir el criterio de selección de series representativas para ACF/PACF
      (60 tiendas, 21 807 productos: no se grafican todas).
- [ ] Screenshot del `git pull` para el entregable 3.
