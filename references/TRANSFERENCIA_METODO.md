# Método transferible — proyecto `house_pricing`

Documento de transferencia. No describe este proyecto para quien lo audite, sino **el método
que se usó**, para aplicarlo a un dataset distinto. Está escrito para que un asistente lo lea
al inicio de un proyecto nuevo y decida, con el usuario, qué se replica y qué no.

- **Proyecto de origen:** House Prices / Ames Iowa (Kaggle), regresión tabular, 1460 × 81.
- **Resultado:** ElasticNet (`alpha=0.001`, `l1_ratio=0.5`), 20 042 USD de RMSE real
  (4.01 % del rango, 0.1401 de RMSE logarítmico) sobre 292 casas reservadas.
- **Alcance cubierto:** de datos crudos a modelo registrado, con script de inferencia,
  pruebas, documentación, repositorio, MLflow y contenedor. Lo único no hecho es el servicio
  de FastAPI.
- **Repositorio:** <https://github.com/dannygarcia-hue/house_pricing>
- **Detalle narrativo:** `ESTADO_PROYECTO.md` (continuidad), `reports/proceso.docx`
  (18 páginas, qué se hizo), `reports/prompts.docx` (12 páginas, prompts para replicarlo).

---

## 1. Cómo usar este documento en el proyecto nuevo

Antes de aplicar nada, el asistente debe responder tres preguntas con el usuario:

1. **¿Es el mismo tipo de problema?** El método está diseñado para **regresión o clasificación
   tabular supervisada, de tamaño pequeño o medio** (miles de filas, decenas o cientos de
   columnas), con un archivo de entrenamiento y uno de prueba. Series de tiempo, texto, imagen,
   datos agrupados o millones de filas cambian decisiones de fondo (ver §7).
2. **¿Cuál es el entregable?** Un notebook de clase, un modelo registrado, un servicio, o todo.
   Determina cuántas fases de §3 se ejecutan.
3. **¿Qué es inaceptable?** En este proyecto fueron la fuga de datos y el sobreajuste, y eso
   definió la arquitectura entera. Si la restricción dominante es otra (latencia,
   interpretabilidad exigida por regulación, costo de inferencia), varias decisiones cambian.

**Secciones por prioridad de lectura:** §4 (reglas reutilizables) y §6 (trampas) son lo que más
tiempo ahorra. §7 es lo que evita copiar decisiones que no aplican.

---

## 2. Orden de las fases

Cada fase consume el artefacto de la anterior y no se adelanta. El orden importa: la fuga de
datos se cuela cuando una fase usa información que la siguiente debería desconocer.

| Fase | Entrada | Salida | Entregable |
|---|---|---|---|
| 0 · Estructura y entorno | — | repo con estructura y `requirements.txt` | — |
| 1 · Exploración | `data/raw/` | entendimiento, lista de problemas | notebook 1.0 |
| 2 · Limpieza | `data/raw/` | `data/interim/*_limpio.csv` | notebook 2.0 |
| 3 · Features | `data/interim/` | `models/pipeline_features.joblib` | notebook 3.0 |
| 4 · Torneo de modelos | `data/raw/` + pipeline | `models/*.joblib`, tabla de métricas | notebook 4.0 |
| 5 · Inferencia | `data/raw/test` + modelo | predicciones | notebook 5.0 |
| 6 · Puesta a disposición | modelo | biblioteca + CLI + pruebas | paquete y `scripts/` |
| 7 · Documentación | todo | `proceso.docx`, `prompts.docx` | `reports/` |
| 8 · Control de versiones | todo | repo remoto, ramas, PR | GitHub |
| 9 · Seguimiento (MLflow) | torneo | registro con alias champion/challenger | `mlflow.db` |
| 10 · Reproducibilidad | `requirements.txt` | imagen y `docker-compose.yml` | `DOCKER.md` |

La fase 4 **vuelve a `data/raw`**, no continúa desde `data/processed`. Es la decisión más
importante del proyecto y está explicada en §4.1.

---

## 3. Qué se hizo en cada fase

### Fase 0 — Estructura y entorno

- Estructura de **Cookiecutter Data Science v2.3.0**: `data/{raw,interim,processed}`,
  `notebooks/`, `<paquete>/`, `scripts/`, `tests/`, `reports/`, `models/`.
- Entorno conda dedicado con Python fijado; `requirements.txt` con **versiones fijadas** como
  única fuente de verdad.
- `ruff` configurado en `pyproject.toml` (línea de 99, `extend-select = ["I"]`) cubriendo el
  paquete, `scripts/` y `tests/`.
- Convención de notebooks: `<orden>-<iniciales>-<descripción>`.

### Fase 1 — Exploración

Vistazo general; separación de numéricas en **discretas y continuas** con una regla explícita
(`nunique <= 15`); mapa de faltantes distinguiendo las columnas que faltan **solo en test**;
correlación de Pearson para numéricas y **η² para categóricas** (η² = SS_entre / SS_total: la
proporción de varianza del target explicada por la variable; su raíz vive en la escala de |r|);
matriz de redundancia entre predictores; detección de outliers con inspección caso por caso, no
por regla automática.

### Fase 2 — Limpieza

- **Faltantes estructurales vs. reales.** Un `NA` que significa *no existe la característica*
  se rellena con `"None"` o `0`; uno que significa *dato perdido* se imputa con estadísticos de
  train. Clasificar columna por columna, con el diccionario de datos a la vista.
- **Outliers con criterio, no con umbral.** Se eliminaron 2 de 4 candidatos: los que tenían una
  explicación que invalidaba el registro (venta de obra inconclusa), no los que simplemente eran
  grandes y caros.
- Codificación de ordinales con escalas escritas a mano; descarte de nominales casi constantes
  (> 96 % de dominancia).
- Verificación de salida: 0 faltantes, formas finales declaradas.

### Fase 3 — Ingeniería de características

El entregable es **un objeto `Pipeline` serializado**, no un CSV. Esto es lo que hace posible la
fase 4 sin fuga. 13 pasos de `feature-engine` y `scikit-learn` en cinco etapas: imputación →
codificación → recorte de outliers (`Winsorizer` p1/p99) → transformación de sesgo → escalado.
30 features crudas → 122 columnas.

Salidas: el `.joblib` del pipeline, los CSV transformados (solo para inspección) y la lista de
features en un `.yaml` versionado.

### Fase 4 — Torneo de modelos

- **Validación cruzada KFold k=10** con búsqueda de hiperparámetros, mínimo 3 configuraciones
  por modelo. En este caso: 8 modelos × 66 configuraciones × 10 pliegues = **660 ajustes del
  pipeline completo**, unos 7 minutos.
- **Al menos 5 familias distintas**: lineales regularizadas (Ridge, Lasso, ElasticNet), lineal
  sin regularizar, ensambles de árboles (Random Forest, Gradient Boosting, Hist GB) y red
  neuronal (MLP).
- **Tres métricas por modelo**: en la escala transformada (comparable con la competencia), en
  unidades reales (interpretable) y **normalizada al rango** (comparable entre datasets).
- Conjunto reservado que ningún ajuste vio, además de la CV.
- **Brecha de sobreajuste medida** (error dentro del entrenamiento vs. validación) como columna
  de la tabla, no como comentario.
- Gráficas por modelo: comparativa de barras, dispersión de pliegues, predicho vs. real,
  residuos, curva de aprendizaje del ganador, importancia por permutación.
- Dos artefactos de modelo (§4.3) y un `.csv` con la tabla del torneo.

### Fase 5 — Notebook de inferencia

De datos **crudos** a predicciones, sin un solo `fit`. Sirve de prueba de que el artefacto es
autocontenido: si el notebook de inferencia necesita repetir lógica del notebook 2 o 3, el
pipeline está incompleto.

### Fase 6 — Puesta a disposición

- **La lógica vive en el paquete** (`<paquete>/modeling/predict.py`), no en el script.
- Firmas tipadas en parámetros y retornos, docstrings, PEP 8.
- Excepciones propias que mapean a códigos HTTP: `EntradaInvalida` → 422, `SalidaIncoherente`
  → 500, `ModeloNoDisponible` → 503.
- **Mecanismo de calidad entre entrada y salida**: cuenta filas que producen resultado coherente
  (rango plausible, sin nulos ni infinitos), con modo `estricto` que aborta y modo tolerante que
  reporta.
- Logging a archivo con estampa de tiempo por ejecución.
- `lru_cache` en la carga del modelo: deserializar tardaba ~4 s, predecir 1459 filas ~0.4 s.
- CLI con `typer` (`demo`, `archivo`, `ficha`) y un mecanismo de prueba con observaciones
  aleatorias del test.
- **29 pruebas** con `pytest`: contrato de entrada, camino feliz, verificación de calidad,
  envoltorio de API, muestreo.
- Criterio de aceptación: el script reproduce la salida del notebook **byte a byte**.

### Fases 7 a 10 — Entrega y operación

- Dos `.docx` generados con `python-docx`: uno narrativo del proceso, uno con los prompts para
  replicarlo.
- Repositorio git con `.gitignore` que excluye `data/`, `models/`, logs y artefactos de MLflow;
  rama por feature y PR descrito para revisión de SRE.
- **MLflow** con backend SQLite (el Model Registry no funciona sobre el almacén de archivos) y
  los tres mejores modelos publicados con alias `champion` / `challenger_1` / `challenger_2`,
  todos ajustados al 100 % para que promover no exija reentrenar. `code_paths` para que el
  artefacto lleve el paquete dentro y cargue fuera del proyecto.
- **Docker** con el repositorio montado como volumen: el contenedor es el intérprete, no el
  lugar donde vive el trabajo.

---

## 4. Reglas reutilizables

Estas son las decisiones que valen más que el código y que conviene replicar.

### 4.1 Cómo se evita la fuga de datos en la validación cruzada

El error típico es ajustar el preprocesamiento sobre todos los datos, guardar el CSV
transformado y hacer CV sobre él. Entonces el escalado, los percentiles de recorte y las
medianas de imputación **ya vieron los pliegues de validación**, y el error de CV sale
optimista.

La solución aplicada: el pipeline de features se **clona sin ajustar** y se pone como primer
paso de un `Pipeline` que termina en el estimador, de modo que las transformaciones se
re-ajustan dentro de cada pliegue.

```python
from sklearn.base import clone
from sklearn.pipeline import Pipeline

def construir(estimador):
    """Envuelve un estimador con un clon sin ajustar del pipeline de características."""
    return Pipeline([("features", clone(pipeline_features)), ("modelo", estimador)])
```

El costo es tiempo de cómputo (el preprocesamiento se repite en cada pliegue). Es el precio
correcto a pagar. Como efecto secundario, el objeto que gana el torneo ya sabe transformar datos
crudos, que es exactamente lo que necesita la inferencia.

**Señal de alarma:** si el error de CV sale sospechosamente bajo, el primer sospechoso es esto.

### 4.2 Cómo se elige el ganador: regla de 1 error estándar

No se elige el mejor promedio de CV. Se calcula SE = σ_pliegues / √k del mejor modelo, se
declaran **empatados** a todos los que caen dentro de un SE, y entre ellos gana el de **menor
brecha de sobreajuste**.

```python
error_estandar = torneo.loc[MEJOR_CV, "rmse_log_cv_sd"] / np.sqrt(K_PLIEGUES)
umbral = torneo.loc[MEJOR_CV, "rmse_log_cv"] + error_estandar
empatados = tabla[tabla["rmse_log_cv"] <= umbral].copy()
GANADOR = empatados["brecha"].idxmin()
```

En este proyecto Gradient Boosting ganó el promedio por 0.0002 —el **1 % de su propia
desviación entre pliegues**— y a cambio sobreajustaba 5.7 veces más y perdía en el conjunto
reservado. La regla coronó al ElasticNet.

Dos propiedades que la hacen valiosa: "más simple" queda definido por una **cantidad medida** y
comparable entre familias, y la decisión está **escrita como código**, así que si los números
cambian la decisión cambia sola, sin reinterpretación.

### 4.3 Dos artefactos de modelo, no uno

- `modelo_ganador.joblib`, ajustado con el 80 % de train, es **evidencia**: es el objeto exacto
  que produjo la métrica reportada, y mientras esa cifra exista es verificable.
- `modelo_produccion.joblib`, ajustado con el 100 %, es la **herramienta**.

Los hiperparámetros del segundo **se heredan** del torneo. Volver a buscarlos sobre los datos
completos usaría las filas de validación para elegir.

### 4.4 El sesgo se trata según su causa, no según su magnitud

Logaritmar todo lo sesgado es un reflejo que destruye información. Verificado sobre estos datos:

| Variable | % ceros | r original | r con log-log | Sesgo original | Sesgo tras log |
|---|---|---|---|---|---|
| `TotalBsmtSF` | 2.5 % | 0.614 | **0.373** | 1.52 | **−5.15** |
| `GarageArea` | 5.5 % | 0.623 | **0.455** | 0.18 | **−3.48** |

Mecanismo: los ceros caen en `log(1) = 0` y el resto cerca de `log(1000) ≈ 6.9`, abriendo un
abismo artificial entre "sin sótano" y "con sótano". Y **ninguna transformación monótona elimina
un pico de masa en un solo valor**: el logaritmo no arregla la inflación de ceros.

Criterio aplicado: `log1p` solo donde hay asimetría genuina sin ceros; Yeo-Johnson donde el
logaritmo no aplica; **los años no se transforman**; el resto solo recorte y escalado. El
**target sí** se transforma si eso corrige su sesgo y coincide con la métrica oficial —con
`log1p`/`expm1`, y las métricas en unidades reales se calculan después de invertir.

### 4.5 Un valor ausente no es un valor perdido

Si `NA` significa "no tiene esa característica", imputar con la moda **inventa** la
característica. Separar faltantes estructurales de reales antes de imputar nada, y añadir un
indicador binario cuando la ausencia misma informa. Caso concreto: el año de construcción del
garaje no se rellena con 0 (inventaría casas del año cero) sino con el año de la casa, más un
indicador `SinGaraje`.

### 4.6 La biblioteca es el contrato, el script es un adaptador

El script de consola solo traduce argumentos a llamadas de la biblioteca. Cualquier servicio
posterior importa **las mismas funciones**, no una copia que se desincronice. La función
pensada para la API recibe y devuelve exactamente lo que produce el deserializador de JSON:
`predecir_registros(registros: list[dict]) -> list[dict]`.

---

## 5. Prompts que funcionaron

El patrón de instrucción que produjo los mejores resultados, al inicio de cada fase:

1. **Restricción explícita de lo inaceptable**: *"evitemos a toda costa sobreajustes y fuga de
   datos en el flujo de construcción de modelos"*. Esto es lo que motivó la arquitectura de
   §4.1; sin la frase, el camino cómodo era leer el CSV ya transformado.
2. **Cierre de alcance**: *"no hagas nada más"*. Evita entregables no pedidos.
3. **Prohibición de suponer**: *"cualquier duda preguntame, no asumas"*. Convierte las
   decisiones implícitas en preguntas.
4. **Requisitos numerados y verificables**: "k=10", "al menos 3 configuraciones", "al menos 5
   modelos distintos", "RMSE real y normalizado al rango".

`reports/prompts.docx` tiene la secuencia completa, fase por fase, lista para adaptar.

---

## 6. Trampas que ya costaron tiempo

| Síntoma | Causa real | Qué hacer |
|---|---|---|
| El kernel muere sin excepción de Python | DLLs de BLAS/MKL fuera del PATH al invocar el `python.exe` del entorno sin activarlo | Ejecutar siempre con el entorno activado: `conda run -n <env> --no-capture-output python ...` |
| Error de CV sospechosamente bajo | Se leyó el CSV ya transformado | §4.1 |
| Primer punto de la curva de aprendizaje vacío (`NaN`) | Un imputador por moda aborta con categorías empatadas en submuestras pequeñas; `learning_curve` convierte el fallo en `NaN` **silencioso** | `error_score="raise"` siempre; retirar el paso del clon de la curva solo tras probar con `assert` que es no-op |
| El óptimo de un hiperparámetro cae en el borde de la rejilla | Rejilla demasiado corta | Comprobarlo automáticamente y ampliar la rejilla |
| `conda run python -c` falla con `NotImplementedError` | No admite argumentos con saltos de línea | Escribir el código a un archivo primero |
| `.gitkeep` no se puede re-incluir | Una regla que excluye el directorio entero (`/data/`) impide re-incluir archivos dentro | Usar `data/**` + `!data/**/` + `!data/**/.gitkeep` |
| MLflow rechaza serializar el modelo | El formato por defecto (skops) no admite transformadores de terceros ni clases propias | `serialization_format="cloudpickle"`, asumiendo y documentando que ejecuta código arbitrario al deserializar |
| `ModuleNotFoundError` del propio paquete al cargar el modelo | `cloudpickle` guarda las clases propias **por referencia**; los notebooks lo ocultan porque insertan la raíz en `sys.path` | `code_paths=[...]` al registrar, y probar la carga desde un proceso que arranque en otro directorio |
| Tablas de `.docx` que parten filas entre páginas | Falta de `w:tblHeader` y `w:cantSplit` | Añadirlos al estilo de tabla |

Regla general que se repitió: **verificar mirando el resultado**, no asumiendo que el código
correcto produce la figura correcta. Varios defectos de las 9 figuras (leyendas tapando barras,
etiquetas colisionando, una línea fuera del lienzo) solo aparecieron al renderizarlas y verlas.

---

## 7. Qué es específico de este dataset y no debe copiarse a ciegas

| Decisión de este proyecto | Por qué fue válida aquí | Cuándo cambia |
|---|---|---|
| `log1p` en el target | Sesgo 1.88 → 0.12 y es la métrica oficial de Kaggle | Target simétrico, con ceros o negativos, o métrica de negocio en unidades reales |
| Ganan las lineales regularizadas | 1166 filas de entrenamiento y 122 columnas: los ensambles no tienen material para su ventaja habitual | Con decenas de miles de filas, el boosting normalmente gana con holgura |
| `nunique <= 15` para separar discretas de continuas | Umbral inspeccionado a mano sobre estas 36 columnas | Depende del dominio; revisar la lista resultante |
| KFold simple | Regresión, filas independientes, sin agrupamiento | Clasificación desbalanceada → `StratifiedKFold`; grupos → `GroupKFold`; tiempo → `TimeSeriesSplit` y **nada de barajar** |
| Winsorizar p1/p99 | Áreas y precios con colas largas pero reales | Si las colas son la señal (fraude, fallos, eventos raros), recortarlas borra el problema |
| Imputar con medianas y modas de train | Faltantes mayormente estructurales y bien entendidos | Faltantes no aleatorios o masivos piden modelado explícito del mecanismo |
| 660 ajustes del pipeline completo | 7 minutos: aceptable | Con datos grandes esto se vuelve inviable; hay que reducir pliegues o rejilla, no volver al CSV pre-transformado |
| 2 de 4 outliers eliminados | Había una explicación documentada que invalidaba esos registros | Nunca se traslada: se decide caso por caso sobre los datos nuevos |
| SQLite como backend de MLflow | Un solo usuario, un solo escritor | Equipo o servicio: servidor de MLflow con Postgres y almacén de objetos |

Las 30 features seleccionadas, las escalas ordinales escritas a mano y los rangos plausibles de
precio (10 000 a 1 500 000 USD) son **íntegramente** de este dominio y no significan nada en
otro.

---

## 8. Lo que el asistente necesita saber del proyecto nuevo

Antes de proponer un plan, pedir:

1. **Los datos**, o una muestra y el diccionario de variables.
2. **El target y la métrica** con la que se evalúa, y si la métrica la fija un tercero.
3. **El tipo de problema**: regresión, clasificación, multiclase, series de tiempo.
4. **Tamaño y forma**: filas, columnas, si hay agrupamiento o estructura temporal.
5. **El entregable y el plazo**: hasta dónde llega el alcance de las 10 fases.
6. **La restricción dominante**: ¿fuga de datos? ¿latencia? ¿interpretabilidad exigida?
7. **El entorno**: conda, Docker, nube, y qué hay instalado en la máquina.

---

## 9. Lista de verificación

- [ ] Estructura del proyecto y entorno con versiones fijadas, antes de escribir código.
- [ ] Separación raw / interim / processed, con `raw` inmutable.
- [ ] Faltantes clasificados en estructurales y reales **antes** de imputar.
- [ ] Outliers decididos uno por uno, con la razón escrita.
- [ ] Transformaciones de sesgo justificadas por causa, con el efecto medido sobre `r` y el sesgo.
- [ ] El preprocesamiento es un `Pipeline` serializado, no un CSV.
- [ ] El pipeline se clona y se re-ajusta dentro de cada pliegue de la CV.
- [ ] `error_score="raise"` en toda validación cruzada y curva de aprendizaje.
- [ ] Al menos 5 familias de modelos y 3 configuraciones por modelo.
- [ ] Métricas en la escala transformada, en unidades reales y normalizada al rango.
- [ ] Brecha de sobreajuste como columna de la tabla.
- [ ] Ganador elegido con la regla de 1 SE, implementada como código.
- [ ] Óptimos de hiperparámetros verificados fuera del borde de la rejilla.
- [ ] Figuras **renderizadas y miradas**, no solo generadas.
- [ ] Dos artefactos: evidencia (parcial) y herramienta (completo).
- [ ] La inferencia parte de datos crudos y no hace ningún `fit`.
- [ ] Lógica en el paquete, scripts como adaptadores, pruebas sobre la biblioteca.
- [ ] Verificación de equivalencia entre el camino del notebook y el del script.
- [ ] Logging con estampa de tiempo y conteo de filas coherentes entre entrada y salida.
- [ ] `requirements.txt` actualizado en el mismo paso en que entra cada librería.
- [ ] Datos, modelos, logs y bases de seguimiento fuera del control de versiones.
- [ ] Documento de continuidad actualizado al cerrar cada fase.
