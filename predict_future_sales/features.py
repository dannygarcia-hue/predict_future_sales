"""Operadores de ingenieria de caracteristicas para Predict Future Sales.

Por que este codigo vive aqui y no dentro de un notebook
--------------------------------------------------------
El pipeline se guarda en un archivo `.pkl`. Al guardarlo, Python NO copia el
codigo de las clases: solo anota donde encontrarlas (por ejemplo
"predict_future_sales.features.LagFeatures"). Si las clases estuvieran escritas
dentro del notebook, al abrir el `.pkl` desde otro proceso Python no sabria
donde buscarlas y fallaria con `ModuleNotFoundError`.

Por eso viven en el paquete: asi el `.pkl` se puede abrir desde cualquier sitio,
que es justo lo que piden los entregables 2 y 3 del proyecto.

Convencion
----------
Todos los operadores siguen el mismo patron de scikit-learn, el mismo que se uso
en clase:

- heredan de `BaseEstimator` y `TransformerMixin`;
- `fit(X, y)` APRENDE de los datos de entrenamiento y guarda lo aprendido en
  atributos que terminan en guion bajo (`self.algo_`);
- `transform(X)` APLICA lo aprendido, sin volver a aprender nada.

Esa separacion es lo que evita la fuga de informacion: lo aprendido sale solo
del entrenamiento, nunca de los datos que se quieren predecir.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Primer mes del dataset: date_block_num = 0 corresponde a enero de 2013.
ANIO_INICIAL = 2013
MES_INICIAL = 1


# ===========================================================================
# Utilidades de panel
# ===========================================================================
def mes_a_periodo(date_block_num: np.ndarray | pd.Series) -> pd.PeriodIndex:
    """Convierte el numero de mes consecutivo (0, 1, 2...) en un periodo real.

    date_block_num = 0 -> enero 2013,  33 -> octubre 2015,  34 -> noviembre 2015.
    """
    meses = np.asarray(date_block_num, dtype=int)
    return pd.PeriodIndex(
        [pd.Period(year=ANIO_INICIAL + m // 12, month=m % 12 + 1, freq="M") for m in meses],
        freq="M",
    )


def construir_rejilla(ventas: pd.DataFrame, meses: list[int] | None = None) -> pd.DataFrame:
    """Construye la rejilla mensual completa, rellenando con cero los meses sin venta.

    El problema:
        `sales_train.csv` solo tiene filas donde HUBO una venta. Si entrenamos
        solo con eso, el modelo nunca ve un cero y aprende que todo se vende.
        Pero el 85 % de las combinaciones posibles (mes, tienda, producto) son
        cero, y casi la mitad de lo que hay que predecir no tiene historial.

    La solucion:
        Para cada mes, generar TODAS las combinaciones de las tiendas y los
        productos que estuvieron activos ESE mes, y poner cero donde no hubo
        venta. No se usa el producto cartesiano global (60 x 21807 x 34 = 44
        millones) porque incluiria tiendas cerradas y productos descatalogados.

    Parametros
    ----------
    ventas : DataFrame con date_block_num, shop_id, item_id, item_cnt_day.
    meses  : lista de meses a incluir; None = todos.

    Devuelve
    --------
    DataFrame con date_block_num, shop_id, item_id, item_cnt_month.
    """
    if meses is not None:
        ventas = ventas[ventas["date_block_num"].isin(meses)]

    bloques = []
    for mes, grupo in ventas.groupby("date_block_num", observed=True):
        tiendas = grupo["shop_id"].unique()
        productos = grupo["item_id"].unique()
        # np.repeat/np.tile arma el producto cartesiano sin bucles de Python.
        bloques.append(
            pd.DataFrame({
                "date_block_num": np.full(len(tiendas) * len(productos), mes, dtype="int8"),
                "shop_id": np.repeat(tiendas, len(productos)).astype("int8"),
                "item_id": np.tile(productos, len(tiendas)).astype("int16"),
            })
        )
    rejilla = pd.concat(bloques, ignore_index=True)

    # Las ventas reales se suman por mes y se pegan sobre la rejilla.
    reales = (
        ventas.groupby(["date_block_num", "shop_id", "item_id"], observed=True)["item_cnt_day"]
        .sum()
        .rename("item_cnt_month")
        .reset_index()
    )
    rejilla = rejilla.merge(reales, on=["date_block_num", "shop_id", "item_id"], how="left")
    # Lo que no aparece en las ventas reales vendio exactamente cero unidades.
    rejilla["item_cnt_month"] = rejilla["item_cnt_month"].fillna(0).astype("float32")
    return rejilla


def panel_mensual(ventas: pd.DataFrame) -> pd.DataFrame:
    """Ventas mensuales observadas por par tienda-producto (sin los ceros).

    Es el historial que `LagFeatures` guarda dentro del artefacto para poder
    calcular los rezagos del mes que se predice.
    """
    return (
        ventas.groupby(["date_block_num", "shop_id", "item_id"], observed=True)["item_cnt_day"]
        .sum()
        .rename("item_cnt_month")
        .reset_index()
        .astype({
            "date_block_num": "int8",
            "shop_id": "int8",
            "item_id": "int16",
            "item_cnt_month": "float32",
        })
    )


class _NombresDeSalida:
    """Permite que scikit-learn conozca los nombres de las columnas resultantes.

    Sin esto, un `ColumnTransformer` que reciba datos de un operador propio no
    sabe como se llaman las columnas y las renombra a `tmp_col_name_0_1`, lo que
    deja ilegible cualquier analisis de importancia de variables mas adelante.

    La forma mas fiable de saber que columnas salen es transformar unas pocas
    filas durante `fit` y mirar el resultado.
    """

    def _registrar_nombres(self, X):
        # Hay una dependencia circular que resolver: scikit-learn envuelve
        # `transform` y, al salir, le pregunta al operador como se llaman sus
        # columnas... que es justo lo que estamos intentando averiguar.
        #
        # Se rompe poniendo un centinela: mientras vale None, el metodo de
        # abajo no responde nada y scikit-learn conserva los nombres que trae
        # el DataFrame. Despues ya se guarda la respuesta definitiva.
        self.feature_names_out_ = None
        muestra = pd.DataFrame(X).head(5)
        self.feature_names_out_ = list(self.transform(muestra).columns)
        return self

    def get_feature_names_out(self, input_features=None):
        nombres = getattr(self, "feature_names_out_", None)
        if nombres is None:
            return None if input_features is None else np.asarray(input_features, dtype=object)
        return np.asarray(nombres, dtype=object)


# ===========================================================================
# 1. UnirCatalogos
# ===========================================================================
class UnirCatalogos(_NombresDeSalida, BaseEstimator, TransformerMixin):
    """Pega a cada fila los datos de catalogo: categoria del producto y nombres.

    Para que sirve:
        `test.csv` solo trae ID, shop_id e item_id. Para predecir hace falta
        saber de que categoria es el producto y en que ciudad esta la tienda.
        Este operador guarda los catalogos dentro del artefacto, de modo que el
        `.pkl` sea autosuficiente y el notebook de inferencia no tenga que
        cargar ningun archivo extra.

    Tambien rellena `date_block_num` cuando falta: `test.csv` no lo trae, y el
    mes a predecir es un valor conocido y fijo (34 = noviembre de 2015).
    """

    def __init__(self, items: pd.DataFrame, shops: pd.DataFrame,
                 categorias: pd.DataFrame, mes_por_defecto: int = 34):
        self.items = items
        self.shops = shops
        self.categorias = categorias
        self.mes_por_defecto = mes_por_defecto

    def fit(self, X, y=None):
        # Se guardan solo las columnas necesarias, para que el .pkl no engorde.
        self.items_ = self.items[["item_id", "item_category_id"]].astype(
            {"item_id": "int16", "item_category_id": "int8"}
        )
        self.shops_ = self.shops[["shop_id", "shop_name"]].copy()
        self.categorias_ = self.categorias[["item_category_id", "item_category_name"]].copy()
        return self._registrar_nombres(X)

    # Orden fijo de las tres columnas de entrada. NO es cosmetico: ver abajo.
    CLAVES: ClassVar[list[str]] = ["date_block_num", "shop_id", "item_id"]

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if "date_block_num" not in X.columns:
            X["date_block_num"] = np.int8(self.mes_por_defecto)
        X = X.merge(self.items_, on="item_id", how="left")
        X = X.merge(self.shops_, on="shop_id", how="left")
        X = X.merge(self.categorias_, on="item_category_id", how="left")

        # ------------------------------------------------------------------
        # Las columnas se reordenan SIEMPRE igual. Esto corrige un error real y
        # silencioso que costo caro:
        #
        # Al entrenar, la entrada trae `date_block_num` en primer lugar. Al
        # predecir, `test.csv` no lo trae y la linea de arriba lo anade al
        # FINAL. Los nombres eran correctos en ambos casos, pero scikit-learn,
        # cuando un operador declara sus nombres de salida, los reasigna POR
        # POSICION: tomaba la lista aprendida en el ajuste y la pegaba sobre
        # unas columnas que venian en otro orden.
        #
        # Resultado: en inferencia, la columna llamada `shop_id` contenia
        # identificadores de producto y la llamada `item_id` contenia el numero
        # de mes. El pipeline no fallaba, no avisaba de nada, y devolvia
        # predicciones perfectamente plausibles y completamente equivocadas.
        #
        # La prueba `test_el_orden_de_columnas_no_cambia_el_resultado` lo vigila.
        # ------------------------------------------------------------------
        # La reordenacion solo se hace si hace falta. Seleccionar columnas crea
        # una copia completa del DataFrame, y aqui dentro todavia viven los dos
        # nombres en cirilico, que sobre 11 millones de filas pesan 2.9 GB:
        # copiarlos sin necesidad duplica el pico de memoria y agota la maquina.
        # Al entrenar el orden ya es el correcto y no se copia nada; al predecir
        # son 214 000 filas y la copia es irrelevante.
        orden = self.CLAVES + [c for c in X.columns if c not in self.CLAVES]
        return X if list(X.columns) == orden else X[orden]


# ===========================================================================
# 2. CaracteristicasCalendario
# ===========================================================================
class CaracteristicasCalendario(_NombresDeSalida, BaseEstimator, TransformerMixin):
    """Deriva del numero de mes todo lo que el calendario puede aportar.

    Por que:
        El EDA mostro dos cosas. Que diciembre vende muchisimo mas que el resto
        (es el unico mes claramente por encima del nivel) y que el sabado vende
        1.82 veces lo del lunes. Lo segundo no se puede modelar directamente en
        una serie mensual, pero si indirectamente: un mes con cinco fines de
        semana vende mas que uno con cuatro.

    No aprende nada de los datos, asi que `fit` no hace nada: el calendario es
    el mismo con o sin datos de entrenamiento.
    """

    def fit(self, X, y=None):
        return self._registrar_nombres(X)

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        meses_num = X["date_block_num"].to_numpy().astype("int16")

        # El mes y el anio salen de dos operaciones aritmeticas, no hace falta
        # construir fechas: date_block_num 0 es enero de 2013, asi que el resto
        # de dividir entre 12 da el mes y el cociente da los anios transcurridos.
        X["mes_anio"] = (meses_num % 12 + 1).astype("int8")
        X["anio"] = (ANIO_INICIAL + meses_num // 12).astype("int16")
        X["es_diciembre"] = (X["mes_anio"] == 12).astype("int8")

        # Dias totales y fines de semana de cada mes.
        #
        # Nota de rendimiento: esto se calcula UNA vez por mes distinto (como
        # mucho 35 valores) y luego se reparte a los 11 millones de filas con
        # una busqueda vectorizada de numpy. La version ingenua -recorrer las
        # filas con una funcion de Python- tardaba 198 segundos; esta tarda
        # menos de uno. Sobre datos de este tamanio, la diferencia entre
        # vectorizar y no vectorizar no es un detalle de estilo.
        meses = X["date_block_num"].to_numpy()
        tope = int(meses.max()) + 1
        tabla_dias = np.zeros(tope, dtype="int8")
        tabla_findes = np.zeros(tope, dtype="int8")
        for m in np.unique(meses):
            p = mes_a_periodo([m])[0]
            dias = pd.date_range(p.start_time, p.end_time, freq="D")
            tabla_dias[m] = len(dias)
            tabla_findes[m] = int((dias.day_of_week >= 5).sum())
        X["dias_del_mes"] = tabla_dias[meses]
        X["findes_del_mes"] = tabla_findes[meses]
        return X


# ===========================================================================
# 3. LagFeatures  (el operador central del proyecto)
# ===========================================================================
class LagFeatures(_NombresDeSalida, BaseEstimator, TransformerMixin):
    """Crea los rezagos: cuanto vendio esta misma combinacion en meses anteriores.

    Que es un rezago (lag):
        Para predecir las ventas de noviembre, la informacion mas util es cuanto
        se vendio en octubre, en septiembre y en noviembre del anio pasado. Cada
        una de esas cifras es un "rezago": el valor de la misma serie, k meses
        antes.

    De donde salen los rezagos del mes que se predice:
        Aqui esta la parte delicada. `test.csv` corresponde al mes 34 y solo
        trae tienda y producto. Sus rezagos estan en los meses 33, 32, 31...,
        que viven en los datos de ENTRENAMIENTO. Por eso `fit` guarda el
        historial completo dentro del operador: asi el `.pkl` puede calcular los
        rezagos del mes 34 sin necesidad de volver a leer `sales_train.csv`.
        El historial pesa unos 13 MB, asi que cabe de sobra.

    Como se evita la fuga de informacion:
        El rezago k de la fila del mes m se busca en el mes m - k, NUNCA en el
        mes m ni en los posteriores. La busqueda es un cruce por la clave
        (tienda, producto, m - k), asi que por construccion es imposible que
        entre informacion del futuro. La prueba automatizada
        `tests/test_features.py` lo verifica.

    Que significa un rezago vacio:
        - Si el mes m - k existe pero ese par no vendio nada, el rezago es CERO.
          Es informacion real: no vendio.
        - Si el mes m - k no existe (por ejemplo, el rezago 12 de un mes
          anterior al numero 12), el rezago es NULO. Es informacion ausente, y
          se imputa mas adelante en el pipeline.
        Distinguir ambos casos importa: confundir "no vendio" con "no sabemos"
        le enseniaria al modelo algo falso.
    """

    def __init__(self, panel: pd.DataFrame, lags=(1, 2, 3, 12),
                 ventanas=(3, 6), precios: pd.DataFrame | None = None):
        self.panel = panel
        self.lags = lags
        self.ventanas = ventanas
        self.precios = precios

    # ---------------------------------------------------------------- fit
    def fit(self, X, y=None):
        panel = self.panel.copy()
        self.mes_minimo_ = int(panel["date_block_num"].min())

        # (a) Historial por par tienda-producto: la serie que se va a rezagar.
        self.panel_ = panel

        # (b) Agregados de apoyo. Cuando un par no tiene historial -casi la
        #     mitad de test-, lo unico que queda es "como le fue al producto en
        #     general", "como le fue a la tienda" y "como le fue a la categoria".
        self.agg_item_ = (
            panel.groupby(["date_block_num", "item_id"], observed=True)["item_cnt_month"]
            .mean().rename("item_media").reset_index()
        )
        self.agg_shop_ = (
            panel.groupby(["date_block_num", "shop_id"], observed=True)["item_cnt_month"]
            .mean().rename("shop_media").reset_index()
        )

        # (c) Precio medio del producto por mes. El EDA mostro que la mezcla de
        #     producto cambia con el tiempo (se vende menos cantidad de cosas
        #     mas caras), asi que el precio aporta senial.
        if self.precios is not None:
            self.precios_ = self.precios
        else:
            self.precios_ = pd.DataFrame(
                columns=["date_block_num", "item_id", "precio_medio"]
            )
        return self._registrar_nombres(X)

    # ----------------------------------------------------------- transform
    def _rezagar(self, X, tabla, claves, columna, k, nombre):
        """Cruza `tabla` desplazada k meses hacia atras sobre X.

        El truco es sencillo: se le suma k al mes de la tabla historica, de modo
        que el valor del mes m-k quede alineado con la fila del mes m.
        """
        desplazada = tabla[claves + [columna]].copy()
        desplazada["date_block_num"] = desplazada["date_block_num"] + k
        desplazada = desplazada.rename(columns={columna: nombre})
        X = X.merge(desplazada, on=claves, how="left")
        # Si el mes de origen existe en el historial, un valor ausente significa
        # "no vendio" -> cero. Si el mes de origen es anterior al inicio de los
        # datos, no hay forma de saberlo -> se deja nulo para imputar despues.
        existe_origen = X["date_block_num"] - k >= self.mes_minimo_
        X[nombre] = X[nombre].where(~existe_origen | X[nombre].notna(), 0.0)
        X[nombre] = X[nombre].astype("float32")
        return X

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # Rezagos del par tienda-producto: la senial principal.
        #
        # Ademas de los rezagos que pide el modelo, se calculan los auxiliares
        # que hagan falta para las medias moviles. Por ejemplo, la media de 6
        # meses necesita los rezagos 1 a 6, aunque el modelo solo use el 1, 2, 3
        # y 12 como variables sueltas. Los auxiliares se descartan al final.
        ventana_maxima = max(self.ventanas) if self.ventanas else 0
        necesarios = sorted(set(self.lags) | set(range(1, ventana_maxima + 1)))
        for k in necesarios:
            X = self._rezagar(X, self.panel_, ["date_block_num", "shop_id", "item_id"],
                              "item_cnt_month", k, f"lag_{k}")

        # Medias moviles: suavizan el ruido de un mes suelto. Responden a
        # "cuanto vende normalmente" en vez de "cuanto vendio justo el mes
        # pasado", que para una demanda intermitente como esta es mas estable.
        for v in self.ventanas:
            columnas = [f"lag_{k}" for k in range(1, v + 1)]
            X[f"media_{v}"] = X[columnas].mean(axis=1).astype("float32")

        # Fuera los rezagos auxiliares que el modelo no va a usar.
        sobrantes = [f"lag_{k}" for k in necesarios if k not in self.lags]
        X = X.drop(columns=sobrantes)

        # Rezago 1 de los agregados de apoyo.
        X = self._rezagar(X, self.agg_item_, ["date_block_num", "item_id"],
                          "item_media", 1, "item_lag_1")
        X = self._rezagar(X, self.agg_shop_, ["date_block_num", "shop_id"],
                          "shop_media", 1, "shop_lag_1")

        # Precio medio del producto el mes anterior.
        if len(self.precios_):
            X = self._rezagar(X, self.precios_, ["date_block_num", "item_id"],
                              "precio_medio", 1, "precio_lag_1")
        else:
            X["precio_lag_1"] = np.float32(np.nan)

        # Antiguedad del par: cuantos meses lleva vendiendose. Un producto nuevo
        # y uno veterano con las mismas ventas el mes pasado no se comportan
        # igual.
        if not hasattr(self, "primera_venta_"):
            self.primera_venta_ = (
                self.panel_.groupby(["shop_id", "item_id"], observed=True)["date_block_num"]
                .min().rename("primer_mes").reset_index()
            )
        X = X.merge(self.primera_venta_, on=["shop_id", "item_id"], how="left")
        X["meses_de_vida"] = (X["date_block_num"] - X["primer_mes"]).astype("float32")
        # Sin historial previo, la antiguedad es desconocida, no cero.
        X.loc[X["primer_mes"].isna(), "meses_de_vida"] = np.nan
        X = X.drop(columns=["primer_mes"])
        return X


# ===========================================================================
# 4. Mapper
# ===========================================================================
class Mapper(_NombresDeSalida, BaseEstimator, TransformerMixin):
    """Traduce texto a codigos usando diccionarios aprendidos del entrenamiento.

    Que hace:
        Un modelo no sabe leer "Москва ТЦ Семеновский". Este operador extrae la
        parte util del texto (la ciudad), construye un diccionario
        {"Москва": 0, "СПб": 1, ...} durante `fit`, y lo aplica durante
        `transform`.

    Por que un operador propio y no `OrdinalEncoder`:
        Porque antes de codificar hay que EXTRAER: la ciudad esta dentro del
        nombre de la tienda, y la plataforma (PS3, XBOX, PC...) dentro del
        nombre de la categoria. El encoder de scikit-learn codifica, no extrae.

    Que pasa con un valor que no estaba en el entrenamiento:
        Se marca como NULO en vez de inventarle un codigo. Eso es honesto -el
        modelo nunca lo vio- y ademas produce los valores ausentes que el
        imputador de variables categoricas se encarga de rellenar mas adelante.
    """

    def __init__(self, extractores: dict[str, tuple[str, callable]],
                 eliminar_origen: bool = True):
        # extractores = {columna_nueva: (columna_origen, funcion_de_extraccion)}
        self.extractores = extractores
        self.eliminar_origen = eliminar_origen

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.mapas_ = {}
        for destino, (origen, funcion) in self.extractores.items():
            # Solo hay 60 nombres de tienda y 84 de categoria distintos. Aplicar
            # la funcion de extraccion a los valores UNICOS y luego repartir el
            # resultado es decenas de veces mas rapido que aplicarla a los 11
            # millones de filas una por una (117 s contra menos de 2 s).
            unicos = X[origen].astype(str).unique()
            extraidos = {u: funcion(u) for u in unicos}
            categorias = sorted({v for v in extraidos.values() if v is not None})
            self.mapas_[destino] = {v: i for i, v in enumerate(categorias)}
            # Se guarda la traduccion directa texto-original -> codigo, para
            # que `transform` sea un unico mapeo de diccionario.
            self.directo_ = getattr(self, "directo_", {})
            self.directo_[destino] = {
                u: self.mapas_[destino].get(v) for u, v in extraidos.items()
            }
        return self._registrar_nombres(X)

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for destino, (origen, _) in self.extractores.items():
            # .map() con un diccionario deja NaN en lo no encontrado: justo lo
            # que queremos para las categorias nunca vistas.
            X[destino] = X[origen].astype(str).map(self.directo_[destino]).astype("float32")
        if self.eliminar_origen:
            # Las columnas de texto ya cumplieron su funcion y ocupan 2.9 GB
            # sobre este volumen de filas. Se retiran para que el pipeline quepa
            # en memoria.
            origenes = {o for o, _ in self.extractores.values()}
            X = X.drop(columns=[c for c in origenes if c in X.columns])
        return X


def extraer_ciudad(nombre_tienda: str) -> str:
    """Primera palabra del nombre de la tienda, que es siempre la ciudad.

    "!Якутск Орджоникидзе, 56 фран" -> "Якутск"
    El "!" inicial se retira: el EDA mostro que distingue dos registros de la
    MISMA tienda, no dos ciudades.
    """
    return nombre_tienda.strip().lstrip("!").split()[0].strip(",")


def extraer_tipo_categoria(nombre_categoria: str) -> str:
    """Parte anterior al guion del nombre de la categoria.

    "Игры - PS3" -> "Игры" (juegos);  "Кино - DVD" -> "Кино" (cine).
    Agrupa las 84 categorias en una decena de familias, que es una variable
    mucho mas util para un modelo que 84 codigos sueltos.
    """
    return nombre_categoria.split("-")[0].strip()


# ===========================================================================
# 5. OutlierCapper
# ===========================================================================
class OutlierCapper(BaseEstimator, TransformerMixin):
    """Recorta los valores extremos a unos percentiles aprendidos en el entrenamiento.

    Adaptado del operador visto en clase (`Clase4-SL1.py`), con dos cambios:

    1. Devuelve un DataFrame y no un `ndarray`. El original perdia los nombres
       de las columnas, y al encadenarlo con otros pasos el pipeline se rompia.
    2. Usa percentiles configurables en lugar del rango intercuartilico fijo.
       Con una distribucion tan sesgada como esta -el 89.6 % de las ventas son
       de una sola unidad- la regla 1.5*IQR recortaria una parte enorme de los
       datos legitimos.

    Importante: los limites se aprenden SOLO con los datos de entrenamiento y
    luego se aplican igual a los de prueba. Calcularlos sobre todo el conjunto
    seria fuga de informacion.
    """

    def __init__(self, inferior: float = 0.01, superior: float = 0.99):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.limites_ = {
            col: (X[col].quantile(self.inferior), X[col].quantile(self.superior))
            for col in X.columns
        }
        self.feature_names_out_ = list(X.columns)
        return self

    def get_feature_names_out(self, input_features=None):
        # Recorta valores, no crea ni elimina columnas: salen las mismas que entran.
        return np.asarray(self.feature_names_out_, dtype=object)

    def transform(self, X):
        # Defensa: si el paso anterior entrego una matriz sin nombres -cosa que
        # ocurre cuando la salida en formato DataFrame no esta activada-, se le
        # vuelven a poner los nombres aprendidos en `fit`. Sin esto, el operador
        # falla con `KeyError` al buscar sus columnas por nombre.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X), columns=self.feature_names_out_)
        else:
            X = X.copy()
        for col, (bajo, alto) in self.limites_.items():
            X[col] = np.clip(X[col], bajo, alto)
        return X


# ===========================================================================
# 6. ReducirPrecision
# ===========================================================================
class ReducirPrecision(BaseEstimator, TransformerMixin):
    """Convierte la matriz a `float32` para que quepa en memoria.

    Por defecto, scikit-learn trabaja en `float64`: ocho bytes por numero. Con
    11 millones de filas y unas 25 columnas eso son 2.2 GB, y esta maquina
    tiene 8 GB en total. En `float32` son la mitad, 1.1 GB, y la precision
    perdida es irrelevante para datos que son conteos de unidades vendidas.

    Es un paso de fontaneria, no de modelado, pero sin el el pipeline no corre.
    """

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_out_ = list(X.columns)
        return self

    def get_feature_names_out(self, input_features=None):
        # Solo cambia el tipo numerico: los nombres pasan intactos.
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray(self.feature_names_out_, dtype=object)

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.astype("float32")
        return np.asarray(X, dtype="float32")


# ===========================================================================
# 7. PerforadorAleatorio
# ===========================================================================
class PerforadorAleatorio(BaseEstimator, TransformerMixin):
    """Borra al azar una fraccion de una columna, para poder demostrar la imputacion.

    Esto NO es una tecnica de modelado: es un requisito explicito del enunciado.
    El dataset no tiene un solo valor nulo en sus variables categoricas, asi que
    el proyecto pide "forzar que sea posible aplicar" la imputacion categorica
    eliminando algunos datos de forma aleatoria.

    Se usa una semilla fija para que el resultado sea identico en cada ejecucion,
    y solo actua sobre los datos de entrenamiento: en inferencia no borra nada,
    porque ahi perder informacion a proposito no tendria ningun sentido.
    """

    def __init__(self, columnas: list[str], fraccion: float = 0.01, semilla: int = 42):
        self.columnas = columnas
        self.fraccion = fraccion
        self.semilla = semilla

    def fit(self, X, y=None):
        self.entrenado_ = True
        return self

    def transform(self, X, perforar: bool = False):
        X = pd.DataFrame(X).copy()
        if not perforar:
            return X
        rng = np.random.default_rng(self.semilla)
        for col in self.columnas:
            mascara = rng.random(len(X)) < self.fraccion
            X.loc[mascara, col] = np.nan
        return X
