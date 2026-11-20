"""Pruebas de los operadores de ingenieria de caracteristicas.

La prueba que importa de verdad es `test_lag_no_mira_el_futuro`. Todo lo demas
de este proyecto puede estar bien y, si los rezagos miran hacia adelante, el
modelo saldra fantastico en validacion y fracasara en Kaggle. Es el error mas
caro y el mas silencioso, asi que se verifica con datos inventados donde la
respuesta correcta se conoce de antemano.

Ejecutar con:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from predict_future_sales.features import (
    UnirCatalogos,
    CaracteristicasCalendario,
    LagFeatures,
    Mapper,
    OutlierCapper,
    construir_rejilla,
    extraer_ciudad,
    extraer_tipo_categoria,
    panel_mensual,
)


# ---------------------------------------------------------------- utilidades
@pytest.fixture
def ventas_de_juguete():
    """Un dataset minimo donde se sabe la respuesta de memoria.

    Una sola tienda (5) y un solo producto (100), vendiendo una cantidad
    distinta y reconocible cada mes: 10 unidades el mes 0, 20 el mes 1, y asi
    hasta 40 el mes 3.
    """
    filas = []
    for mes, cantidad in enumerate([10, 20, 30, 40]):
        filas.append({
            "date": pd.Timestamp(2013, mes + 1, 15),
            "date_block_num": mes,
            "shop_id": 5,
            "item_id": 100,
            "item_price": 100.0,
            "item_cnt_day": float(cantidad),
        })
    return pd.DataFrame(filas)


# ============================================================ LA PRUEBA CLAVE
def test_lag_no_mira_el_futuro(ventas_de_juguete):
    """El rezago 1 de cada mes debe ser la venta del mes ANTERIOR, nunca la suya.

    Con la serie 10, 20, 30, 40 la respuesta correcta es inequivoca:

        mes 0 -> no hay mes -1  -> nulo
        mes 1 -> venta del mes 0 -> 10
        mes 2 -> venta del mes 1 -> 20
        mes 3 -> venta del mes 2 -> 30

    Si en el mes 2 apareciera un 30, el operador se estaria copiando a si mismo
    y el modelo entrenaria con la respuesta puesta.
    """
    panel = panel_mensual(ventas_de_juguete)
    X = panel[["date_block_num", "shop_id", "item_id"]].copy()

    resultado = LagFeatures(panel=panel, lags=(1,), ventanas=()).fit(X).transform(X)
    lags = resultado.set_index("date_block_num")["lag_1"]

    assert np.isnan(lags.loc[0]), "El mes 0 no tiene pasado: su rezago debe ser nulo"
    assert lags.loc[1] == 10.0
    assert lags.loc[2] == 20.0
    assert lags.loc[3] == 30.0

    # Y la comprobacion en negativo: ningun rezago coincide con su propio mes.
    propio = resultado.set_index("date_block_num")["item_cnt_month"] \
        if "item_cnt_month" in resultado else None
    if propio is not None:
        for mes in (1, 2, 3):
            assert lags.loc[mes] != propio.loc[mes], f"El rezago del mes {mes} se copio a si mismo"


def test_lag_12_solo_existe_con_doce_meses_de_historia(ventas_de_juguete):
    """Con cuatro meses de datos, el rezago 12 no puede existir: debe ser nulo."""
    panel = panel_mensual(ventas_de_juguete)
    X = panel[["date_block_num", "shop_id", "item_id"]].copy()
    resultado = LagFeatures(panel=panel, lags=(12,), ventanas=()).fit(X).transform(X)
    assert resultado["lag_12"].isna().all()


def test_mes_sin_venta_da_cero_y_no_nulo():
    """Distinguir "no vendio" de "no sabemos" es una decision de modelado.

    Si el mes anterior existe en el historial pero ese par no vendio nada, el
    rezago correcto es CERO (informacion real). El nulo se reserva para cuando
    el mes anterior ni siquiera existe.
    """
    ventas = pd.DataFrame([
        # El par (5, 100) vende en los meses 0 y 2, pero no en el 1.
        {"date_block_num": 0, "shop_id": 5, "item_id": 100, "item_cnt_day": 10.0},
        {"date_block_num": 1, "shop_id": 5, "item_id": 999, "item_cnt_day": 7.0},
        {"date_block_num": 2, "shop_id": 5, "item_id": 100, "item_cnt_day": 30.0},
    ])
    panel = panel_mensual(ventas)
    X = pd.DataFrame([{"date_block_num": 2, "shop_id": 5, "item_id": 100}])
    resultado = LagFeatures(panel=panel, lags=(1,), ventanas=()).fit(X).transform(X)
    assert resultado["lag_1"].iloc[0] == 0.0, "El mes 1 existe y no vendio: debe ser cero"


# ======================================================== ORDEN DE COLUMNAS
def test_el_orden_de_columnas_no_cambia_el_resultado():
    """Las mismas filas, con las columnas en otro orden, deben dar lo mismo.

    Esta prueba existe porque el fallo ya ocurrio. `UnirCatalogos` anade
    `date_block_num` al final cuando falta -el caso de `test.csv`-, mientras que
    al entrenar viene al principio. Como el operador declara sus nombres de
    salida, scikit-learn los reasignaba POR POSICION y la columna llamada
    `shop_id` acababa conteniendo identificadores de producto.

    El pipeline no fallaba: devolvia predicciones plausibles y equivocadas. Es
    el error mas peligroso de un sistema de inferencia, porque no deja rastro.
    """
    items = pd.DataFrame({"item_id": [100, 200], "item_category_id": [1, 2]})
    shops = pd.DataFrame({"shop_id": [5, 6], "shop_name": ["Москва ТЦ", "СПб ТК"]})
    cats = pd.DataFrame({"item_category_id": [1, 2],
                         "item_category_name": ["Игры - PS3", "Кино - DVD"]})

    op = UnirCatalogos(items, shops, cats, mes_por_defecto=34)
    normal = pd.DataFrame({"date_block_num": [34, 34], "shop_id": [5, 6],
                           "item_id": [100, 200]})
    op.fit(normal)

    # a) misma informacion, pero con las columnas en otro orden
    revuelto = normal[["item_id", "shop_id", "date_block_num"]]
    # b) sin date_block_num, como viene test.csv
    sin_mes = normal[["shop_id", "item_id"]]

    a, b, c = op.transform(normal), op.transform(revuelto), op.transform(sin_mes)

    for nombre, salida in [("revuelto", b), ("sin date_block_num", c)]:
        assert list(salida.columns) == list(a.columns), f"El orden cambia con {nombre}"
        for col in ("shop_id", "item_id", "date_block_num", "item_category_id"):
            assert salida[col].tolist() == a[col].tolist(), (
                f"La columna {col} trae datos distintos con {nombre}: "
                f"{salida[col].tolist()} en vez de {a[col].tolist()}"
            )


# ==================================================================== rejilla
def test_rejilla_rellena_los_ceros():
    """La rejilla debe inventar las filas que faltan y ponerles cero."""
    ventas = pd.DataFrame([
        {"date_block_num": 0, "shop_id": 1, "item_id": 10, "item_cnt_day": 5.0},
        {"date_block_num": 0, "shop_id": 2, "item_id": 20, "item_cnt_day": 3.0},
    ])
    rejilla = construir_rejilla(ventas)
    # 2 tiendas x 2 productos = 4 combinaciones, aunque solo 2 tuvieron venta.
    assert len(rejilla) == 4
    assert (rejilla["item_cnt_month"] == 0).sum() == 2
    assert rejilla["item_cnt_month"].sum() == 8.0


# ===================================================================== Mapper
def test_mapper_codifica_y_marca_lo_no_visto():
    entrenamiento = pd.DataFrame({"shop_name": ["Москва ТЦ", "СПб ТК", "Москва ТРК"]})
    mapper = Mapper({"ciudad": ("shop_name", extraer_ciudad)}).fit(entrenamiento)

    # Dos ciudades distintas -> dos codigos distintos; la misma ciudad -> el mismo.
    salida = mapper.transform(entrenamiento)
    assert salida["ciudad"].iloc[0] == salida["ciudad"].iloc[2]
    assert salida["ciudad"].iloc[0] != salida["ciudad"].iloc[1]

    # Una ciudad nunca vista no recibe un codigo inventado, sino un nulo.
    nueva = pd.DataFrame({"shop_name": ["Казань ТЦ"]})
    assert mapper.transform(nueva)["ciudad"].isna().all()


def test_extractores_de_texto():
    assert extraer_ciudad("!Якутск Орджоникидзе, 56 фран") == "Якутск"
    assert extraer_ciudad("Москва ТЦ \"Семеновский\"") == "Москва"
    assert extraer_tipo_categoria("Игры - PS3") == "Игры"
    assert extraer_tipo_categoria("Кино - DVD") == "Кино"


# ============================================================== OutlierCapper
def test_capper_aprende_solo_del_entrenamiento():
    """Los limites salen del entrenamiento y se aplican igual a datos nuevos."""
    entrenamiento = pd.DataFrame({"precio": list(range(100))})
    capper = OutlierCapper(inferior=0.05, superior=0.95).fit(entrenamiento)

    nuevos = pd.DataFrame({"precio": [-500.0, 50.0, 10_000.0]})
    salida = capper.transform(nuevos)
    assert salida["precio"].iloc[0] > -500, "El valor bajo debio recortarse"
    assert salida["precio"].iloc[1] == 50.0, "Un valor normal no debe tocarse"
    assert salida["precio"].iloc[2] < 10_000, "El valor alto debio recortarse"


# ======================================================== calendario
def test_calendario_identifica_diciembre():
    X = pd.DataFrame({"date_block_num": [0, 11, 23, 34]})
    salida = CaracteristicasCalendario().fit(X).transform(X)
    # mes 0 = enero 2013, mes 11 = diciembre 2013, mes 23 = diciembre 2014,
    # mes 34 = noviembre 2015 (el que hay que predecir).
    assert list(salida["mes_anio"]) == [1, 12, 12, 11]
    assert list(salida["es_diciembre"]) == [0, 1, 1, 0]
    assert list(salida["anio"]) == [2013, 2013, 2014, 2015]
