# =============================================================================
# PROCESAMIENTO BATCH CON APACHE SPARK - DEMANDA DE TAXIS AMARILLOS DE NUEVA YORK
# Curso: Big Data (UNAD)  |  Autor: [Aldemar Mora Sabogal]
# =============================================================================
#
# DESCRIPCION DE LA SOLUCION
# --------------------------
# Problema: entender cuando, donde y cuanto se usa el servicio de taxis amarillos
# de Nueva York, a partir de mas de tres millones de viajes registrados en un mes.
#
# Datos: "Yellow Taxi Trip Records", enero de 2023, publicados por la NYC Taxi and
# Limousine Commission (TLC). Archivo yellow_tripdata_2023-01.parquet (45,5 MB,
# 3.066.766 registros, 19 columnas), almacenado en HDFS en /Proyecto.
#
# Solucion: aplicacion PySpark (modo local, 1 GB de memoria del driver) con un
# flujo ETL + analisis exploratorio:
#   1. CARGA          Lee el parquet desde HDFS y cuenta los registros originales.
#   2. LIMPIEZA       Elimina nulos, duplicados y valores fuera de rango
#                     (distancia, tarifa y numero de pasajeros).
#   3. TRANSFORMACION Calcula la duracion del viaje, la hora y el dia de la semana.
#   4. ANALISIS (EDA) Resumen de registros, estadisticas descriptivas, demanda por
#                     hora, metodo de pago, zonas, dia de la semana y pasajeros
#                     (este ultimo calculado con un RDD).
#   5. VISUALIZACION  Genera tres graficas PNG con matplotlib en la carpeta output/.
#
#
#
# REQUISITOS
# ----------
#   - Ubuntu con Java 11, Hadoop 3.3.6 (HDFS encendido) y Spark 3.5.x
#   - Python 3.10 con matplotlib:  pip3 install --user matplotlib
#   - El dataset cargado en HDFS (ver INSTRUCCIONES DE EJECUCION)
#
# INSTRUCCIONES DE EJECUCION
# --------------------------
#   1. Encender HDFS (usuario hadoop):
#        su - hadoop
#        start-dfs.sh
#        hdfs dfsadmin -safemode wait
#
#   2. Cargar el dataset en HDFS (usuario hadoop, una sola vez):
#        cd ~
#        wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet
#        hdfs dfs -mkdir -p /Proyecto
#        hdfs dfs -put yellow_tripdata_2023-01.parquet /Proyecto/
#        hdfs dfs -ls -h /Proyecto
#        exit
#
#   3. Ejecutar el script (usuario vboxuser, desde la carpeta del proyecto):
#        cd ~/proyecto
#        spark-submit --driver-memory 1g batch_analysis.py 2> log_spark.txt | tee resultados_batch.txt
#
#      El reporte se muestra en pantalla y se guarda en resultados_batch.txt.
#
#
#   4. Ver las graficas desde Windows:
#        cd ~/proyecto/output && python3 -m http.server 8000
#        Abrir http://IP-DE-LA-VM:8000/viajes_por_hora.png (y las otras dos).
#
# SALIDAS
# -------
#   resultados_batch.txt        Reporte con 8 tablas (ver secciones 1 a 8 abajo)
#   output/viajes_por_hora.png  Viajes y tarifa promedio por hora
#   output/top_zonas.png        Diez zonas de recogida con mas viajes
#   output/viajes_por_dia.png   Viajes por dia de la semana
# =============================================================================
import os
import sys
import matplotlib
matplotlib.use("Agg")            # sin pantalla (VM por SSH)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F

# Ruta del dataset en HDFS (se puede cambiar con el primer argumento)
DATA = sys.argv[1] if len(sys.argv) > 1 else "hdfs://localhost:9000/Proyecto/yellow_tripdata_2023-01.parquet"
os.makedirs("output", exist_ok=True)

# Codigos del diccionario de datos de la TLC, traducidos para el reporte
PAGOS = {1: "Tarjeta de credito", 2: "Efectivo", 3: "Sin cargo",
         4: "Disputa", 5: "Desconocido", 6: "Viaje anulado"}
DIAS = {1: "Domingo", 2: "Lunes", 3: "Martes", 4: "Miercoles",
        5: "Jueves", 6: "Viernes", 7: "Sabado"}


def num(x, dec=2):
    """Formato colombiano: 1.234.567,89 (maximo 'dec' decimales)."""
    s = f"{x:,.{dec}f}"
    return s.replace(",", "#").replace(".", ",").replace("#", ".")


def tabla(titulo, encabezados, filas):
    """Imprime una tabla de texto alineada."""
    filas = [[str(c) for c in f] for f in filas]
    anchos = [max([len(h)] + [len(f[i]) for f in filas])
              for i, h in enumerate(encabezados)]

    def linea(celdas):
        return " | ".join(c.ljust(anchos[i]) if i == 0 else c.rjust(anchos[i])
                          for i, c in enumerate(celdas))
    print(f"\n{titulo}")
    print(linea(encabezados))
    print("-+-".join("-" * a for a in anchos))
    for f in filas:
        print(linea(f))


# Sesion de Spark en modo local; 8 particiones de shuffle bastan para 3 M de filas
spark = (SparkSession.builder.appName("TaxiBatch")
         .master("local[*]")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------- 1. CARGA
# Lectura del parquet desde HDFS; el conteo inicial es el total de registros
df_raw = spark.read.parquet(DATA)
total_original = df_raw.count()
tabla("1. ESTRUCTURA DEL DATASET ORIGINAL",
      ["Columna", "Tipo de dato"], df_raw.dtypes)

# ------------------------------------------------------------- 2. LIMPIEZA
# - dropna: sin estas columnas no se puede calcular duracion, tarifa ni demanda
# - dropDuplicates: un mismo viaje no debe contarse dos veces
# - filtros: excluyen distancias, tarifas y pasajeros imposibles
df = (df_raw.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime",
                            "passenger_count", "trip_distance", "fare_amount"])
        .dropDuplicates(["tpep_pickup_datetime", "tpep_dropoff_datetime",
                         "PULocationID", "DOLocationID", "fare_amount"])
        .filter((F.col("trip_distance") > 0) & (F.col("trip_distance") < 100))
        .filter((F.col("fare_amount") > 0) & (F.col("fare_amount") < 500))
        .filter((F.col("passenger_count") > 0) & (F.col("passenger_count") <= 6)))

# ------------------------------------------------------- 3. TRANSFORMACION
# Variables derivadas: duracion en minutos, hora de recogida y dia de la semana.
# persist() guarda el resultado en memoria/disco para reutilizarlo en el analisis.
df = (df.withColumn("duracion_min",
                    (F.unix_timestamp("tpep_dropoff_datetime") -
                     F.unix_timestamp("tpep_pickup_datetime")) / 60)
        .filter((F.col("duracion_min") > 1) & (F.col("duracion_min") < 180))
        .withColumn("hora", F.hour("tpep_pickup_datetime"))
        .withColumn("dia_semana", F.dayofweek("tpep_pickup_datetime"))
        .persist(StorageLevel.MEMORY_AND_DISK))

# Comparacion antes y despues de la limpieza
total_limpio = df.count()
descartados = total_original - total_limpio
tabla("2. RESUMEN DE REGISTROS",
      ["Concepto", "Cantidad", "Porcentaje"],
      [("Total de registros del dataset (antes de la limpieza)",
        num(total_original, 0), "100,00 %"),
       ("Registros descartados en la limpieza",
        num(descartados, 0), num(descartados / total_original * 100) + " %"),
       ("Registros validos (despues de la limpieza)",
        num(total_limpio, 0), num(total_limpio / total_original * 100) + " %")])

# ------------------------------------------------------------------- 4. EDA
# Estadisticas descriptivas: se calculan en una sola pasada con agg()
variables = [("trip_distance", "Distancia del viaje (millas)"),
             ("duracion_min", "Duracion del viaje (minutos)"),
             ("fare_amount", "Tarifa base (USD)"),
             ("tip_amount", "Propina (USD)"),
             ("total_amount", "Total pagado (USD)")]
aggs = []
for c, _ in variables:
    aggs += [F.avg(c).alias(c + "_avg"), F.stddev(c).alias(c + "_std"),
             F.min(c).alias(c + "_min"), F.max(c).alias(c + "_max")]
r = df.agg(*aggs).first()
tabla("3. ESTADISTICAS DESCRIPTIVAS (datos validos)",
      ["Variable", "Promedio", "Desv. estandar", "Minimo", "Maximo"],
      [(nombre, num(r[c + "_avg"]), num(r[c + "_std"]),
        num(r[c + "_min"]), num(r[c + "_max"])) for c, nombre in variables])

# --- Viajes por hora: agrupa por hora de recogida y mide viajes, tarifa y duracion
h = (df.groupBy("hora")
       .agg(F.count("*").alias("viajes"),
            F.avg("fare_amount").alias("tarifa"),
            F.avg("duracion_min").alias("duracion"))
       .orderBy("hora").collect())
tabla("4. DEMANDA POR HORA DEL DIA",
      ["Hora", "Viajes", "% del total", "Tarifa promedio (USD)", "Duracion promedio (min)"],
      [(f"{x['hora']:02d}:00", num(x["viajes"], 0),
        num(x["viajes"] / total_limpio * 100) + " %",
        num(x["tarifa"]), num(x["duracion"])) for x in h])
pico = max(h, key=lambda x: x["viajes"])
valle = min(h, key=lambda x: x["viajes"])
print(f"\nHora de mayor demanda: {pico['hora']:02d}:00 ({num(pico['viajes'], 0)} viajes)")
print(f"Hora de menor demanda: {valle['hora']:02d}:00 ({num(valle['viajes'], 0)} viajes)")

# --- Metodo de pago: cantidad de viajes y propina promedio por forma de pago
p = (df.groupBy("payment_type")
       .agg(F.count("*").alias("viajes"), F.avg("tip_amount").alias("propina"))
       .orderBy(F.desc("viajes")).collect())
tabla("5. METODO DE PAGO",
      ["Metodo de pago", "Viajes", "% del total", "Propina promedio (USD)"],
      [(PAGOS.get(x["payment_type"], str(x["payment_type"])), num(x["viajes"], 0),
        num(x["viajes"] / total_limpio * 100) + " %", num(x["propina"])) for x in p])
print("\nNota: las propinas en efectivo no quedan registradas en el sistema (aparecen en 0).")

# --- Zonas: las diez zonas de recogida con mas viajes (se muestra el ID de zona)
z = df.groupBy("PULocationID").count().orderBy(F.desc("count")).limit(10).collect()
tabla("6. TOP 10 ZONAS DE RECOGIDA",
      ["Puesto", "ID de zona", "Viajes", "% del total"],
      [(i + 1, x["PULocationID"], num(x["count"], 0),
        num(x["count"] / total_limpio * 100) + " %") for i, x in enumerate(z)])

# --- Dia de la semana (dayofweek: 1 = domingo ... 7 = sabado)
d = df.groupBy("dia_semana").count().orderBy("dia_semana").collect()
tabla("7. VIAJES POR DIA DE LA SEMANA",
      ["Dia", "Viajes", "% del total"],
      [(DIAS[x["dia_semana"]], num(x["count"], 0),
        num(x["count"] / total_limpio * 100) + " %") for x in d])

# --- Pasajeros: se calcula con un RDD (map + reduceByKey) para mostrar esa API
rdd = (df.select("passenger_count").rdd
         .map(lambda row: (int(row[0]), 1))
         .reduceByKey(lambda a, b: a + b)
         .sortByKey())
pas = rdd.collect()
tabla("8. VIAJES POR NUMERO DE PASAJEROS (calculado con RDD)",
      ["Pasajeros", "Viajes", "% del total"],
      [(k, num(v, 0), num(v / total_limpio * 100) + " %") for k, v in pas])

# ---------------------------------------------------------- 5. VISUALIZACION
# Solo se grafican resultados ya agregados (pocas filas), no los datos crudos
miles = FuncFormatter(lambda v, _: num(v, 0))

# Grafica 1: viajes por hora (barras) y tarifa promedio por hora (linea)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].bar([x["hora"] for x in h], [x["viajes"] for x in h], color="#2a7fba")
ax[0].set_title("Viajes por hora del dia")
ax[0].set_xlabel("Hora del dia (0-23)"); ax[0].set_ylabel("Numero de viajes")
ax[0].yaxis.set_major_formatter(miles)
ax[1].plot([x["hora"] for x in h], [x["tarifa"] for x in h], marker="o", color="#d9534f")
ax[1].set_title("Tarifa promedio por hora")
ax[1].set_xlabel("Hora del dia (0-23)"); ax[1].set_ylabel("Tarifa promedio (USD)")
plt.tight_layout(); plt.savefig("output/viajes_por_hora.png", dpi=120); plt.close()

# Grafica 2: top 10 zonas de recogida
plt.figure(figsize=(7, 4.5))
plt.barh([f"Zona {x['PULocationID']}" for x in z][::-1],
         [x["count"] for x in z][::-1], color="#5cb85c")
plt.title("Top 10 zonas de recogida"); plt.xlabel("Numero de viajes")
plt.gca().xaxis.set_major_formatter(miles)
plt.tight_layout(); plt.savefig("output/top_zonas.png", dpi=120); plt.close()

# Grafica 3: viajes por dia de la semana
plt.figure(figsize=(7, 4.5))
plt.bar([DIAS[x["dia_semana"]] for x in d], [x["count"] for x in d], color="#f0ad4e")
plt.title("Viajes por dia de la semana"); plt.ylabel("Numero de viajes")
plt.gca().yaxis.set_major_formatter(miles)
plt.xticks(rotation=30)
plt.tight_layout(); plt.savefig("output/viajes_por_dia.png", dpi=120); plt.close()

print("\nGraficas guardadas en la carpeta output/")
spark.stop()
