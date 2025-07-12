import pandas as pd
from multiprocessing import Pool, cpu_count
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

CHUNK_SIZE = 500_000
CSV_PATH = "eldoria.csv"

def procesar_chunk(chunk):
    if 'GÉNERO' in chunk.columns:
        chunk.rename(columns={'GÉNERO': 'GENERO'}, inplace=True)

    chunk["estrato"] = chunk["CP ORIGEN"].astype(str).str[0]
    chunk["FECHA NACIMIENTO"] = pd.to_datetime(chunk["FECHA NACIMIENTO"], errors="coerce")
    chunk["edad"] = chunk["FECHA NACIMIENTO"].apply(lambda x: datetime.now().year - x.year if pd.notnull(x) else None)

    conteo_estrato = chunk["estrato"].value_counts().to_dict()

    edad_stats = (
        chunk.groupby(["ESPECIE", "GENERO"])["edad"]
        .agg(["mean", "median"])
        .dropna()
        .reset_index()
        .to_dict(orient="records")
    )

    def clasificar_edad(e):
        if pd.isna(e): return None
        if e < 18: return "0-17"
        elif e <= 35: return "18-35"
        elif e <= 60: return "36-60"
        else: return "61+"

    chunk["tramo"] = chunk["edad"].apply(clasificar_edad)
    tramos = chunk.groupby(["ESPECIE", "GENERO", "tramo"]).size().to_dict()

    menores_15 = chunk[chunk["edad"] < 15].shape[0]
    mayores_64 = chunk[chunk["edad"] > 64].shape[0]
    edad_trabajo = chunk[(chunk["edad"] >= 15) & (chunk["edad"] <= 64)].shape[0]
    dependencia = (menores_15 + mayores_64, edad_trabajo)

    viajes = (
        chunk.groupby(["CP ORIGEN", "CP DESTINO"])
        .size()
        .reset_index(name="conteo")
        .values
        .tolist()
    )

    # Pirámide de edades
    chunk["grupo_edad"] = chunk["edad"].apply(clasificar_grupo_5anios)
    piramide = chunk.groupby(["grupo_edad", "GENERO"]).size()

    return (conteo_estrato, edad_stats, tramos, dependencia, viajes, piramide)

def clasificar_grupo_5anios(edad):
    if pd.isna(edad) or edad < 0:
        return None
    elif edad >= 90:
        return "90+"
    else:
        base = int(edad // 5) * 5
        return f"{base}-{base+4}"


def combinar_resultados(resultados):
    total_estrato = defaultdict(int)
    edad_registros = []
    tramos_total = defaultdict(int)
    dep_numerador = 0
    dep_denominador = 0
    viajes_total = defaultdict(int)
    piramide_total = []

    for res in resultados:
        estrato, edades, tramos, dependencia, viajes, piramide = res

        for k, v in estrato.items():
            total_estrato[k] += v
        edad_registros.extend(edades)
        for k, v in tramos.items():
            tramos_total[k] += v
        dep_numerador += dependencia[0]
        dep_denominador += dependencia[1]
        for ori, dst, count in viajes:
            viajes_total[(ori, dst)] += count
        piramide_total.append(piramide)

    total_poblacion = sum(total_estrato.values())
    porcentaje_estrato = {k: round(v * 100 / total_poblacion, 2) for k, v in total_estrato.items()}
    top_10k = sorted(viajes_total.items(), key=lambda x: x[1], reverse=True)[:10000]
    piramide_final = sum(piramide_total)

    return {
        "conteo_estrato": dict(total_estrato),
        "porcentaje_estrato": porcentaje_estrato,
        "edad_estadisticas": edad_registros,
        "tramos": dict(tramos_total),
        "indice_dependencia": round(dep_numerador / dep_denominador, 3) if dep_denominador > 0 else None,
        "top_viajes": top_10k,
        "piramide_edades": piramide_final.reset_index().rename(columns={0: "cantidad"})
    }

def graficar_piramide(piramide_df):
    pivot = piramide_df.pivot_table(index="grupo_edad", columns="GENERO", values="cantidad", fill_value=0)
    grupos = sorted(pivot.index, key=lambda x: int(x.split("-")[0]) if "-" in x else 90)

    hembras = -pivot.loc[grupos]["HEMBRA"] if "HEMBRA" in pivot.columns else [0]*len(grupos)
    machos = pivot.loc[grupos]["MACHO"] if "MACHO" in pivot.columns else [0]*len(grupos)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(grupos, hembras, color="lightcoral", label="HEMBRA")
    ax.barh(grupos, machos, color="steelblue", label="MACHO")

    ax.set_title("Pirámide de Edades - Eldoria")
    ax.set_xlabel("Población")
    ax.set_ylabel("Grupo Etario")
    ax.legend()
    ax.axvline(0, color='black')
    plt.tight_layout()
    plt.savefig("piramide_edades.png")
    print("Pirámide de edades guardada como 'piramide_edades.png'.")

if __name__ == "__main__":
    print("Procesando datos por chunks con paralelismo...\n")
    pool = Pool(processes=cpu_count())
    reader = pd.read_csv(CSV_PATH, sep=";", quotechar='"', chunksize=CHUNK_SIZE, encoding="utf-8")
    resultados = pool.map(procesar_chunk, reader)
    pool.close()
    pool.join()

    final = combinar_resultados(resultados)

    print("\n=== RESULTADOS ===\n")

    print("1. ¿Cuántas personas pertenecen a cada estrato social?")
    for k, v in sorted(final["conteo_estrato"].items()):
        print(f"   - Estrato {k}: {v} personas")

    print("\n2. ¿Qué porcentaje de la población pertenece a cada estrato social?")
    for k, v in sorted(final["porcentaje_estrato"].items()):
        print(f"   - Estrato {k}: {v}%")

    print("\n3. ¿Cuál es la edad promedio según cada especie y género?")
    for row in final["edad_estadisticas"][:10]:
        print(f"   - {row['ESPECIE']} / {row['GENERO']}: Promedio = {round(row['mean'], 2)}")

    print("\n4. ¿Cuál es la edad mediana según cada especie y género?")
    for row in final["edad_estadisticas"][:10]:
        print(f"   - {row['ESPECIE']} / {row['GENERO']}: Mediana = {round(row['median'], 2)}")

    print("\n5. ¿Qué proporción de la población tiene menos de 18 años, entre 18–35, 36–60, más de 60 según especie y género?")
    for k, v in list(final["tramos"].items())[:10]:
        especie, genero, tramo = k
        print(f"   - {especie} / {genero} / {tramo}: {v} personas")

    print("\n6. ¿Cuál es la pirámide de edades de la población según especie, género?")
    graficar_piramide(final["piramide_edades"])

    print("\n7. ¿Cuál es el índice de dependencia?")
    print(f"   - Índice de dependencia: {final['indice_dependencia']}")

    print("\n8. ¿Cuáles son los 5 poblados con más viajes (CP ORIGEN -> CP DESTINO)?")
    for ((ori, dst), count) in final["top_viajes"][:5]:
        print(f"   - {ori} -> {dst}: {count} viajes")
