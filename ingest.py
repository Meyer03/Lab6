"""Ingesta: transcripciones de juntas → turnos → chunks con metadatos → (después) ChromaDB.

Uso:
    python ingest.py --dry-run      # genera chunks SIN embeddings → revisa chunks_preview.jsonl
    python ingest.py --dry-run --archivo "150926"   # solo un archivo, sin embeddings
    python ingest.py                  # indexa todo en ChromaDB (reanudable: se salta lo ya indexado)
    python ingest.py --archivo "150926"   # reemplaza los chunks de ese archivo (tras re-auditar)
    python ingest.py --reset          # borra la colección y reindexa todo (si cambió el chunking)

Flujo:
    1. catalogo.yaml dice de qué UN y qué junta es cada archivo.
    2. leer_transcript() (del proyecto de Arturo) convierte .txt/.docx/.vtt en texto.
    3. partir_en_turnos() reconoce quién habla y en qué minuto (5 formatos distintos).
    4. limpiar_turnos() (limpieza.py) corrige, normaliza nombres/áreas y quita ruido.
    5. armar_chunks() junta turnos consecutivos hasta ~MAX_PALABRAS, sin partir un turno
       salvo que él solo sea demasiado largo.
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import yaml

from limpieza import limpiar_turnos
from rag_agent.rag_core import AUDITORIA_DIR, PROJECT_DIR, TRANSCRIPTS_DIR, get_collection

# Reutilizamos el lector del proyecto de Arturo: así ambos sistemas leen igual los archivos.
sys.path.insert(0, str(AUDITORIA_DIR / "src"))
from procesar_junta import leer_transcript  # noqa: E402

MAX_PALABRAS = 220   # e5-small lee ~300 palabras; dejamos margen para el encabezado
MIN_PALABRAS = 60    # un chunk más corto que esto se une con el siguiente
LIMITE_DURO = 270    # tope absoluto (+ ~15 palabras de encabezado = < 300)
EXTENSIONES = {".txt", ".docx", ".vtt"}
BATCH = 50           # chunks por collection.add

MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7,
         "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Catálogo: ¿de qué UN y junta es cada archivo?
# ─────────────────────────────────────────────────────────────────────────────

def _normalizar(texto: str) -> str:
    """Minúsculas y forma Unicode estándar (macOS guarda 'Ó' distinto que Windows)."""
    return unicodedata.normalize("NFC", texto).lower()


def cargar_catalogo() -> list[dict]:
    return yaml.safe_load((PROJECT_DIR / "catalogo.yaml").read_text(encoding="utf-8"))["reglas"]


def buscar_regla(nombre_archivo: str, reglas: list[dict]) -> dict | None:
    nombre = _normalizar(nombre_archivo)
    for regla in reglas:
        if _normalizar(regla["patron"]) in nombre:
            return regla
    return None


def detectar_fecha(nombre_archivo: str, texto: str) -> str | None:
    """Busca la fecha primero en el nombre del archivo y luego en el encabezado del texto."""
    candidatos = [
        (r"(20\d{2})(\d{2})(\d{2})_\d{6}", "ymd"),        # Teams: ...-20260915_090203-...
        (r"(20\d{2})-(\d{2})-(\d{2})", "ymd"),            # arranque_Dedicados_2026-07-17
        (r"(\d{2})_(\d{2})_(20\d{2})", "dmy"),            # kpi_semanal_02_09_2026
        (r"(\d{2})(\d{2})(\d{2})\.docx$", "dmy2"),        # Stand Up_Sesión Diaria 150926.docx
    ]
    encabezado = texto[:600]
    for fuente in (nombre_archivo, encabezado):
        for patron, orden in candidatos:
            m = re.search(patron, fuente)
            if not m:
                continue
            a, b, c = m.groups()
            if orden == "ymd":
                return f"{a}-{b}-{c}"
            if orden == "dmy":
                return f"{c}-{b}-{a}"
            return f"20{c}-{b}-{a}"
    # "15 de septiembre de 2026" en el encabezado
    m = re.search(r"(\d{1,2}) de ([a-záéíóú]+) de (20\d{2})", _normalizar(encabezado))
    if m and m.group(2) in MESES:
        return f"{m.group(3)}-{MESES[m.group(2)]:02d}-{int(m.group(1)):02d}"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Turnos: ¿quién habla y en qué minuto?
# ─────────────────────────────────────────────────────────────────────────────
# Formatos encontrados en transcripts/:
#   A. Teams .docx   "Michelle Barcenas (TE, CAL, OP)   0:05"   ← el texto viene en las líneas siguientes
#   B. txt manual    "[04:15] Carlos (Transfer / Operaciones): texto"
#   C. txt corchetes "[Grace / Moderador]: texto"
#   D. txt simple    "Hablante 1 (Finanzas): texto"   o   "Michel: texto"
#   E. marca de hora "[00:00] Bloque 1: Reporte de Seguridad..."  ← no es un turno, solo actualiza el minuto

RE_TEAMS = re.compile(r"^(?P<hablante>\S.{0,80}?)\s{2,}(?P<min>\d{1,2}:\d{2}(?::\d{2})?)$")
RE_TS_HABLANTE = re.compile(r"^\[(?P<min>\d{1,2}:\d{2}(?::\d{2})?)\]\s*(?P<hablante>[^:\[\]]{2,80}?):\s*(?P<texto>.*)$")
RE_CORCHETES = re.compile(r"^\[(?P<hablante>[^\]\d][^\]]{1,60})\]:\s*(?P<texto>.*)$")
RE_SIMPLE = re.compile(r"^(?P<hablante>[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ .()/,&-]{0,60}?):\s+(?P<texto>\S.*)$")
RE_MARCA = re.compile(r"^\[(?P<min>\d{1,2}:\d{2}(?::\d{2})?)\]\s*(?P<resto>.*)$")

# Palabras que parecen "Nombre:" pero no son hablantes (títulos de sección en los txt)
NO_HABLANTES = {"fecha", "división", "division", "objetivo", "fuente", "propósito", "proposito",
                "origen", "nota", "bloque", "tema"}


def _a_segundos(minuto: str | None) -> int | None:
    if not minuto:
        return None
    partes = [int(p) for p in minuto.split(":")]
    return partes[0] * 3600 + partes[1] * 60 + partes[2] if len(partes) == 3 else partes[0] * 60 + partes[1]


def _formato_min(segundos: int | None) -> str:
    if segundos is None:
        return ""
    return f"{segundos // 60:02d}:{segundos % 60:02d}"


def _separar_rol(hablante: str) -> tuple[str, str]:
    """'Carlos (Transfer / Operaciones)' → ('Carlos', 'Transfer / Operaciones')."""
    hablante = hablante.strip(" []")
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", hablante)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    if " / " in hablante:  # "[Grace / Moderador]"
        nombre, rol = hablante.split(" / ", 1)
        return nombre.strip(), rol.strip()
    return hablante, ""


def partir_en_turnos(texto: str) -> list[dict]:
    """Convierte el texto en una lista de turnos: {hablante, rol, segundo, texto}."""
    turnos: list[dict] = []
    actual: dict | None = None
    segundo_vigente: int | None = None  # para formatos donde la hora solo aparece de vez en cuando

    def abrir(hablante: str, segundo: int | None, texto_inicial: str = "", exacto: bool = True):
        # exacto=False: la hora se heredó de una marca anterior, no es de este turno
        nonlocal actual
        nombre, rol = _separar_rol(hablante)
        actual = {"hablante": nombre, "rol": rol, "segundo": segundo, "exacto": exacto, "lineas": []}
        if texto_inicial.strip():
            actual["lineas"].append(texto_inicial.strip())
        turnos.append(actual)

    for linea in texto.splitlines():
        linea = linea.strip()
        if not linea or set(linea) <= set("=-_*"):
            continue
        if linea.lower().endswith("inició la transcripción") or linea.lower().endswith("detuvo la transcripción"):
            continue

        if m := RE_TEAMS.match(linea):                                        # formato A
            segundo_vigente = _a_segundos(m["min"])
            abrir(m["hablante"], segundo_vigente)
        elif m := RE_TS_HABLANTE.match(linea):                                # formato B
            nombre = m["hablante"].split("(")[0].strip().lower()
            if nombre.split()[0] in NO_HABLANTES:                             # "[00:00] Bloque 1: ..."
                segundo_vigente = _a_segundos(m["min"])
                continue
            segundo_vigente = _a_segundos(m["min"])
            abrir(m["hablante"], segundo_vigente, m["texto"])
        elif m := RE_MARCA.match(linea):                                      # formato E
            segundo_vigente = _a_segundos(m["min"])
        elif m := RE_CORCHETES.match(linea):                                  # formato C
            abrir(m["hablante"], segundo_vigente, m["texto"], exacto=False)
        elif (m := RE_SIMPLE.match(linea)) and m["hablante"].split()[0].lower() not in NO_HABLANTES:
            abrir(m["hablante"], segundo_vigente, m["texto"], exacto=False)   # formato D
        elif actual is not None:
            actual["lineas"].append(linea)                                    # continuación del turno
        # líneas antes del primer turno (título, fecha, duración) se ignoran

    for t in turnos:
        t["texto"] = " ".join(t.pop("lineas"))
    return [t for t in turnos if t["texto"]]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Chunks: juntar turnos hasta ~MAX_PALABRAS
# ─────────────────────────────────────────────────────────────────────────────

def _partir_turno_largo(turno: dict) -> list[dict]:
    """Si un solo turno pasa de MAX_PALABRAS, lo corta en fin de oración."""
    oraciones = []
    for oracion in re.split(r"(?<=[.!?])\s+", turno["texto"]):
        palabras = oracion.split()
        # la transcripción automática a veces no pone puntos: cortamos por palabras
        for i in range(0, len(palabras), MAX_PALABRAS):
            oraciones.append(" ".join(palabras[i:i + MAX_PALABRAS]))
    pedazos, buffer = [], []
    for oracion in oraciones:
        if buffer and len(" ".join(buffer + [oracion]).split()) > MAX_PALABRAS:
            pedazos.append(" ".join(buffer))
            buffer = []
        buffer.append(oracion)
    if buffer:
        pedazos.append(" ".join(buffer))
    return [{**turno, "texto": p} for p in pedazos]


def armar_chunks(turnos: list[dict], info: dict) -> list[dict]:
    """Agrupa turnos consecutivos en chunks. info = {source, un, junta, fecha}."""
    # los turnos ya vienen numerados ("n") desde build_chunks, antes de la limpieza
    piezas = [p for t in turnos for p in _partir_turno_largo(t)]

    def palabras_linea(p):
        # contamos también la etiqueta "Nombre (Área, Sede):" porque el modelo la lee
        etiqueta = f"{p['hablante']} {p.get('area', '')} {p.get('sede', '')}"
        return len(p["texto"].split()) + len(etiqueta.split())

    grupos, grupo, palabras = [], [], 0
    for p in piezas:
        n = palabras_linea(p)
        lleno = palabras + n > MAX_PALABRAS and palabras >= MIN_PALABRAS
        if grupo and (lleno or palabras + n > LIMITE_DURO):
            grupos.append(grupo)
            grupo, palabras = [], 0
        grupo.append(p)
        palabras += n
    if grupo:
        ultimo = sum(palabras_linea(p) for p in grupos[-1]) if grupos else 0
        if grupos and palabras < MIN_PALABRAS and ultimo + palabras <= LIMITE_DURO:
            grupos[-1].extend(grupo)  # el último pedacito se pega al anterior
        else:
            grupos.append(grupo)

    chunks = []
    for i, g in enumerate(grupos):
        # quién habló más en este chunk
        conteo = Counter()
        for p in g:
            conteo[p["hablante"]] += len(p["texto"].split())
        principal = conteo.most_common(1)[0][0]
        de_principal = next(p for p in g if p["hablante"] == principal)
        areas = sorted({p["area"] for p in g if p["area"]})

        primero = next((p for p in g if p["segundo"] is not None), None)
        minuto = _formato_min(primero["segundo"]) if primero else ""
        exacto = bool(primero and primero["exacto"])
        if minuto and exacto:
            cita = f"[{info['source']}, {minuto}]"
        elif minuto:   # hora aproximada (heredada de una marca de bloque) → también damos el turno
            cita = f"[{info['source']}, ~{minuto}, turno {g[0]['n']}]"
        else:          # el formato no trae horas
            cita = f"[{info['source']}, turno {g[0]['n']}]"

        lineas = []
        for p in g:
            etiqueta = ", ".join(x for x in (p["area"], p["sede"]) if x)
            quien = f"{p['hablante']} ({etiqueta})" if etiqueta else p["hablante"]
            lineas.append(f"{quien}: {p['texto']}")
        cuerpo = "\n".join(lineas)
        encabezado = (f"Junta {info['junta'].replace('_', ' ')} | UN: {info['un']} | "
                      f"Fecha: {info['fecha']} | Minuto: {(minuto if exacto else '~' + minuto) if minuto else 'n/d'}")
        chunks.append({
            "id": f"{info['un']}_{info['junta']}_{info['fecha']}_{i:03d}",  # determinista
            "text": f"{encabezado}\n{cuerpo}",
            "metadata": {
                "source": info["source"],
                "un": info["un"],
                "junta": info["junta"],
                "fecha": info["fecha"],
                "fecha_num": int(info["fecha"].replace("-", "")),  # para filtrar rangos: $gte / $lte
                "minuto": minuto,                                   # "" si el formato no trae hora
                "minuto_exacto": exacto,                            # False = aproximado
                "turno_inicio": g[0]["n"],
                "hablante_principal": principal,
                "area_principal": de_principal["area"],             # "Mantenimiento", "Transfer / Operaciones"...
                "sede_principal": de_principal["sede"],             # solo en Teams
                "areas": "; ".join(areas),
                "hablantes": "; ".join(conteo),                     # Chroma no acepta listas
                "cita": cita,
                "palabras": sum(len(p["texto"].split()) for p in g),
            },
        })
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 4. Juntar todo
# ─────────────────────────────────────────────────────────────────────────────

def build_chunks(ruta: Path, reglas: list[dict]) -> list[dict]:
    regla = buscar_regla(ruta.name, reglas)
    if regla is None:
        print(f"⚠️  {ruta.name}: no está en catalogo.yaml → se omite")
        return []
    if regla.get("excluir"):
        print(f"⏭️  {ruta.name}: excluido en catalogo.yaml")
        return []

    texto = leer_transcript(ruta)
    fecha = str(regla["fecha"]) if regla.get("fecha") else detectar_fecha(ruta.name, texto)
    if not fecha:
        print(f"⚠️  {ruta.name}: no encontré la fecha → agrégala en catalogo.yaml")
        return []

    turnos = partir_en_turnos(texto)
    for i, t in enumerate(turnos, start=1):
        t["n"] = i                      # numeramos ANTES de limpiar: la cita "turno N" apunta al original
    total_turnos = len(turnos)
    turnos, descartados = limpiar_turnos(turnos)
    info = {"source": ruta.name, "un": regla["un"], "junta": regla["junta"], "fecha": fecha}
    chunks = armar_chunks(turnos, info)

    con_minuto = sum(1 for t in turnos if t["segundo"] is not None and t["exacto"])
    print(f"✅ {ruta.name}: {regla['un']} · {regla['junta']} · {fecha} → "
          f"{total_turnos} turnos (−{descartados} ruido, {con_minuto} con minuto exacto) → {len(chunks)} chunks")
    return chunks


def index(chunks: list[dict], collection) -> None:
    """Agrega los chunks a ChromaDB por lotes. Reanudable: se salta los ids que ya existen.

    Si se corta a la mitad (Ctrl+C, se apagó la Mac...), volver a correr el script
    continúa donde se quedó, porque los ids son deterministas.
    """
    existentes = set(collection.get(ids=[c["id"] for c in chunks], include=[])["ids"]) if chunks else set()
    pendientes = [c for c in chunks if c["id"] not in existentes]
    print(f"{len(existentes)} ya indexados · {len(pendientes)} por indexar")

    for i in range(0, len(pendientes), BATCH):
        lote = pendientes[i:i + BATCH]
        collection.add(
            ids=[c["id"] for c in lote],
            documents=[c["text"] for c in lote],
            metadatas=[c["metadata"] for c in lote],
        )
        print(f"  lote {i // BATCH + 1}: {min(i + BATCH, len(pendientes))}/{len(pendientes)}")
    print(f"✅ La colección tiene {collection.count()} chunks")


def borrar_fuentes(chunks: list[dict], collection) -> None:
    """Borra TODOS los chunks previos de los archivos que se van a reindexar.

    Se usa con --archivo: si una junta se re-audita o se corrigió el chunking de un archivo,
    el número de chunks puede cambiar; borrar por 'source' evita que queden chunks viejos.
    """
    for source in sorted({c["metadata"]["source"] for c in chunks}):
        viejos = collection.get(where={"source": source}, include=[])["ids"]
        if viejos:
            collection.delete(ids=viejos)
            print(f"🗑️  {source}: {len(viejos)} chunks viejos borrados")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no calcula embeddings")
    ap.add_argument("--archivo", help="procesa solo archivos cuyo nombre contenga este texto")
    ap.add_argument("--reset", action="store_true", help="borra TODA la colección y reindexa todo")
    args = ap.parse_args()

    reglas = cargar_catalogo()
    archivos = sorted(p for p in TRANSCRIPTS_DIR.iterdir() if p.suffix.lower() in EXTENSIONES)
    if args.archivo:
        archivos = [p for p in archivos if _normalizar(args.archivo) in _normalizar(p.name)]

    chunks = [c for ruta in archivos for c in build_chunks(ruta, reglas)]
    tamanos = sorted(c["metadata"]["palabras"] for c in chunks) or [0]
    print(f"\n{len(chunks)} chunks de {len(archivos)} archivos · palabras por chunk: "
          f"mín {tamanos[0]}, mediana {tamanos[len(tamanos) // 2]}, máx {tamanos[-1]}")

    ids = [c["id"] for c in chunks]
    if len(ids) != len(set(ids)):
        repetidos = [i for i, n in Counter(ids).items() if n > 1]
        print(f"❌ IDs repetidos (dos archivos con misma UN, junta y fecha): {repetidos[:5]}")

    if args.dry_run:
        with (PROJECT_DIR / "chunks_preview.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print("Dry run: revisa chunks_preview.jsonl. No se calcularon embeddings.")
        return

    collection = get_collection()
    if args.reset:
        ids = collection.get(include=[])["ids"]
        if ids:
            collection.delete(ids=ids)
        print(f"Colección vaciada ({len(ids)} chunks borrados).")
    elif args.archivo:
        borrar_fuentes(chunks, collection)   # reemplazo limpio de esos archivos
    index(chunks, collection)


if __name__ == "__main__":
    main()
