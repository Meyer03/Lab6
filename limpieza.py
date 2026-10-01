"""Etapa 3 — limpieza de turnos antes del chunking.

Todo lo que es DATO (correcciones, alias, códigos) vive en glosario.yaml.
Aquí solo está la lógica. ingest.py llama a limpiar_turnos() después de partir_en_turnos().

Qué hace, en orden:
  1. Resuelve "Hablante 2 (Pablo)" → Pablo. Normaliza el nombre del hablante ("Luis.Palacios" → "Luis Palacios") y aplica alias
     ("Careli" → "Karely Cortazar").
  2. Traduce los códigos de Teams "(TE, QRO, OP)" → sede = Querétaro, area = Operaciones.
  3. Limpia el texto: correcciones del glosario, muletillas sueltas ("eh") y palabras
     repetidas por tartamudeo ("de de" → "de").
  4. Marca como ruido los turnos que no aportan ("Okay.", "Gracias.", inglés inventado),
     EXCEPTO si traen un número o responden a una pregunta del turno anterior.
"""

import re
from functools import lru_cache
from pathlib import Path

import yaml

GLOSARIO = Path(__file__).resolve().parent / "glosario.yaml"

# Palabras frecuentes en inglés: la transcripción de Teams a veces "traduce" ruido a inglés
# ("That actions for the same.", "I have some fun.").
INGLES = {"the", "you", "i", "i'm", "have", "has", "some", "fun", "going", "there", "that", "this",
          "for", "same", "say", "said", "is", "it", "and", "of", "to", "in", "on", "my", "we", "what",
          "actions", "switch", "hello", "yes", "okay", "thank", "thanks", "so", "do", "not", "be", "are",
          "was", "with", "they", "he", "she", "about", "right", "here", "hi"}

RE_PALABRA = re.compile(r"[a-záéíóúüñ']+|\d+", re.IGNORECASE)
RE_REPETIDA = re.compile(r"\b([a-záéíóúüñ]+)(?:[\s,]+\1\b)+", re.IGNORECASE)  # "de de", "que, que"


@lru_cache(maxsize=1)
def cargar_glosario() -> dict:
    g = yaml.safe_load(GLOSARIO.read_text(encoding="utf-8"))
    g["_correcciones"] = [(re.compile(k, re.IGNORECASE), v) for k, v in (g.get("correcciones") or {}).items()]
    g["_muletillas_turno"] = {w.lower() for w in g.get("muletillas_turno", [])}
    dentro = "|".join(re.escape(w) for w in g.get("muletillas_dentro", []))
    g["_re_dentro"] = re.compile(rf"(?<!\w)(?:{dentro})(?!\w)[,.]?\s*", re.IGNORECASE) if dentro else None
    return g


# Palabras que, dentro de "Hablante 1 (Finanzas)", son un ÁREA y no un nombre
PALABRAS_AREA = {"finanzas", "comercial", "cobranza", "operaciones", "mantenimiento", "moderador",
                 "moderadora", "seguridad", "planeación", "facilitadora", "líder", "lider"}


def resolver_hablante_generico(nombre: str, rol: str) -> tuple[str, str]:
    """En las KPIs viene 'Hablante 2 (Pablo)': el nombre real está en el paréntesis.
    'Hablante 5 (Cristian / Presentador)' → ('Cristian', 'Presentador').
    'Hablante 1 (Finanzas)' → se queda igual: Finanzas es un área."""
    if not re.match(r"^Hablante \d+$", nombre) or not rol:
        return nombre, rol
    primero = re.split(r"\s*[/-]\s*", rol)[0].strip()
    if len(primero.split()) == 1 and primero.lower() not in PALABRAS_AREA:
        return primero, rol[len(primero):].strip(" /-")
    return nombre, rol


def normalizar_hablante(nombre: str) -> str:
    g = cargar_glosario()
    nombre = re.sub(r"\s+", " ", nombre.replace(".", " ")).strip()
    return (g.get("alias_hablantes") or {}).get(nombre, nombre)


def decodificar_rol(rol: str) -> tuple[str, str]:
    """'TE, QRO, OP' → ('Querétaro', 'Operaciones').  'Transfer / Operaciones' → ('', 'Transfer / Operaciones')."""
    g = cargar_glosario()
    if "," not in rol:                       # formato de los .txt: el rol ya viene en palabras
        return "", rol
    sedes, areas, ignorar = g.get("sedes", {}), g.get("areas", {}), set(g.get("ignorar_codigos", []))
    sede, area = "", ""
    for codigo in (c.strip().upper() for c in rol.split(",")):
        if codigo in ignorar:
            continue
        if codigo in sedes and not sede:
            sede = sedes[codigo]
        elif codigo in areas and not area:
            area = areas[codigo]
    return sede, area


def limpiar_texto(texto: str) -> str:
    g = cargar_glosario()
    for patron, reemplazo in g["_correcciones"]:
        texto = patron.sub(reemplazo, texto)
    if g["_re_dentro"]:
        texto = g["_re_dentro"].sub("", texto)
    texto = RE_REPETIDA.sub(r"\1", texto)
    return re.sub(r"\s+", " ", texto).strip()


def es_ruido(texto: str, turno_anterior: str = "") -> bool:
    """True si el turno no aporta información."""
    palabras = [p.lower() for p in RE_PALABRA.findall(texto)]
    if not palabras:
        return True
    if any(p.isdigit() for p in palabras):
        return False                                   # "Tres." / "Son 84" sí es dato
    if turno_anterior.rstrip().endswith("?"):
        return False                                   # "Sí." respondiendo una pregunta sí es dato
    g = cargar_glosario()
    if all(p in g["_muletillas_turno"] for p in palabras):
        return True                                    # "Okay, muchas gracias."
    if len(palabras) <= 10 and sum(p in INGLES for p in palabras) / len(palabras) >= 0.5:
        return True                                    # "That actions for the same."
    return False


def limpiar_turnos(turnos: list[dict]) -> tuple[list[dict], int]:
    """Limpia una lista de turnos (ya numerados con 'n'). Regresa (turnos_utiles, n_descartados)."""
    utiles, descartados, anterior = [], 0, ""
    for t in turnos:
        texto = limpiar_texto(t["texto"])
        if es_ruido(texto, anterior):
            descartados += 1
            continue                                   # el número de turno 'n' se conserva para citar
        hablante, rol = resolver_hablante_generico(t["hablante"], t["rol"])
        sede, area = decodificar_rol(rol)
        utiles.append({**t, "texto": texto, "hablante": normalizar_hablante(hablante), "rol": rol,
                       "sede": sede, "area": area})
        anterior = texto
    return utiles, descartados
