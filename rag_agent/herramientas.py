"""Herramientas (tools) del agente. Son funciones de Python normales.

ADK usa el NOMBRE, los PARÁMETROS (con sus tipos) y el DOCSTRING de cada función para
explicarle al modelo cuándo y cómo usarla. Un docstring pobre = un agente que decide mal.

Dos fuentes de datos:
  - ChromaDB (lo que se DIJO en la junta)         → buscar_transcripciones
  - JSON de auditoría de Arturo (lo ESTRUCTURADO) → consultar_pendientes, consultar_calificacion
  - Catálogo de ambos                             → listar_juntas

SEGURIDAD: si la sesión trae state["un_permitida"] (se fija al iniciar sesión el jefe de UN),
todas las tools IGNORAN la UN que pida el modelo y usan solo esa. Así un jefe de Dedicados no
puede ver XB aunque se lo pida al agente. Sin ese valor (modo lab / adk web) se permiten todas.
"""

import json
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache

from .rag_core import AUDITORIAS_DIR, get_collection

try:
    from google.adk.tools.tool_context import ToolContext
except ImportError:  # permite probar las funciones sin ADK instalado
    ToolContext = object

UNS = ("XB", "Dedicados", "Especializados")
MAX_K = 10
# Iteración A del lab: cuántos fragmentos regresa la búsqueda si el modelo no pide otro número.
# Se cambia desde rag_agent/.env (K_BUSQUEDA=10) sin tocar código. Default original: 6.
K_BUSQUEDA = max(1, min(int(os.getenv("K_BUSQUEDA", "10")), MAX_K))
RE_JSON = re.compile(r"auditoria_(?P<junta>arranque_operativo|kpis_semanales)_(?P<un>[A-Za-z]+)_(?P<fecha>\d{4}-\d{2}-\d{2})\.json$")


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades internas (no son tools)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _coleccion():
    return get_collection()  # cargar el modelo de embeddings tarda: solo una vez


def _un_efectiva(un: str, tool_context) -> tuple[str, str]:
    """Regresa (un_a_usar, aviso). Aplica el permiso de la sesión si existe."""
    permitida = ""
    if tool_context is not None and hasattr(tool_context, "state"):
        permitida = tool_context.state.get("un_permitida", "") or ""
    un = _normalizar_un(un)
    if permitida:
        if un and un != permitida:
            return permitida, f"Solo tienes acceso a la UN {permitida}; se ignoró el filtro '{un}'."
        return permitida, ""
    return un, ""


def _normalizar_un(un: str) -> str:
    for opcion in UNS:
        if un and un.strip().lower() == opcion.lower():
            return opcion
    return ""


def _a_num(fecha: str) -> int | None:
    """'2026-09-15' → 20260915 (para comparar rangos). Vacío o inválido → None."""
    if fecha and re.fullmatch(r"\d{4}-\d{2}-\d{2}", fecha.strip()):
        return int(fecha.strip().replace("-", ""))
    return None


def _auditorias(un: str = "", junta: str = "", desde: str = "", hasta: str = ""):
    """Itera (meta, datos) de los JSON de auditoría que cumplan los filtros, del más viejo al más nuevo."""
    d, h = _a_num(desde), _a_num(hasta)
    archivos = []
    for ruta in AUDITORIAS_DIR.glob("auditoria_*.json"):
        m = RE_JSON.search(ruta.name)
        if not m:
            continue
        meta = m.groupdict()
        n = int(meta["fecha"].replace("-", ""))
        if (un and meta["un"] != un) or (junta and meta["junta"] != junta):
            continue
        if (d and n < d) or (h and n > h):
            continue
        archivos.append((meta["fecha"], meta, ruta))
    for _, meta, ruta in sorted(archivos, key=lambda x: x[0]):
        yield meta, json.loads(ruta.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────

def buscar_transcripciones(consulta: str, un: str = "", fecha_desde: str = "", fecha_hasta: str = "",
                           area: str = "", junta: str = "", k: int = 0,
                           tool_context: ToolContext = None) -> dict:
    """Busca en las TRANSCRIPCIONES de las juntas lo que se DIJO sobre un tema (búsqueda semántica).

    Úsala para preguntas sobre explicaciones, causas, cifras reportadas de viva voz, incidentes,
    problemas o comentarios ("¿qué dijeron de las llantas?", "¿por qué bajó el ingreso de XB?").
    NO la uses para listar compromisos/pendientes ni calificaciones: para eso hay otras tools.

    Args:
        consulta: qué buscar, en palabras parecidas a como se diría en la junta
            (p. ej. "unidades en taller por falta de refacciones").
        un: unidad de negocio: "XB", "Dedicados" o "Especializados". Vacío = todas.
        fecha_desde: fecha mínima "AAAA-MM-DD" (inclusive). Vacío = sin límite.
        fecha_hasta: fecha máxima "AAAA-MM-DD" (inclusive). Vacío = sin límite.
        area: filtra por área de quien habla (p. ej. "Mantenimiento", "Transfer", "Capital Humano").
            Coincidencia parcial, sin distinguir mayúsculas. Vacío = todas.
        junta: "arranque_operativo" (diaria) o "kpis_semanales". Vacío = ambas.
        k: cuántos fragmentos regresar (1 a 10). Déjalo en 0 para usar el valor por defecto.

    Returns:
        dict con "resultados": lista de fragmentos con su "cita" ([archivo, mm:ss] o [archivo, turno N]),
        un, fecha, junta, hablantes, áreas y el texto. Si no hay resultados, "resultados" viene vacío
        y "mensaje" lo explica.
    """
    un, aviso = _un_efectiva(un, tool_context)
    k = max(1, min(int(k or K_BUSQUEDA), MAX_K))

    condiciones = []
    if un:
        condiciones.append({"un": un})
    if junta in ("arranque_operativo", "kpis_semanales"):
        condiciones.append({"junta": junta})
    if (d := _a_num(fecha_desde)):
        condiciones.append({"fecha_num": {"$gte": d}})
    if (h := _a_num(fecha_hasta)):
        condiciones.append({"fecha_num": {"$lte": h}})
    where = None if not condiciones else condiciones[0] if len(condiciones) == 1 else {"$and": condiciones}

    # El filtro de área es por coincidencia parcial ("Mantenimiento" dentro de "Mantenimiento; Operaciones"):
    # Chroma no filtra subcadenas en metadatos, así que pedimos más resultados y filtramos aquí.
    pedir = min(k * 8, 60) if area else k
    col = _coleccion()
    res = col.query(query_texts=[consulta], n_results=min(pedir, max(col.count(), 1)), where=where)

    resultados = []
    for meta, dist, doc in zip(res["metadatas"][0], res["distances"][0], res["documents"][0]):
        if area and area.lower() not in (meta.get("areas", "") + " " + meta.get("area_principal", "")).lower():
            continue
        resultados.append({
            "cita": meta["cita"],
            "un": meta["un"],
            "fecha": meta["fecha"],
            "junta": meta["junta"],
            "hablantes": meta["hablantes"],
            "areas": meta.get("areas", ""),
            "distancia": round(dist, 3),                    # menor = más parecido
            "texto": doc.split("\n", 1)[1] if "\n" in doc else doc,  # sin el encabezado repetido
        })
        if len(resultados) == k:
            break

    salida = {"filtros_aplicados": {"un": un or "todas", "fecha_desde": fecha_desde, "fecha_hasta": fecha_hasta,
                                    "area": area, "junta": junta or "ambas"},
              "resultados": resultados}
    if not resultados:
        salida["mensaje"] = "No hay fragmentos que cumplan esos filtros. Prueba otra redacción o quita filtros."
    if aviso:
        salida["aviso"] = aviso
    return salida


def consultar_pendientes(un: str = "", fecha_desde: str = "", fecha_hasta: str = "", responsable: str = "",
                         area: str = "", solo_abiertos: bool = True, tool_context: ToolContext = None) -> dict:
    """Lista los COMPROMISOS/PENDIENTES que se levantaron en las juntas (quién se llevó qué), según la
    auditoría. Incluye el último estatus detectado en juntas posteriores.

    Úsala para: "¿qué pendientes tiene X?", "¿qué compromisos quedaron abiertos en Dedicados?",
    "¿se cumplió lo que prometió Mantenimiento?". Es más exacta y barata que buscar en transcripciones.

    Args:
        un: "XB", "Dedicados" o "Especializados". Vacío = todas.
        fecha_desde: fecha mínima de la junta donde se levantó, "AAAA-MM-DD". Vacío = sin límite.
        fecha_hasta: fecha máxima, "AAAA-MM-DD". Vacío = sin límite.
        responsable: nombre (o parte) del responsable. Vacío = todos.
        area: área responsable (o parte, p. ej. "Seguridad"). Vacío = todas.
        solo_abiertos: True = omite los que ya se detectaron como resueltos.

    Returns:
        dict con "pendientes" (máx. 25, los más recientes primero), cada uno con id, fecha de la junta,
        UN, descripción, responsable, área, cita textual, fecha_compromiso, estatus_original y
        ultimo_estatus (abierto / en_proceso / resuelto / sin_mencion) con la fecha en que se detectó.
    """
    un, aviso = _un_efectiva(un, tool_context)
    pendientes, seguimiento = {}, {}
    # el seguimiento de un pendiente aparece en juntas POSTERIORES: hay que leer todas las de esa UN
    for meta, datos in _auditorias(un=un):
        for p in datos.get("pendientes", []):
            pendientes[p["id"]] = {**p, "_un": meta["un"], "_fecha": meta["fecha"], "_junta": meta["junta"]}
        for s in datos.get("seguimiento_pendientes_previos", []):
            seguimiento[s["id"]] = {"estatus": s.get("estatus_detectado"), "fecha": meta["fecha"],
                                    "comentario": s.get("comentario", "")}

    d, h = _a_num(fecha_desde), _a_num(fecha_hasta)
    salida = []
    for pid, p in pendientes.items():
        n = int(p["_fecha"].replace("-", ""))
        if (d and n < d) or (h and n > h):
            continue
        if responsable and responsable.lower() not in (p.get("responsable") or "").lower():
            continue
        if area and area.lower() not in (p.get("area_responsable") or p.get("area") or "").lower():
            continue
        seg = seguimiento.get(pid)
        ultimo = seg["estatus"] if seg else p.get("estatus")
        if solo_abiertos and ultimo == "resuelto":
            continue
        salida.append({
            "id": pid, "fecha_junta": p["_fecha"], "un": p["_un"], "junta": p["_junta"],
            "descripcion": p.get("descripcion"), "responsable": p.get("responsable"),
            "area": p.get("area_responsable") or p.get("area"), "cita": p.get("cita"),
            "fecha_compromiso": p.get("fecha_compromiso"), "estatus_original": p.get("estatus"),
            "ultimo_estatus": ultimo, "detectado_el": seg["fecha"] if seg else p["_fecha"],
            "comentario_seguimiento": seg["comentario"] if seg else "",
        })
    salida.sort(key=lambda x: x["fecha_junta"], reverse=True)
    resultado = {"total": len(salida), "pendientes": salida[:25],
                 "conteo_por_estatus": dict(Counter(x["ultimo_estatus"] for x in salida))}
    if len(salida) > 25:
        resultado["nota"] = f"Se muestran 25 de {len(salida)}. Usa filtros para acotar."
    if not salida:
        resultado["mensaje"] = "No hay pendientes con esos filtros en las auditorías disponibles."
    if aviso:
        resultado["aviso"] = aviso
    return resultado


def consultar_calificacion(un: str, fecha: str = "", junta: str = "arranque_operativo",
                           tool_context: ToolContext = None) -> dict:
    """Da la CALIFICACIÓN de una junta según la auditoría: % de cumplimiento, semáforo, y qué indicadores
    del estándar NO se mencionaron o quedaron parciales, además de los puntos críticos sin dueño.

    Úsala para: "¿cómo salió la junta de XB del 23?", "¿qué indicadores no se cubrieron?",
    "¿cómo va la tendencia de cumplimiento de Dedicados?" (sin fecha = todas, para ver tendencia).

    Args:
        un: "XB", "Dedicados" o "Especializados".
        fecha: "AAAA-MM-DD" de la junta. Vacío = resumen de TODAS las juntas auditadas de esa UN (tendencia).
        junta: "arranque_operativo" (default) o "kpis_semanales".

    Returns:
        Con fecha: dict con porcentaje, semáforo, conteo Sí/Parcial/No, indicadores en No y en Parcial
        (con comentario) y puntos críticos. Sin fecha: lista de {fecha, porcentaje, semáforo} por junta.
    """
    un, aviso = _un_efectiva(un, tool_context)
    if not un:
        return {"mensaje": "Indica la UN: XB, Dedicados o Especializados."}

    auditorias = list(_auditorias(un=un, junta=junta, desde=fecha, hasta=fecha))
    if not auditorias:
        disponibles = [m["fecha"] for m, _ in _auditorias(un=un, junta=junta)]
        return {"mensaje": f"No hay auditoría de {junta} de {un}" + (f" del {fecha}" if fecha else "") + ".",
                "fechas_disponibles": disponibles}

    if not fecha:  # tendencia
        return {"un": un, "junta": junta, "tendencia": [
            {"fecha": m["fecha"], "porcentaje": round(100 * d["resumen"]["porcentaje_cumplimiento"], 1),
             "semaforo": d["resumen"]["semaforo"]} for m, d in auditorias]}

    meta, d = auditorias[-1]
    por_veredicto = defaultdict(list)
    for r in d.get("resultados", []):
        veredicto = r.get("veredicto_corregido") or r.get("mencionado")  # si el auditor corrigió, manda su veredicto
        if veredicto in ("No", "Parcial"):
            por_veredicto[veredicto].append({"indicador": r.get("indicador_texto"), "comentario": r.get("comentario")})
    salida = {
        "un": un, "fecha": meta["fecha"], "junta": junta,
        "porcentaje": round(100 * d["resumen"]["porcentaje_cumplimiento"], 1),
        "semaforo": d["resumen"]["semaforo"], "conteo": d["resumen"]["conteo"],
        "indicadores_no": por_veredicto["No"], "indicadores_parcial": por_veredicto["Parcial"],
        "puntos_criticos": [{"descripcion": p.get("descripcion"), "severidad": p.get("severidad"),
                             "mencionado_por": p.get("mencionado_por"), "atendido": p.get("atendido")}
                            for p in d.get("puntos_criticos", [])],
        "fuente": f"auditoria_{junta}_{un}_{meta['fecha']}.json",
    }
    if aviso:
        salida["aviso"] = aviso
    return salida


def listar_juntas(tool_context: ToolContext = None) -> dict:
    """Catálogo de lo que hay disponible: qué juntas (UN, tipo, fecha) están en las transcripciones y
    cuáles tienen auditoría, y qué áreas aparecen. Úsala cuando no sepas qué fechas/UN existen o
    para interpretar "la última junta", "la semana pasada", o para saber qué valores de área filtrar.

    Returns:
        dict con "transcripciones" (un, junta, fecha, archivo), "auditorias" (un, junta, fecha)
        y "areas" (las más frecuentes).
    """
    un, _ = _un_efectiva("", tool_context)
    col = _coleccion()
    metas = col.get(where={"un": un} if un else None, include=["metadatas"])["metadatas"]
    juntas = sorted({(m["un"], m["junta"], m["fecha"], m["source"]) for m in metas}, key=lambda x: x[2])
    areas = Counter(a for m in metas for a in m.get("areas", "").split("; ") if a)
    return {
        "transcripciones": [{"un": u, "junta": j, "fecha": f, "archivo": s} for u, j, f, s in juntas],
        "auditorias": [{"un": m["un"], "junta": m["junta"], "fecha": m["fecha"]}
                       for m, _ in _auditorias(un=un)],
        "areas": [a for a, _ in areas.most_common(25)],
    }
