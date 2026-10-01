"""Verifica que el índice esté listo para el agente (adaptado del check_index.py del Lab 6).

Uso:
    python check_index.py
    python check_index.py --query "¿cuántas unidades hay en taller?"            # prueba una búsqueda
    python check_index.py --query "unidades en taller" --un Dedicados           # con filtro

Diferencias con el del lab (las transcripciones no tienen páginas):
    page (int)     → turno_inicio (int, empieza en 1): número de turno citable
    doc_type (str) → junta (str): arranque_operativo / kpis_semanales
"""

import argparse
from collections import Counter

from rag_agent.rag_core import DB_DIR, get_collection

REQUIRED = {"source": str, "turno_inicio": int, "junta": str, "cita": str}
DOMINIO = {"un", "fecha", "fecha_num", "area_principal", "hablante_principal"}


def check(desc, ok, hint=""):
    print(f"✅ {desc}" if ok else f"❌ {desc} — {hint}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", help="prueba una búsqueda")
    ap.add_argument("--un", help="filtra la búsqueda de prueba por UN (XB, Dedicados, Especializados)")
    args = ap.parse_args()

    print(f"Índice en: {DB_DIR} · embeddings: multilingual-e5-small\n")
    col = get_collection()
    n = col.count()
    if not check(f"la colección tiene chunks ({n})", n > 0, "corre python ingest.py"):
        return
    data = col.get(include=["metadatas", "documents", "embeddings"], limit=n)
    metas, docs = data["metadatas"], data["documents"]

    for key, typ in REQUIRED.items():
        check(f"todos los chunks tienen '{key}' ({typ.__name__})",
              all(isinstance(m.get(key), typ) for m in metas), f"agrega '{key}' a los metadatos")
    presentes = DOMINIO & set(metas[0])
    check(f"metadatos propios del dominio: {sorted(presentes)}", len(presentes) >= 1,
          "agrega un, fecha, área...")
    check("los turnos empiezan en 1", min(m["turno_inicio"] for m in metas) >= 1, "numera turnos desde 1")
    check(f"cada chunk tiene embedding ({len(data['embeddings'][0])} dimensiones)",
          len(data["embeddings"]) == n, "vuelve a correr ingest.py")

    words = [len(d.split()) for d in docs]
    check(f"tamaño de chunk razonable (prom. {sum(words) // n} palabras, máx. {max(words)})",
          40 <= sum(words) / n <= 300, "revisa armar_chunks()")
    long_chunks = sum(w > 300 for w in words)
    check(f"ningún chunk rebasa ~300 palabras ({long_chunks} lo hacen)", long_chunks == 0,
          "multilingual-e5-small solo lee 512 tokens: el resto del chunk se ignora")
    check(f"no hay textos duplicados ({n - len(set(docs))} duplicados)", len(set(docs)) >= 0.95 * n,
          "¿transcripciones duplicadas en transcripts/?")
    sin_minuto = sum(not m["minuto"] for m in metas)
    print(f"ℹ️  {n - sin_minuto} chunks con minuto · {sin_minuto} citan por turno (formato sin horas)")

    print(f"\nPor junta: {dict(Counter(m['junta'] for m in metas))}")
    print(f"Por UN:    {dict(Counter(m['un'] for m in metas))}")
    print("Chunks por archivo:")
    for src, c in Counter(m["source"] for m in metas).most_common():
        print(f"  {c:5d}  {src}")

    if args.query:
        where = {"un": args.un} if args.un else None
        res = col.query(query_texts=[args.query], n_results=3, where=where)
        print(f"\nBúsqueda de prueba: {args.query!r}" + (f" (UN = {args.un})" if args.un else ""))
        for meta, dist, doc in zip(res["metadatas"][0], res["distances"][0], res["documents"][0]):
            cuerpo = doc.split("\n", 1)[1] if "\n" in doc else doc
            print(f"  {meta['cita']} d={dist:.3f}\n     {cuerpo[:160]!r}")


if __name__ == "__main__":
    main()
