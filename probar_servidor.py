import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVIDOR = Path(__file__).resolve().parent / "servidor_mcp.py"


async def main(consulta: str):
    params = StdioServerParameters(command=sys.executable, args=[str(SERVIDOR)])
    async with stdio_client(params) as (leer, escribir):
        async with ClientSession(leer, escribir) as sesion:
            info = await sesion.initialize()
            print(f"✅ Conectado a '{info.serverInfo.name}'\n")
            print("Menú de herramientas (lo único que ve el modelo):")
            for t in (await sesion.list_tools()).tools:
                linea = (t.description or "(sin descripción)").strip().splitlines()[0]
                params_ = ", ".join(t.inputSchema.get("properties", {}))
                print(f"  • {t.name}({params_})\n      {linea}")
            print(f"\nPrueba: search_documents(query={consulta!r}, k=2)")
            r = await sesion.call_tool("search_documents", {"query": consulta, "k": 2})
            print(r.content[0].text[:800])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))