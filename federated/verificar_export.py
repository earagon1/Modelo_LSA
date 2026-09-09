"""
Verifica un `lsa_samples.json` exportado desde la app.

Es la comprobacion que cierra el circuito: la app compila y los tests pasan,
pero eso no prueba que el archivo que sale del telefono tenga lo que hace
falta. Este script lo abre y dice si sirve.

Chequea:

  - Que el esquema sea el esperado (t, d, by_date) y las secuencias tengan la
    forma correcta.
  - Que cada muestra traiga `client_id`. Sin ese campo el dataset no se puede
    particionar por persona y el experimento federado no pasa de sintetico.
  - Cuantas muestras aporto cada persona y de que clases, que es lo que define
    si los clientes van a tener datos suficientes.

No necesita TensorFlow, asi que arranca en segundos.

Uso:
    python federated/verificar_export.py ruta/al/lsa_samples.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MINIMO_RECOMENDADO = 20  # muestras por persona para que entrenar local tenga sentido


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("archivo", type=Path, help="El lsa_samples.json exportado desde la app")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.archivo.exists():
        print(f"[ERROR] No existe {args.archivo}")
        return 1

    try:
        datos = json.loads(args.archivo.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[ERROR] El archivo no es JSON valido: {e}")
        return 1

    print("=" * 68)
    print(f"VERIFICACION DE {args.archivo.name}")
    print("=" * 68)

    problemas = []

    t = datos.get("t")
    d = datos.get("d")
    print(f"t (frames por secuencia): {t}")
    print(f"d (features por frame):   {d}")
    if t != 15 or d != 126:
        problemas.append(f"Se esperaba t=15 y d=126, vino t={t} d={d}")

    by_date = datos.get("by_date")
    if not isinstance(by_date, dict) or not by_date:
        print("[ERROR] No hay muestras (falta 'by_date' o esta vacio)")
        return 1

    total = 0
    sin_client = 0
    forma_mala = 0
    por_cliente = Counter()
    clases_por_cliente = defaultdict(Counter)
    por_fecha = Counter()

    for fecha, items in by_date.items():
        for item in items:
            total += 1
            por_fecha[fecha] += 1

            seq = item.get("seq")
            if not isinstance(seq, list) or len(seq) != t or any(len(f) != d for f in seq):
                forma_mala += 1

            cid = item.get("client_id")
            if not cid:
                sin_client += 1
                cid = "(sin client_id)"
            por_cliente[cid] += 1
            clases_por_cliente[cid][item.get("label", "?")] += 1

    print(f"\nFechas de grabacion: {len(by_date)}")
    for fecha in sorted(por_fecha):
        print(f"  {fecha}: {por_fecha[fecha]} muestras")
    print(f"\nTotal de muestras: {total}")

    if forma_mala:
        problemas.append(f"{forma_mala} muestras con secuencias de forma incorrecta")

    # ------------------------------------------------------- el campo clave
    print("\n" + "-" * 68)
    print("ATRIBUCION POR PERSONA")
    print("-" * 68)
    if sin_client == total:
        problemas.append(
            "NINGUNA muestra tiene client_id. El export es de una version de la "
            "app anterior al campo, o se grabo sin sesion iniciada."
        )
    elif sin_client:
        problemas.append(f"{sin_client} de {total} muestras no tienen client_id")

    for cid, n in por_cliente.most_common():
        clases = clases_por_cliente[cid]
        aviso = ""
        if cid == "(sin client_id)":
            aviso = "  <-- no sirve para particionar"
        elif n < MINIMO_RECOMENDADO:
            aviso = f"  <-- pocas muestras (menos de {MINIMO_RECOMENDADO})"
        print(f"  {cid:28s} {n:4d} muestras, {len(clases)} clases{aviso}")
        detalle = ", ".join(f"{k}:{v}" for k, v in sorted(clases.items()))
        print(f"    {detalle}")

    personas = [c for c in por_cliente if c != "(sin client_id)"]

    # ------------------------------------------------------------ veredicto
    print("\n" + "=" * 68)
    if problemas:
        print("PROBLEMAS")
        for p in problemas:
            print(f"  - {p}")
        print()

    if not personas:
        print("NO SIRVE todavia para el experimento federado por persona.")
        print("Revisar que la app este actualizada y que se grabe con sesion iniciada.")
    elif len(personas) == 1:
        print(f"El campo client_id funciona ({personas[0]}).")
        print("Falta grabar mas personas: con una sola no hay nada que federar.")
    else:
        print(f"LISTO: {len(personas)} personas distintas, {total - sin_client} muestras atribuidas.")
        print("Se puede usar data.partition_by_client() para el experimento real.")

    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
