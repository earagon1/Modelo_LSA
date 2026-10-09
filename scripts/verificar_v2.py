"""Revisa uno o más `lsa_samples_v2.jsonl` exportados después de la sesión de captura.

Dice, por participante y por código de planilla, cuántas tomas hay por bloque, tanda
y clase, y marca lo que no coincide con el protocolo (12 señas × 4 tomas por tanda,
10 de "nada"). Además convierte cada muestra con el contrato v2, para confirmar que
el archivo sirve para entrenar. No escribe nada.

Uso:
    python scripts/verificar_v2.py ruta/lsa_samples_v2.jsonl [otro.jsonl ...]
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsa import contrato, dataset_v2

TOMAS_POR_TANDA = 4
TOMAS_NADA = 10
CLASES = 12


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archivos", nargs="+")
    args = ap.parse_args()

    problemas = []
    por_codigo = defaultdict(Counter)
    personas = defaultdict(set)
    dispositivos = Counter()
    fuera_de_sesion = Counter()
    n_total = 0
    for ruta in args.archivos:
        for m in dataset_v2.leer(ruta):
            n_total += 1
            dispositivos[m["dispositivo"]] += 1
            try:
                x = dataset_v2.secuencia(m)
                assert x.shape == (contrato.T, contrato.D)
            except Exception as e:  # se reporta, no se corta: se quiere ver todo el archivo
                problemas.append(f"{m['uuid']}: no se puede convertir con el contrato v2 ({e})")
            if not m.get("participant_id"):
                problemas.append(f"{m['uuid']}: sin participant_id (¿se grabó sin sesión iniciada?)")
            s = m.get("sesion")
            if s is None:
                fuera_de_sesion[m["label"]] += 1
                continue
            por_codigo[s["codigo"]][(s["bloque"], s["tanda"], m["label"])] += 1
            personas[s["codigo"]].add(m.get("participant_id"))

    print(f"{n_total} muestras en {len(args.archivos)} archivo(s)")
    print("Dispositivos:", dict(dispositivos))
    if fuera_de_sesion:
        print("Fuera de la sesión guiada:", dict(fuera_de_sesion))

    for codigo in sorted(por_codigo):
        conteo = por_codigo[codigo]
        print(f"\n== {codigo}  (participant_id: {', '.join(sorted(map(str, personas[codigo])))})")
        if codigo == "PRUEBA":
            print("   ensayo: se descarta al analizar")
            continue
        if len(personas[codigo]) > 1:
            problemas.append(f"{codigo}: más de una cuenta grabó con este código")
        for tanda in (1, 2):
            clases = {c: n for (b, t, c), n in conteo.items() if b == "A" and t == tanda}
            faltan = [c for c, n in clases.items() if n != TOMAS_POR_TANDA]
            print(f"   tanda {tanda}: {sum(clases.values())} tomas, {len(clases)} clases"
                  + (f"  <-- distinto de {TOMAS_POR_TANDA}: {faltan}" if faltan else ""))
            if len(clases) != CLASES or faltan:
                problemas.append(f"{codigo} tanda {tanda}: {len(clases)} clases, desparejas: {faltan}")
        nada = sum(n for (b, _, c), n in conteo.items() if b == "N")
        print(f"   nada: {nada}")
        if nada != TOMAS_NADA:
            problemas.append(f"{codigo}: {nada} tomas de 'nada' (se esperaban {TOMAS_NADA})")

    print()
    if problemas:
        print("PROBLEMAS")
        for p in problemas:
            print("  -", p)
        return 1
    print("OK: todo coincide con el protocolo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
