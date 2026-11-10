"""Códigos de participación del servidor federado (R8 de plan/federado_robustez.md).

Cada persona necesita un código para mandar ajustes o tomas. Se le da de alta con el
código seudónimo que muestra su app (Ajustes → Aprendizaje federado, «Tu código»), y
este script imprime un código de participación de 16 caracteres que la persona escribe
una vez en la app. El servidor guarda solo su sha256 en <dir>/participantes.json: si se
pierde, se da de alta de nuevo (el anterior deja de valer).

Uso, desde la raíz de modelo_LSA:
    python federated/alta_participante.py --dir federated/piloto --participante p_3f2a91c0
    python federated/alta_participante.py --dir federated/piloto --listar
    python federated/alta_participante.py --dir federated/piloto --revocar p_3f2a91c0
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seguridad as seg  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--participante", help="código seudónimo de la persona (el que muestra su app)")
    g.add_argument("--revocar", metavar="PARTICIPANTE")
    g.add_argument("--listar", action="store_true")
    args = ap.parse_args()
    aut = seg.AutenticadorCodigos(os.path.join(args.dir, "participantes.json"))
    if args.listar:
        print("\n".join(aut.participantes()) or "(nadie)")
    elif args.revocar:
        print("revocado" if aut.revocar(args.revocar) else f"{args.revocar} no estaba dado de alta")
    else:
        codigo = aut.alta(args.participante)
        print(f"Código de participación de {args.participante}:\n\n    {seg.codigo_para_mostrar(codigo)}\n\n"
              "Escribilo en la app. No se guarda en ningún lado: si se pierde, dalo de alta de nuevo.")


if __name__ == "__main__":
    main()
