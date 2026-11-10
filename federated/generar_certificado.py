"""Certificado TLS autofirmado del servidor federado (R7 de plan/federado_robustez.md).

Se genera una vez por directorio de trabajo, en <dir>/tls/: servidor.key (clave privada,
no sale de esta PC y está en el .gitignore) y servidor.crt. La app no lo valida contra
una autoridad sino por la huella de la clave pública, que este script imprime y el
servidor repite al arrancar (ver seguridad.py).

Regenerarlo cambia la huella: cada teléfono va a rechazar el servidor hasta que la
persona toque «Olvidar la huella» y confirme la nueva. Por eso pide --forzar.

Usa el `openssl` de la línea de comandos (viene con Git para Windows); Python solo no
puede crear certificados sin bibliotecas de afuera.

Uso, desde la raíz de modelo_LSA:
    python federated/generar_certificado.py --dir federated/piloto
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seguridad as seg  # noqa: E402


def generar(directorio: str, forzar: bool = False, dias: int = 3650) -> tuple[str, bytes]:
    """(ruta del certificado, huella). Clave EC P-256: corta y la soporta Android 7+."""
    openssl = shutil.which("openssl")
    if openssl is None:
        raise SystemExit("No encuentro openssl. Viene con Git para Windows: correlo desde Git Bash.")
    carpeta = os.path.join(directorio, "tls")
    cert, clave = os.path.join(carpeta, "servidor.crt"), os.path.join(carpeta, "servidor.key")
    if os.path.exists(cert) and not forzar:
        raise SystemExit(f"Ya hay un certificado en {cert}. Regenerarlo cambia la huella y los teléfonos lo "
                         "van a rechazar hasta que la confirmen de nuevo: usá --forzar si es lo que querés.")
    os.makedirs(carpeta, exist_ok=True)
    subprocess.run([openssl, "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
                    "-nodes", "-days", str(dias), "-subj", "/CN=senar-federado",
                    "-keyout", clave, "-out", cert], check=True, capture_output=True,
                   env={**os.environ, "MSYS_NO_PATHCONV": "1"})  # Git Bash convertiría "/CN=..." en una ruta
    with open(cert, encoding="ascii") as f:
        return cert, seg.huella(f.read())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--forzar", action="store_true", help="reemplazar el certificado que ya hay")
    args = ap.parse_args()
    cert, huella = generar(args.dir, args.forzar)
    print(f"Certificado: {cert}\nHuella: {seg.huella_para_mostrar(huella)}\n"
          f"(completa, base64: {seg.huella_base64(huella)})")


if __name__ == "__main__":
    main()
