"""Publica un modelo en la app: el único camino para escribir sus assets (auditoría, X3).

Copia juntos, a `TraductorLSA/app/src/main/assets/`:
  actions_15_opt.tflite   el modelo (sin ops Flex)
  words.json              las clases; su orden es el índice de salida del modelo
  calibracion.json        temperatura y umbral, sellados con el sha del modelo
  modelo.json             el manifiesto que la app valida al cargar (nuevo)
y, si la corrida lo trae, `paridad_modelo.json` a `app/src/androidTest/assets/`.

Antes de copiar nada verifica lo que la app no puede verificar sola: que no haya ops
Flex, que la entrada sea (1, T, D) del contrato v2, que la salida tenga una neurona por
clase y que los nombres sigan la regla (minúsculas, sin tildes, con espacio). Si algo
falla no escribe ningún archivo. La app, al cargar, vuelve a comparar el sha, la versión
de features y la cantidad de clases contra `modelo.json`, y si no coinciden lo muestra
en vez de traducir con etiquetas equivocadas.

Antes los assets se copiaban a mano y nada impedía copiar un `.tflite` sin su
`words.json`: la app recortaba o rellenaba la salida y mostraba otra seña sin aviso.

Uso, desde la raíz de modelo_LSA:
    python scripts/publicar_modelo.py --desde runs/<fecha>_final \\
        --calibracion resultados_calibracion/calibracion_app.json [--origen "..."] [--app ../../TraductorLSA]
    python scripts/publicar_modelo.py --verificar      # revisa lo que hoy está publicado
"""
import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from lsa.contrato import FEATURES_VERSION, T  # noqa: E402

D = 126
MODELO = "actions_15_opt.tflite"
PALABRAS = "words.json"
CALIBRACION = "calibracion.json"
MANIFIESTO = "modelo.json"
PARIDAD = "paridad_modelo.json"
APP_POR_DEFECTO = os.path.normpath(os.path.join(RAIZ, "..", "..", "TraductorLSA"))
# La clase "no es una seña": no tiene imagen en el diccionario a propósito.
SIN_IMAGEN = {"nada"}
NOMBRE_VALIDO = re.compile(r"^[a-zñ]+( [a-zñ]+)*$")


def sha256(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def problemas_de_clases(clases: list) -> list[str]:
    """La regla de nombres del CLAUDE.md: minúsculas, sin tildes, palabras separadas por un espacio."""
    p = []
    if not clases:
        p.append("words.json no tiene clases")
    for c in clases:
        if not isinstance(c, str) or not NOMBRE_VALIDO.match(c):
            p.append(f"clase con nombre inválido: {c!r} (minúsculas, sin tildes, con espacio)")
    repetidas = sorted({c for c in clases if clases.count(c) > 1})
    if repetidas:
        p.append(f"clases repetidas: {repetidas}")
    return p


def problemas_del_modelo(forma_entrada: list, forma_salida: list, flex: list, n_clases: int) -> list[str]:
    p = []
    if flex:
        p.append(f"el modelo tiene ops Flex/custom {flex}: la app no los carga")
    if list(forma_entrada)[1:] != [T, D]:
        p.append(f"entrada {list(forma_entrada)}: el contrato v{FEATURES_VERSION} arma (1, {T}, {D})")
    if not forma_salida or forma_salida[-1] != n_clases:
        p.append(f"el modelo tiene {forma_salida[-1] if forma_salida else '?'} salidas y words.json {n_clases} clases")
    return p


def problemas_de_calibracion(cal: dict, sha_modelo: str) -> list[str]:
    p = []
    t = cal.get("temperatura")
    if not isinstance(t, (int, float)) or t <= 0:
        p.append(f"temperatura inválida: {t!r}")
    u = cal.get("umbral")
    if u is not None and not (isinstance(u, (int, float)) and 0 < u < 1):
        p.append(f"umbral inválido: {u!r}")
    if cal.get("modelo_sha256") not in (None, sha_modelo):
        p.append("la calibración dice que es de otro modelo (modelo_sha256 distinto)")
    return p


def armar_manifiesto(sha_modelo: str, clases: list, palabras: bytes, calibracion: bytes,
                     origen: str, fecha: str) -> dict:
    return {
        "modelo": MODELO, "modelo_sha256": sha_modelo,
        "features_version": FEATURES_VERSION, "entrada": [1, T, D], "n_salidas": len(clases),
        "clases": clases, "words_sha256": sha256(palabras), "calibracion_sha256": sha256(calibracion),
        "origen": origen, "fecha": fecha,
        "aviso": "Lo escribe scripts/publicar_modelo.py (modelo_LSA). No editar a mano: la app lo valida al cargar.",
    }


def problemas_publicados(assets: str) -> list[str]:
    """Que lo que hay en assets sea coherente entre sí (sin TensorFlow: lo que valida la app)."""
    try:
        man = json.load(open(os.path.join(assets, MANIFIESTO), encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"no se pudo leer {MANIFIESTO}: {e}"]
    p = []
    modelo = open(os.path.join(assets, MODELO), "rb").read()
    palabras = open(os.path.join(assets, PALABRAS), "rb").read()
    calibracion = open(os.path.join(assets, CALIBRACION), "rb").read()
    clases = json.loads(palabras)["word_ids"]
    if man.get("modelo_sha256") != sha256(modelo):
        p.append(f"{MODELO} no es el del manifiesto")
    if man.get("words_sha256") != sha256(palabras) or man.get("clases") != clases:
        p.append(f"{PALABRAS} no es el del manifiesto")
    if man.get("calibracion_sha256") != sha256(calibracion):
        p.append(f"{CALIBRACION} no es la del manifiesto")
    if man.get("features_version") != FEATURES_VERSION:
        p.append(f"features_version {man.get('features_version')!r}; el contrato es {FEATURES_VERSION!r}")
    if man.get("n_salidas") != len(clases):
        p.append("n_salidas no coincide con las clases")
    p += problemas_de_calibracion(json.loads(calibracion), sha256(modelo))
    return p


def avisos_de_diccionario(clases: list, assets: str) -> list[str]:
    """No impiden publicar: la app muestra la seña igual, sin foto en el diccionario."""
    faltan = [c for c in clases if c not in SIN_IMAGEN
              and not os.path.exists(os.path.join(assets, "dictionary", c.replace(" ", "_") + ".jpg"))]
    return [f"sin imagen en assets/dictionary: {faltan}"] if faltan else []


def forma_y_flex(modelo: bytes):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import tensorflow as tf

    from lsa.export import ops_flex
    interp = tf.lite.Interpreter(model_content=modelo)
    return (interp.get_input_details()[0]["shape"].tolist(), interp.get_output_details()[0]["shape"].tolist(),
            ops_flex(modelo))


def escribir(ruta: str, datos: bytes):
    tmp = ruta + ".tmp"
    with open(tmp, "wb") as f:
        f.write(datos)
    os.replace(tmp, ruta)


def publicar(desde: str, ruta_calibracion: str, app: str, origen: str, fecha: str, forma_y_flex=forma_y_flex) -> dict:
    """Valida todo y recién después escribe. Devuelve el manifiesto; si algo falla, ValueError."""
    modelo = open(os.path.join(desde, MODELO), "rb").read()
    palabras = open(os.path.join(desde, PALABRAS), "rb").read()
    clases = json.loads(palabras)["word_ids"]
    cal = json.load(open(ruta_calibracion, encoding="utf-8"))
    sha = sha256(modelo)

    entrada, salida, flex = forma_y_flex(modelo)
    problemas = problemas_de_clases(clases) + problemas_del_modelo(entrada, salida, flex, len(clases)) \
        + problemas_de_calibracion(cal, sha)
    paridad = os.path.join(desde, PARIDAD)
    if os.path.exists(paridad) and json.load(open(paridad, encoding="utf-8")).get("modelo_sha256") != sha:
        problemas.append(f"{PARIDAD} es de otro modelo")
    if problemas:
        raise ValueError("no se publicó nada:\n  - " + "\n  - ".join(problemas))

    # La calibración queda sellada con el modelo con el que viaja: la app la descarta si no coincide.
    cal_app = {"temperatura": cal["temperatura"], "umbral": cal.get("umbral"), "modelo_sha256": sha,
               "origen": cal.get("origen", os.path.relpath(ruta_calibracion, RAIZ).replace(os.sep, "/"))}
    calibracion = (json.dumps({k: v for k, v in cal_app.items() if v is not None}, indent=2, ensure_ascii=False)
                   + "\n").encode("utf-8")
    man = armar_manifiesto(sha, clases, palabras, calibracion, origen, fecha)

    assets = os.path.join(app, "app", "src", "main", "assets")
    escribir(os.path.join(assets, MODELO), modelo)
    escribir(os.path.join(assets, PALABRAS), palabras)
    escribir(os.path.join(assets, CALIBRACION), calibracion)
    # El manifiesto va último: si algo se cortara antes, la app ve el desajuste y avisa.
    escribir(os.path.join(assets, MANIFIESTO),
             (json.dumps(man, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    if os.path.exists(paridad):
        escribir(os.path.join(app, "app", "src", "androidTest", "assets", PARIDAD), open(paridad, "rb").read())
    return man


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desde", help=f"carpeta con {MODELO} y {PALABRAS} (por ejemplo runs/<fecha>_final)")
    ap.add_argument("--calibracion", help="calibracion_app.json de scripts/calibrar.py")
    ap.add_argument("--origen", help="de dónde sale el modelo (va al manifiesto)")
    ap.add_argument("--app", default=APP_POR_DEFECTO, help="raíz de TraductorLSA")
    ap.add_argument("--verificar", action="store_true", help="solo revisar lo publicado")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    assets = os.path.join(args.app, "app", "src", "main", "assets")

    if args.verificar:
        p = problemas_publicados(assets)
        for a in avisos_de_diccionario(json.load(open(os.path.join(assets, PALABRAS), encoding="utf-8"))["word_ids"],
                                       assets):
            print("aviso:", a)
        print("\n".join(f"PROBLEMA: {x}" for x in p) or "Lo publicado es coherente.")
        sys.exit(1 if p else 0)

    if not args.desde or not args.calibracion:
        ap.error("--desde y --calibracion son obligatorios (o --verificar)")
    try:
        man = publicar(args.desde, args.calibracion, args.app,
                       args.origen or os.path.relpath(args.desde, RAIZ).replace(os.sep, "/"), date.today().isoformat())
    except ValueError as e:
        print(e)
        sys.exit(1)
    for a in avisos_de_diccionario(man["clases"], assets):
        print("aviso:", a)
    print(f"Publicado {MODELO} {man['modelo_sha256'][:12]}… con {man['n_salidas']} clases en {assets}")
    print("Anotar en plan/progreso.md y commitear los assets en TraductorLSA.")


if __name__ == "__main__":
    main()
