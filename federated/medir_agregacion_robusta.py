"""Ataques contra el federado y las defensas del servidor (R5 de plan/federado_robustez.md).

Amplía `medir_limite_y_rotacion.py`: aquella medía una sola ronda con FedAvg y el límite
de norma. Esta corre varias rondas, cuatro ataques y todas las estrategias de agregación
que tiene el servidor, en dos escenarios:

  a) las 3 personas del piloto (P1, P2, P3) + 1 atacante: lo que pasaría hoy;
  b) cada persona partida en 5 sub-clientes (15 honestos) + 1 o 3 atacantes: el
     escenario con muchos clientes donde Krum puede correr (pide 2f + 3).

Cada ronda, como en el teléfono: cada honesto baja la cabeza global, la entrena 2 épocas
con sus tomas (las de la sesión 1 que no están en el test fijo) y sube el delta. El
atacante ve los deltas honestos de la ronda (el peor caso) y manda:

  grande     ruido al azar con 50 veces la norma honesta mediana;
  invertido  −10 veces el delta honesto medio, con un n realista;
  n_inflado  lo mismo, declarando 10⁶ muestras (el ataque de la auditoría, X5);
  ruido_x2   ruido con el doble de la norma honesta: pasa por debajo del límite de 2,0.

Estrategias, aplicadas como en `servidor.py` (primero el límite de norma, después el
tope de n, después la agregación):

  fedavg_sin_defensa        sin límite ni tope;
  fedavg_limite             límite 2,0, sin tope;
  fedavg_limite_tope        límite 2,0 + tope 2000 (lo que tiene el servidor hoy);
  fedavg_limite_tope200     límite 2,0 + tope 200 (para la decisión D-R6);
  mediana, recortada, krum  límite 2,0 + la agregación robusta (no usan n).

`ninguno` (sin atacante) mide lo que cuesta cada defensa cuando nadie ataca. Se evalúa
con la sesión 2, que no entra al federado ni al control de calidad, y se reporta el
acierto de todas las tomas y el promedio entre personas con su desvío (auditoría,
ML-M1: el ± entre semillas no dice cuánto cambia de una persona a otra).

Uso, desde la raíz de modelo_LSA:
    python federated/medir_agregacion_robusta.py --dir federated/piloto
    python federated/medir_agregacion_robusta.py --dir federated/piloto --rapido   # prueba
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.append(AQUI)
import agregacion as ag  # noqa: E402
import protocolo as pr  # noqa: E402

LIMITE = 2.0  # el del servidor (servidor.LIMITE_NORMA); no se importa para no cargar http.server
N_INFLADO = 10 ** 6
ATAQUES = ("ninguno", "grande", "invertido", "n_inflado", "ruido_x2")
ESTRATEGIAS = {
    "fedavg_sin_defensa": {"agregacion": "fedavg", "limite": None, "tope": None},
    "fedavg_limite": {"agregacion": "fedavg", "limite": LIMITE, "tope": None},
    "fedavg_limite_tope": {"agregacion": "fedavg", "limite": LIMITE, "tope": pr.N_MAX_MUESTRAS},
    "fedavg_limite_tope200": {"agregacion": "fedavg", "limite": LIMITE, "tope": 200},
    "mediana": {"agregacion": "mediana", "limite": LIMITE, "tope": None},
    "recortada": {"agregacion": "recortada", "limite": LIMITE, "tope": None},
    "krum": {"agregacion": "krum", "limite": LIMITE, "tope": None},
}


# --- partes puras (tests/test_medir_agregacion.py) ---

def _al_azar_con_norma(formas, norma, rng):
    ruido = [rng.normal(0, 1, f).astype(np.float32) for f in formas]
    escala = np.float32(norma / pr.norma(ruido))
    return [x * escala for x in ruido]


def delta_atacante(tipo: str, honestos: list[tuple[list[np.ndarray], int]], rng) -> tuple[list[np.ndarray], int]:
    """(delta, n declarado) del atacante para esta ronda, viendo los deltas honestos."""
    normas = [pr.norma(d) for d, _ in honestos]
    mediana_norma = float(np.median(normas))
    n_tipico = int(np.median([n for _, n in honestos]))
    formas = [d.shape for d in honestos[0][0]]
    medio = [np.mean([d[i] for d, _ in honestos], axis=0).astype(np.float32) for i in range(len(formas))]
    if tipo == "grande":
        return _al_azar_con_norma(formas, 50 * mediana_norma, rng), n_tipico
    if tipo == "invertido":
        return [np.float32(-10) * x for x in medio], n_tipico
    if tipo == "n_inflado":
        return [np.float32(-10) * x for x in medio], N_INFLADO
    if tipo == "ruido_x2":
        return _al_azar_con_norma(formas, 2 * mediana_norma, rng), n_tipico
    raise ValueError(f"ataque {tipo!r}: los que hay son {', '.join(ATAQUES[1:])}")


def beta_para(k: int) -> float:
    """La media recortada descarta al menos 1 de cada punta: con beta = 0,2 y k = 4 no
    descartaría nada (floor(0,8) = 0) y sería la media."""
    return max(ag.BETA_POR_DEFECTO, 1.0 / k)


def combinar_ronda(pesos, updates, estrategia: dict, f: int = 1, m: int = 1):
    """La cabeza que publicaría el servidor con estos updates: límite de norma, tope de n
    y agregación, en el mismo orden que `servidor.EstadoServidor` (recibir y _cerrar).
    Devuelve (pesos nuevos, motivo si la estrategia no pudo correr). Krum sin clientes
    suficientes no publica: se queda con la cabeza que había, como el servidor."""
    procesados = []
    for delta, n in updates:
        d, _, _ = pr.recortar(delta, estrategia["limite"])
        procesados.append((d, pr.peso_efectivo(n, estrategia["tope"])[0]))
    nombre = estrategia["agregacion"]
    if nombre == "krum":
        motivo = ag.validar_krum(len(procesados), f, m)
        if motivo:
            return [np.asarray(w, np.float32) for w in pesos], motivo
    return pr.agregar(pesos, procesados, nombre, beta=beta_para(len(procesados)), f=f, m=m), None


# --- entrenamiento local (TF) ---

class Entrenador:
    """La cabeza del teléfono (igual que `cliente_emulado.entrenar_cabeza`), armada una
    sola vez: reconstruir y compilar el modelo en cada llamada hacía que la grilla
    tardara horas. Entre clientes se pone en cero el estado de Adam."""

    def __init__(self, n_entrada: int, n_clases: int):
        import tensorflow as tf

        from cliente_emulado import DROPOUT, LR
        from fedavg import reset_optimizer_state
        self.tf, self.reset, self.c = tf, reset_optimizer_state, n_clases
        entrada = tf.keras.Input((n_entrada,))
        x = tf.keras.layers.Dropout(DROPOUT)(entrada)
        l2 = tf.keras.regularizers.l2(0.001)
        x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=l2)(x)
        x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=l2)(x)
        salida = tf.keras.layers.Dense(n_clases, activation="softmax")(x)
        self.m = tf.keras.Model(entrada, salida)
        self.m.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="categorical_crossentropy")

    def delta(self, pesos, emb, y, epocas, semilla):
        self.tf.keras.utils.set_random_seed(semilla)
        self.m.set_weights(pesos)
        self.reset(self.m)
        self.m.fit(emb, self.tf.keras.utils.to_categorical(y, self.c), epochs=epocas, batch_size=8, verbose=0)
        return [(a - b).astype(np.float32) for a, b in zip(self.m.get_weights(), pesos, strict=True)]


def evaluar(pesos, emb, y, personas):
    acierto = pr.predecir(pesos, emb).argmax(1) == y
    por_persona = {p: float(np.mean(acierto[personas == p])) for p in sorted(set(personas.tolist()))}
    return {"acierto": float(np.mean(acierto)), "por_persona": por_persona,
            "media_personas": float(np.mean(list(por_persona.values()))),
            "desvio_personas": float(np.std(list(por_persona.values())))}


def correr(entrenador, pesos0, clientes, atacantes, ataque, estrategia, rondas, epocas, semilla, evaluacion):
    """Una corrida de `rondas` rondas. `clientes`: lista de (emb, y)."""
    pesos = [np.asarray(w, np.float32) for w in pesos0]
    rng = np.random.default_rng(semilla + 7919)
    f = max(1, atacantes)
    k = len(clientes) + (atacantes if ataque != "ninguno" else 0)
    m = max(1, k - 2 * f)  # Multi-Krum: promedia la mitad central, más o menos
    curva, motivos = [], set()
    for r in range(rondas):
        honestos = [(entrenador.delta(pesos, e, y, epocas, semilla * 1000 + 31 * r + i), len(y))
                    for i, (e, y) in enumerate(clientes)]
        updates = list(honestos)
        if ataque != "ninguno":
            updates += [delta_atacante(ataque, honestos, rng) for _ in range(atacantes)]
        pesos, motivo = combinar_ronda(pesos, updates, estrategia, f=f, m=m)
        if motivo:
            motivos.add(motivo)
        curva.append(evaluar(pesos, *evaluacion)["acierto"])
    final = evaluar(pesos, *evaluacion)
    final.update({"curva": curva, "no_corrio": sorted(motivos), "clientes": k, "krum_f": f, "krum_m": m,
                  "beta": beta_para(k)})
    return final, pesos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directorio del piloto (paquete con backbone y cabeza)")
    ap.add_argument("--semillas", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--semillas-b", nargs="+", type=int, default=[0, 1, 2],
                    help="semillas del escenario b (15 clientes: es el que más tarda)")
    ap.add_argument("--rondas", type=int, default=5)
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--sub-clientes", type=int, default=5)
    ap.add_argument("--rapido", action="store_true", help="1 semilla, 2 rondas: para ver que todo enlaza")
    args = ap.parse_args()
    if args.rapido:
        args.semillas, args.semillas_b, args.rondas = [0], [0], 2

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")  # modelos chicos: la CPU va más rápido que copiar a la GPU
    from data import partition_subclientes
    from servidor import leer_paquete, ultimo_paquete

    from lsa import dataset as ds
    from lsa.export import inferir_tflite

    n = ultimo_paquete(args.dir)
    man, pesos0, clases, _ = leer_paquete(args.dir, n)
    with open(os.path.join(args.dir, "paquetes", f"v{n}", "backbone.tflite"), "rb") as f:
        tfl = f.read()
    d = ds.cargar(incluir_nada=True, clases=clases)
    excluir = list(man.get("uuids_test", []))
    tel = (d.fuente == "telefono") & ~np.isin(d.uuid, excluir)
    emb = np.zeros((len(d.y), 128), np.float32)
    emb[tel] = inferir_tflite(tfl, d.X[tel]).astype(np.float32)
    b = ds.cargar(incluir_nada=True, bloques=("B",), clases=clases)
    b_tel = b.fuente == "telefono"
    evaluacion = (inferir_tflite(tfl, b.X[b_tel]).astype(np.float32), b.y[b_tel], b.persona[b_tel])
    print(f"paquete v{n} | entrenan {int(tel.sum())} tomas de la sesión 1 | evalúan {int(b_tel.sum())} de la "
          f"sesión 2: {dict(zip(*np.unique(evaluacion[2], return_counts=True), strict=True))}", flush=True)

    idx = np.where(tel)[0]
    por_persona = [(emb[idx[p]], d.y[idx[p]]) for p in partition_subclientes(d.persona[idx], d.y[idx], 1)[0]]
    partes, _ = partition_subclientes(d.persona[idx], d.y[idx], args.sub_clientes, seed=0)
    sub = [(emb[idx[p]], d.y[idx[p]]) for p in partes]
    escenarios = {
        "a_3_personas_1_atacante": (por_persona, 1, args.semillas),
        f"b_{len(sub)}_subclientes_1_atacante": (sub, 1, args.semillas_b),
        f"b_{len(sub)}_subclientes_3_atacantes": (sub, 3, args.semillas_b),
    }
    entrenador = Entrenador(128, len(clases))
    filas = []
    salida = os.path.join(RAIZ, "resultados_federado_robusto", date.today().isoformat() + ("_rapido" if args.rapido
                                                                                         else ""))
    os.makedirs(salida, exist_ok=True)
    t0 = time.time()
    for esc, (clientes, atacantes, semillas) in escenarios.items():
        for semilla in semillas:
            for ataque in ATAQUES:
                for nombre, estrategia in ESTRATEGIAS.items():
                    r, _ = correr(entrenador, pesos0, clientes, atacantes, ataque, estrategia, args.rondas,
                                  args.epocas, semilla, evaluacion)
                    filas.append({"escenario": esc, "semilla": semilla, "ataque": ataque, "estrategia": nombre, **r})
                    print(f"[{time.time() - t0:6.0f} s] {esc} s{semilla} {ataque:9s} {nombre:22s} "
                          f"{100 * r['acierto']:5.1f} %  (personas {100 * r['media_personas']:.1f} ± "
                          f"{100 * r['desvio_personas']:.1f}){'  NO CORRIÓ' if r['no_corrio'] else ''}", flush=True)
            with open(os.path.join(salida, "resultados.json"), "w", encoding="utf-8") as fh:
                json.dump({"config": vars(args), "paquete": n, "limite": LIMITE, "estrategias": ESTRATEGIAS,
                           "filas": filas}, fh, indent=1, ensure_ascii=False)
    escribir_resumen(salida, filas, args)
    print(f"\n-> {salida} ({(time.time() - t0) / 60:.1f} min)")


def resumir(filas):
    """{escenario: {ataque: {estrategia: (media, desvío entre semillas, media personas, desvío personas)}}}."""
    tabla = {}
    for f in filas:
        tabla.setdefault(f["escenario"], {}).setdefault(f["ataque"], {}).setdefault(f["estrategia"], []).append(f)
    out = {}
    for esc, por_ataque in tabla.items():
        for ataque, por_estr in por_ataque.items():
            for estr, fs in por_estr.items():
                out.setdefault(esc, {}).setdefault(ataque, {})[estr] = {
                    "acierto": float(np.mean([x["acierto"] for x in fs])),
                    "desvio_semillas": float(np.std([x["acierto"] for x in fs])),
                    "media_personas": float(np.mean([x["media_personas"] for x in fs])),
                    "desvio_personas": float(np.mean([x["desvio_personas"] for x in fs])),
                    "no_corrio": sorted({m for x in fs for m in x["no_corrio"]}),
                    "semillas": len(fs)}
    return out


def escribir_resumen(salida, filas, args):
    pct = lambda v: f"{100 * v:.1f}".replace(".", ",")  # noqa: E731
    r = resumir(filas)
    md = ["# Ataques contra el federado y defensas del servidor", "",
          f"{args.rondas} rondas, {args.epocas} épocas locales. Acierto sobre la sesión 2 (no entra al federado). "
          "Cada celda: media entre semillas ± desvío entre semillas; entre corchetes, el desvío entre personas.", ""]
    for esc, por_ataque in r.items():
        md += [f"## {esc}", "", "| Estrategia | " + " | ".join(ATAQUES) + " |",
               "|---|" + "---|" * len(ATAQUES)]
        for estr in ESTRATEGIAS:
            celdas = []
            for ataque in ATAQUES:
                v = por_ataque.get(ataque, {}).get(estr)
                if v is None:
                    celdas.append("")
                elif v["no_corrio"]:
                    celdas.append("no corre")
                else:
                    celdas.append(f"{pct(v['acierto'])} ± {pct(v['desvio_semillas'])} [{pct(v['desvio_personas'])}]")
            md.append(f"| {estr} | " + " | ".join(celdas) + " |")
        md.append("")
    with open(os.path.join(salida, "resumen.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
