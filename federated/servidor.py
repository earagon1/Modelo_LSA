"""Servidor del federado por red local (PY-F6, contrato `contract/federado_v1.json`).

Corre en la PC, en la misma Wi-Fi (o hotspot) que los teléfonos y sin internet.
Hace dos cosas:

1. Rondas federadas de la cabeza. Los clientes bajan la cabeza global (GET /modelo),
   la entrenan con sus datos y suben el delta a 8 bits (POST /update). Con el quórum,
   o al vencer el tiempo con al menos un update, el servidor agrega con FedAvg, evalúa
   la cabeza nueva con su test fijo y la publica solo si no empeora: un cliente con
   datos malos no puede romper el modelo de todos.
2. Reparto del modelo completo. Las apps pueden aportar sus tomas (POST /muestras,
   solo landmarks, con consentimiento). `federated/reentrenar.py` reentrena con ellas
   y deja un paquete nuevo en paquetes/v<N>/ (backbone + cabeza + calibración +
   manifiesto con sha256). El servidor lo detecta solo, lo reparte (GET /paquete) y
   las rondas siguen sobre la cabeza de ese paquete.

Todo queda en el directorio de trabajo: estado.json, cabezas/v<N>.json (cada cabeza
publicada), rondas.jsonl (una línea por ronda cerrada o paquete nuevo, para el
informe), muestras/<participante>.jsonl y paquetes/.

Solo biblioteca estándar + numpy: http.server alcanza para unos pocos teléfonos.

Uso, desde la raíz de modelo_LSA (después de federated/reentrenar.py --dir <dir>):
    python federated/servidor.py --dir federated/piloto --quorum 2 --timeout 600
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agregacion as ag
import protocolo as pr

# Límite a la norma L2 de cada update (propuesta, tabla de riesgos: "limitar la magnitud
# de actualizaciones"). 3 veces la mediana de los updates honestos del piloto (0,68-0,76:
# cabeza del paquete entrenada 2 épocas con las tomas de P1, P2 y P3), medido con
# federated/medir_limite_y_rotacion.py. Ningún update honesto lo alcanza; uno malicioso
# que invierte el aprendizaje (×10) pasa de bajar el test 10 puntos a bajarlo 3.
LIMITE_NORMA = 2.0
TAMANO_MAXIMO = 1 << 20  # 1 MB: un delta de la cabeza a 8 bits ocupa ~17 KB en base64
TAMANO_MAXIMO_MUESTRAS = 32 << 20  # una sesión entera de 154 tomas v2 ocupa ~1,3 MB
# Segundos que una conexión puede quedarse sin mandar nada. Sin esto, un cliente que
# anuncia un Content-Length y manda menos deja un hilo del servidor colgado para
# siempre (ML-M10). Corre por cada lectura, no por el pedido entero: 32 MB por una
# Wi-Fi lenta tardan, pero no se quedan 30 s sin mandar ni un byte.
TIMEOUT_CONEXION_S = 30
ESQUEMA_MUESTRA = "senar-muestra-v2"


def validar_muestra(m: dict, participante: str) -> str | None:
    """Motivo de rechazo de una toma aportada, o None. Las etiquetas siguen la regla de
    los nombres de clase (minúsculas, sin tildes): si no, cruzarían con words.json."""
    if m.get("esquema") != ESQUEMA_MUESTRA:
        return f"esquema {m.get('esquema')!r}"
    if not m.get("uuid"):
        return "falta uuid"
    if m.get("participant_id") != participante:
        return "participant_id distinto del que envía"
    etiqueta = m.get("label")
    if not isinstance(etiqueta, str) or not etiqueta or etiqueta != etiqueta.strip().lower() \
            or not all(c.isascii() and (c.isalnum() or c == " ") for c in etiqueta):
        return f"etiqueta {etiqueta!r}: minúsculas, sin tildes, palabras separadas por espacio"
    frames = m.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 120:
        return "frames fuera de rango"
    for f in frames:
        for mano in f.get("manos", []):
            if len(mano.get("lm", [])) != 63:
                return "una mano sin sus 21 landmarks"
    return None


def ultimo_paquete(directorio: str) -> int:
    """El número del último paquete completo (con manifiesto) en paquetes/, o 0.
    reentrenar.py escribe el manifiesto al final: un paquete a medio copiar no cuenta."""
    raiz = os.path.join(directorio, "paquetes")
    if not os.path.isdir(raiz):
        return 0
    nums = [int(n[1:]) for n in os.listdir(raiz) if n.startswith("v") and n[1:].isdigit()
            and os.path.exists(os.path.join(raiz, n, "manifiesto.json"))]
    return max(nums, default=0)


def leer_paquete(directorio: str, n: int):
    """(manifiesto, pesos de la cabeza, clases, test) del paquete v<n>."""
    carpeta = os.path.join(directorio, "paquetes", f"v{n}")
    with open(os.path.join(carpeta, "manifiesto.json"), encoding="utf-8") as f:
        manifiesto = json.load(f)
    with open(os.path.join(carpeta, "cabeza.json"), encoding="utf-8") as f:
        pesos, clases = pr.cabeza_desde_json(json.load(f))
    test = None
    if os.path.exists(os.path.join(carpeta, "test_servidor.npz")):
        z = np.load(os.path.join(carpeta, "test_servidor.npz"))
        test = (z["embeddings"], z["etiquetas"])
    return manifiesto, pesos, clases, test


def _arquitectura(pesos) -> dict:
    return {"entrada": int(pesos[0].shape[0]),
            "capas": [int(pesos[i].shape[1]) for i in range(0, len(pesos), 2)],
            "activaciones": ["relu"] * (len(pesos) // 2 - 1) + ["softmax"]}


class EstadoServidor:
    """Las rondas y los paquetes, sin HTTP. `reloj` se inyecta para testear el timeout."""

    def __init__(self, pesos, clases, backbone_sha, directorio, quorum=2, timeout_s=600.0,
                 test=None, tolerancia=0.01, reloj=time.time, reanudar=None, limite_norma=LIMITE_NORMA,
                 max_muestras=pr.N_MAX_MUESTRAS, agregacion="fedavg", beta=ag.BETA_POR_DEFECTO):
        if agregacion not in ag.ESTRATEGIAS:
            raise ValueError(f"agregación {agregacion!r}: las opciones son {', '.join(ag.ESTRATEGIAS)}")
        ag.descartados_por_lado(1, beta)  # valida beta al arrancar, no en la primera ronda
        self.agregacion = agregacion
        self.beta = beta
        self.pesos = [np.asarray(w, np.float32) for w in pesos]
        self.clases = list(clases)
        self.backbone_sha = backbone_sha
        self.dir = directorio
        self.quorum = quorum
        self.timeout_s = timeout_s
        self.test = test  # (embeddings, etiquetas) o None
        self.tolerancia = tolerancia
        self.limite_norma = limite_norma if limite_norma and limite_norma > 0 else None
        self.max_muestras = max_muestras if max_muestras and max_muestras > 0 else None
        self.reloj = reloj
        self.arquitectura = _arquitectura(self.pesos)
        self.ronda = reanudar["ronda"] if reanudar else 0
        self.version = reanudar["version"] if reanudar else 0
        self.paquete = reanudar["paquete"] if reanudar else 0
        self.manifiesto = None
        if reanudar:
            with open(os.path.join(directorio, "paquetes", f"v{self.paquete}", "manifiesto.json"),
                      encoding="utf-8") as f:
                self.manifiesto = json.load(f)
        self.pendientes: dict[str, dict] = {}
        self.inicio_ronda = reloj()
        self.lock = threading.Lock()
        self.acc_actual = self.evaluar(self.pesos)
        os.makedirs(os.path.join(directorio, "cabezas"), exist_ok=True)
        os.makedirs(os.path.join(directorio, "muestras"), exist_ok=True)
        self.uuids_aportados = self._uuids_aportados()
        self._guardar_version()
        self._guardar_estado()
        self.revisar_paquete()

    @classmethod
    def desde_paquetes(cls, directorio, **kw):
        """Arranca desde el último paquete del directorio (el caso normal del piloto)."""
        n = ultimo_paquete(directorio)
        if n == 0:
            raise SystemExit(f"No hay paquetes en {directorio}/paquetes: correr federated/reentrenar.py --dir ...")
        manifiesto, pesos, clases, test = leer_paquete(directorio, n)
        # Si el servidor se reinicia, sigue donde estaba: misma ronda y la cabeza que ya
        # mejoraron las rondas anteriores, mientras el paquete sea el mismo.
        previo = None
        ruta_estado = os.path.join(directorio, "estado.json")
        if os.path.exists(ruta_estado):
            with open(ruta_estado, encoding="utf-8") as f:
                previo = json.load(f)
            ruta_cabeza = os.path.join(directorio, "cabezas", f"v{previo['version']}.json")
            if previo.get("paquete") == n and os.path.exists(ruta_cabeza):
                with open(ruta_cabeza, encoding="utf-8") as f:
                    pesos, clases = pr.cabeza_desde_json(json.load(f))
            else:
                previo = None
        estado = cls(pesos, clases, manifiesto["backbone_sha256"], directorio, test=test,
                     reanudar=previo, **kw)
        return estado

    # --- consultas ---
    def formas(self):
        return [list(w.shape) for w in self.pesos]

    def modelo(self) -> dict:
        with self.lock:
            return {"protocolo": pr.PROTOCOLO, "ronda": self.ronda, "version": self.version,
                    "paquete": self.paquete, "clases": self.clases, "arquitectura": self.arquitectura,
                    "backbone_sha256": self.backbone_sha, "pesos": pr.codificar_pesos(self.pesos),
                    "sha256": pr.sha_pesos(self.pesos)}

    def estado(self) -> dict:
        with self.lock:
            return {"protocolo": pr.PROTOCOLO, "ronda": self.ronda, "version": self.version,
                    "paquete": self.paquete, "muestras_aportadas": len(self.uuids_aportados),
                    "recibidos": len(self.pendientes), "quorum": self.quorum, "timeout_s": self.timeout_s,
                    "segundos_restantes": (max(0.0, self.timeout_s - (self.reloj() - self._primer_update()))
                                           if self.pendientes else None),
                    "accuracy_test": self.acc_actual, "agregacion": self.agregacion}

    def evaluar(self, pesos) -> float | None:
        if self.test is None:
            return None
        emb, y = self.test
        return float(np.mean(pr.predecir(pesos, emb).argmax(1) == y))

    # --- rondas federadas ---
    def recibir(self, u: dict) -> tuple[int, dict]:
        """(código HTTP, respuesta). Idempotente por (participante, ronda)."""
        with self.lock:
            motivo = pr.validar_update(u, self.ronda, self.formas())
            if motivo is not None:
                codigo = 409 if motivo.startswith("ronda") else 400
                return codigo, {"aceptado": False, "motivo": motivo, "ronda": self.ronda}
            pid = u["participant_id"]
            if pid in self.pendientes:
                return 200, {"aceptado": False, "motivo": "duplicado: ya hay un update tuyo en esta ronda",
                             "ronda": self.ronda}
            delta, norma, recortado = pr.recortar(pr.decodificar_delta(u["tensores"]), self.limite_norma)
            n, n_recortado = pr.peso_efectivo(u["n_muestras"], self.max_muestras)
            self.pendientes[pid] = {"delta": delta, "n": n, "n_declarado": u["n_muestras"],
                                    "n_recortado": n_recortado, "norma": norma, "recortado": recortado,
                                    "rotado": any("rotacion" in t for t in u["tensores"]),
                                    "bytes": pr.bytes_payload(u["tensores"]),
                                    "metricas": u.get("metricas_locales", {}), "t": self.reloj()}
            respuesta = {"aceptado": True, "ronda": self.ronda, "recibidos": len(self.pendientes),
                         "quorum": self.quorum, "recortado": recortado, "n_muestras_usadas": n}
            if len(self.pendientes) >= self.quorum:
                respuesta["cierre"] = self._cerrar("quorum")
            return 200, respuesta

    def revisar_timeout(self) -> dict | None:
        """El plazo corre desde el primer update de la ronda, no desde que se abrió: una
        ronda puede quedar abierta horas sin nadie, y el primero que llega no tiene que
        cerrarla solo (pasó en el piloto del 2026-10-18)."""
        with self.lock:
            if self.pendientes and self.reloj() - self._primer_update() >= self.timeout_s:
                return self._cerrar("timeout")
            return None

    def _primer_update(self) -> float:
        return min(p["t"] for p in self.pendientes.values())

    def _cerrar(self, motivo: str) -> dict:
        updates = [(p["delta"], p["n"]) for p in self.pendientes.values()]
        candidata = pr.agregar(self.pesos, updates, self.agregacion, self.beta)
        acc = self.evaluar(candidata)
        publicada = acc is None or self.acc_actual is None or acc >= self.acc_actual - self.tolerancia
        fila = {"evento": "ronda", "ronda": self.ronda, "paquete": self.paquete, "motivo": motivo,
                "clientes": sorted(self.pendientes),
                "n_muestras": {k: v["n"] for k, v in self.pendientes.items()},
                # Lo que declaró cada cliente que pasó el tope: el informe necesita poder
                # mostrar cuándo alguien quiso pesar más de lo que le corresponde.
                "n_declarado_sobre_tope": {k: v["n_declarado"] for k, v in self.pendientes.items()
                                           if v["n_recortado"]},
                "max_muestras": self.max_muestras,
                "bytes_subidos": {k: v["bytes"] for k, v in self.pendientes.items()},
                "normas": {k: round(v["norma"], 4) for k, v in self.pendientes.items()},
                "recortados": sorted(k for k, v in self.pendientes.items() if v["recortado"]),
                "rotados": sorted(k for k, v in self.pendientes.items() if v["rotado"]),
                "limite_norma": self.limite_norma,
                "agregacion": self.agregacion,
                "beta": self.beta if self.agregacion == "recortada" else None,
                # Una ronda que cierra por tiempo puede tener menos clientes que el quórum:
                # queda escrito si la estrategia robusta no defendió nada en esa ronda.
                "advertencia_agregacion": (ag.advertencia(self.agregacion, len(updates), self.beta)
                                           if self.agregacion != "fedavg" else None),
                "metricas_locales": {k: v["metricas"] for k, v in self.pendientes.items()},
                "duracion_s": self.reloj() - self.inicio_ronda,
                "accuracy_antes": self.acc_actual, "accuracy_candidata": acc, "publicada": publicada}
        if publicada:
            self.pesos, self.acc_actual = candidata, acc
            self.version += 1
            self._guardar_version()
        fila["version"] = self.version
        self._registrar(fila)
        self.ronda += 1
        self.pendientes = {}
        self.inicio_ronda = self.reloj()
        self._guardar_estado()
        return {"ronda_cerrada": fila["ronda"], "publicada": publicada, "version": self.version,
                "accuracy": self.acc_actual}

    # --- tomas aportadas y paquetes ---
    def _uuids_aportados(self) -> set[str]:
        vistos = set()
        carpeta = os.path.join(self.dir, "muestras")
        for nombre in os.listdir(carpeta):
            with open(os.path.join(carpeta, nombre), encoding="utf-8") as f:
                vistos.update(json.loads(linea)["uuid"] for linea in f if linea.strip())
        return vistos

    def recibir_muestras(self, cuerpo: dict) -> tuple[int, dict]:
        """Guarda las tomas nuevas de un participante. Reenviar las mismas no duplica."""
        pid = cuerpo.get("participant_id")
        if cuerpo.get("protocolo") != pr.PROTOCOLO or not isinstance(pid, str) or not pid \
                or not all(c.isalnum() or c == "_" for c in pid):
            return 400, {"motivo": "protocolo o participant_id inválido"}
        muestras = cuerpo.get("muestras")
        if not isinstance(muestras, list):
            return 400, {"motivo": "falta la lista de muestras"}
        nuevas, repetidas, rechazadas = [], 0, []
        with self.lock:
            for m in muestras:
                motivo = validar_muestra(m, pid) if isinstance(m, dict) else "no es un objeto"
                if motivo:
                    rechazadas.append({"uuid": m.get("uuid") if isinstance(m, dict) else None, "motivo": motivo})
                elif m["uuid"] in self.uuids_aportados:
                    repetidas += 1
                else:
                    self.uuids_aportados.add(m["uuid"])
                    nuevas.append(m)
            if nuevas:
                with open(os.path.join(self.dir, "muestras", f"{pid}.jsonl"), "a", encoding="utf-8") as f:
                    for m in nuevas:
                        f.write(json.dumps(m, ensure_ascii=False) + "\n")
            total = len(self.uuids_aportados)
        return 200, {"aceptadas": len(nuevas), "repetidas": repetidas, "rechazadas": rechazadas,
                     "total_aportadas": total}

    def revisar_paquete(self) -> dict | None:
        """Si reentrenar.py dejó un paquete nuevo, pasa a repartirlo. Los updates
        pendientes se descartan: se calcularon sobre el backbone anterior."""
        n = ultimo_paquete(self.dir)
        with self.lock:
            if n <= self.paquete:
                return None
            manifiesto, pesos, clases, test = leer_paquete(self.dir, n)
            descartados = sorted(self.pendientes)
            self.pesos, self.clases, self.test = pesos, clases, test
            self.backbone_sha = manifiesto["backbone_sha256"]
            self.arquitectura = _arquitectura(pesos)
            self.paquete, self.manifiesto = n, manifiesto
            self.acc_actual = self.evaluar(pesos)
            self.version += 1
            self.pendientes = {}
            self.inicio_ronda = self.reloj()
            self._guardar_version()
            evento = {"evento": "paquete_nuevo", "paquete": n, "version": self.version, "ronda": self.ronda,
                      "clases": clases, "accuracy_test": self.acc_actual, "updates_descartados": descartados}
            self._registrar(evento)
            self._guardar_estado()
            return evento

    def archivo_de_paquete(self, n: int, nombre: str) -> str | None:
        """Ruta de un archivo listado en el manifiesto del paquete v<n>; nada más se sirve."""
        try:
            manifiesto, *_ = leer_paquete(self.dir, n)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        if nombre not in {a["nombre"] for a in manifiesto["archivos"]}:
            return None
        return os.path.join(self.dir, "paquetes", f"v{n}", nombre)

    # --- persistencia ---
    def _registrar(self, fila: dict):
        fila = {"t": self.reloj(), **fila}
        with open(os.path.join(self.dir, "rondas.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")

    def _guardar_version(self):
        capas = [{"W": self.pesos[i].tolist(), "b": self.pesos[i + 1].tolist()} for i in range(0, len(self.pesos), 2)]
        with open(os.path.join(self.dir, "cabezas", f"v{self.version}.json"), "w", encoding="utf-8") as f:
            json.dump({"version": self.version, "paquete": self.paquete, "clases": self.clases,
                       "sha256": pr.sha_pesos(self.pesos), "accuracy_test": self.acc_actual, "capas": capas}, f)

    def _guardar_estado(self):
        with open(os.path.join(self.dir, "estado.json"), "w", encoding="utf-8") as f:
            json.dump({"ronda": self.ronda, "version": self.version, "paquete": self.paquete,
                       "quorum": self.quorum, "timeout_s": self.timeout_s, "accuracy_test": self.acc_actual},
                      f, indent=1)


def crear_servidor(estado: EstadoServidor, host="0.0.0.0", puerto=8765) -> ThreadingHTTPServer:
    class Manejador(BaseHTTPRequestHandler):
        # StreamRequestHandler lo aplica al socket; handle_one_request atrapa el
        # TimeoutError y cierra la conexión.
        timeout = TIMEOUT_CONEXION_S

        def _responder(self, codigo, cuerpo):
            datos = json.dumps(cuerpo, ensure_ascii=False).encode("utf-8")
            self.send_response(codigo)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(datos)))
            self.end_headers()
            self.wfile.write(datos)

        def _archivo(self, ruta):
            with open(ruta, "rb") as f:
                datos = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(datos)))
            self.end_headers()
            self.wfile.write(datos)

        def do_GET(self):
            estado.revisar_paquete()
            estado.revisar_timeout()
            partes = self.path.strip("/").split("/")
            if self.path == "/modelo":
                self._responder(200, estado.modelo())
            elif self.path == "/estado":
                self._responder(200, estado.estado())
            elif self.path == "/paquete":
                if estado.manifiesto is None:
                    self._responder(404, {"motivo": "todavía no hay paquete publicado"})
                else:
                    self._responder(200, estado.manifiesto)
            elif len(partes) == 3 and partes[0] == "paquete" and partes[1][:1] == "v" and partes[1][1:].isdigit():
                ruta = estado.archivo_de_paquete(int(partes[1][1:]), partes[2])
                if ruta is None:
                    self._responder(404, {"motivo": "no existe"})
                else:
                    self._archivo(ruta)
            else:
                self._responder(404, {"motivo": "no existe"})

        def do_POST(self):
            estado.revisar_paquete()
            estado.revisar_timeout()
            limites = {"/update": TAMANO_MAXIMO, "/muestras": TAMANO_MAXIMO_MUESTRAS}
            if self.path not in limites:
                self._responder(404, {"motivo": "no existe"})
                return
            # Antes: int(... , 0). Sin la cabecera leía un cuerpo vacío; con una no numérica
            # se caía el hilo, y con una negativa rfile.read(-1) esperaba al fin de la
            # conexión. La app y el cliente emulado siempre la mandan (largo fijo).
            texto = self.headers.get("Content-Length")
            if texto is None:
                self._responder(411, {"aceptado": False, "motivo": "falta Content-Length"})
                return
            try:
                largo = int(texto)
            except ValueError:
                largo = -1
            if largo < 0:
                self._responder(400, {"aceptado": False, "motivo": "Content-Length inválido"})
                return
            if largo > limites[self.path]:
                self._responder(413, {"aceptado": False, "motivo": f"más de {limites[self.path]} bytes"})
                return
            try:
                cuerpo = json.loads(self.rfile.read(largo))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._responder(400, {"aceptado": False, "motivo": "JSON inválido"})
                return
            if not isinstance(cuerpo, dict):
                self._responder(400, {"aceptado": False, "motivo": "se esperaba un objeto JSON"})
                return
            if self.path == "/update":
                codigo, respuesta = estado.recibir(cuerpo)
            else:
                codigo, respuesta = estado.recibir_muestras(cuerpo)
            self._responder(codigo, respuesta)

        def log_message(self, formato, *args):
            sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {formato % args}\n")

    return ThreadingHTTPServer((host, puerto), Manejador)


def ip_local() -> str:
    """La IP de esta PC en la red local, para escribirla en la app."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # no manda nada: solo elige la interfaz
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directorio de trabajo del piloto")
    ap.add_argument("--cabeza", help="cabeza inicial (JSON de exportar_cabeza.py); sin esto, el último paquete")
    ap.add_argument("--backbone-sha", help="sha256 del backbone.tflite de la app (con --cabeza)")
    ap.add_argument("--test", help="npz con 'embeddings' y 'etiquetas' (con --cabeza)")
    ap.add_argument("--quorum", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--tolerancia", type=float, default=0.01)
    ap.add_argument("--limite-norma", type=float, default=LIMITE_NORMA,
                    help="norma L2 máxima de un update (0 = sin límite)")
    ap.add_argument("--max-muestras", type=int, default=pr.N_MAX_MUESTRAS,
                    help="peso máximo de un cliente en FedAvg; más que eso entra recortado (0 = sin tope)")
    ap.add_argument("--agregacion", choices=ag.ESTRATEGIAS, default="fedavg",
                    help="fedavg (ponderado por n), mediana o recortada (robustas, sin usar n)")
    ap.add_argument("--beta", type=float, default=ag.BETA_POR_DEFECTO,
                    help="fracción que la media recortada descarta de cada punta, en [0; 0,5)")
    ap.add_argument("--puerto", type=int, default=8765)
    args = ap.parse_args()

    opciones = {"limite_norma": args.limite_norma, "max_muestras": args.max_muestras,
                "agregacion": args.agregacion, "beta": args.beta}
    if args.cabeza:
        with open(args.cabeza, encoding="utf-8") as f:
            pesos, clases = pr.cabeza_desde_json(json.load(f))
        test = None
        if args.test:
            z = np.load(args.test)
            test = (z["embeddings"], z["etiquetas"])
        estado = EstadoServidor(pesos, clases, args.backbone_sha, args.dir, args.quorum, args.timeout,
                                test, args.tolerancia, **opciones)
    else:
        estado = EstadoServidor.desde_paquetes(args.dir, quorum=args.quorum, timeout_s=args.timeout,
                                               tolerancia=args.tolerancia, **opciones)
    if args.agregacion != "fedavg":
        aviso = ag.advertencia(args.agregacion, args.quorum, args.beta)
        if aviso:
            print(f"ATENCIÓN: {aviso}.", flush=True)
    servidor = crear_servidor(estado, puerto=args.puerto)

    def vigilar():
        while True:
            time.sleep(5)
            evento = estado.revisar_paquete()
            if evento:
                print(f"paquete nuevo publicado: v{evento['paquete']}", flush=True)
            cierre = estado.revisar_timeout()
            if cierre:
                print(f"ronda cerrada por tiempo: {cierre}", flush=True)
    threading.Thread(target=vigilar, daemon=True).start()
    print(f"Servidor federado en http://{ip_local()}:{args.puerto}  (paquete v{estado.paquete}, ronda "
          f"{estado.ronda}, accuracy de test {estado.acc_actual}). Escribí esa dirección en la app. "
          "Ctrl+C para cortar.", flush=True)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
