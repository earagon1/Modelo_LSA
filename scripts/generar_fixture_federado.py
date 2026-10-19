"""Genera el contrato del federado por red local y su fixture de paridad (J-04).

  contract/federado_v1.json: endpoints, formatos y reglas que tienen que respetar el
      servidor (federated/servidor.py) y la app (federado/ en TraductorLSA).
  contract/fixtures/paridad_federado_v1.json: casos que Kotlin tiene que reproducir
      bit a bit: cuantización a 8 bits con uniformes inyectados, sha256 de los pesos y
      la codificación base64.

Después regenera SHA256SUMS. Hay que copiar `contract/` entero a TraductorLSA/contract/.

Uso, desde la raíz de modelo_LSA:
    python scripts/generar_fixture_federado.py
"""
import hashlib
import json
import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "federated"))
import protocolo as pr  # noqa: E402

CONTRATO = os.path.join(RAIZ, "contract")

especificacion = {
    "protocolo": pr.PROTOCOLO,
    "descripcion": "Federado por red local de SeÑAR: rondas FedAvg de la cabeza y reparto del modelo completo "
                   "(plan/federado_completo.md). HTTP/1.1 + JSON en la red local; sin internet.",
    "numeros": {"little_endian": True, "pesos": "float32 en base64",
                "delta": "int8 en base64 + escala float32 por tensor"},
    "cabeza": {"entrada": 128, "capas": [64, 64, "C"], "activaciones": ["relu", "relu", "softmax"],
               "dropout_entrada": 0.5, "l2": [0.001, 0.001, 0.0],
               "orden_tensores": ["W1 (128x64)", "b1", "W2 (64x64)", "b2", "W3 (64xC)", "b3"],
               "W": "matriz [entrada][salida], como el kernel de Keras"},
    "sha256_pesos": "sha256 de los tensores en orden, cada uno como float32 little-endian, concatenados",
    "cuantizacion": {
        "bits": pr.BITS, "niveles": pr.NIVELES,
        "escala": "float32(max|x|) / 127 en float32; si max|x| = 0, escala 0 y todos los q en 0",
        "redondeo": "s = float32(x / escala); q = floor(s) + (u < s - floor(s) ? 1 : 0); q = clamp(q, -127, 127)",
        "uniformes": "u ~ U[0,1) float32, uno por valor; en el fixture vienen dados",
        "descuantizar": "float32(q) * escala",
    },
    "rotacion": {
        "opcional": True,
        "tipo": pr.ROTACION,
        "cuando": "si el tensor trae 'rotacion': {tipo, semilla}, se rotó antes de cuantizar",
        "rotar": "x completado con ceros hasta n = potencia de 2; y = H(D·x) / sqrt(n), H = Walsh-Hadamard sin "
                 "normalizar, D = diag(signos(semilla, n))",
        "desrotar": "x = D·H(y) / sqrt(n), truncado al largo original",
        "signos": "splitmix64: estado_i = semilla + (i+1)·0x9E3779B97F4A7C15 (mod 2^64); z = estado_i; "
                  "z = (z ^ z>>30)·0xBF58476D1CE4E5B9; z = (z ^ z>>27)·0x94D049BB133111EB; z ^= z>>31; "
                  "signo = +1 si el bit 63 de z es 0, -1 si no",
        "datos": "los n valores rotados cuantizados (n = potencia de 2 del largo del tensor)",
        "semilla_por_tensor": "semilla base + índice del tensor",
        "por_defecto": "apagada: con las actualizaciones de la cabeza no baja el error (no tienen picos; "
                       "resultados_federado_limite_rotacion/)",
    },
    "limite_norma": {
        "donde": "servidor, al recibir",
        "regla": "si ||delta||_2 > limite, delta *= limite / ||delta||_2",
        "valor": 2.0,
    },
    "endpoints": {
        "GET /estado": {"respuesta": ["protocolo", "ronda", "version", "paquete", "muestras_aportadas", "recibidos",
                                      "quorum", "timeout_s", "segundos_restantes", "accuracy_test"]},
        "GET /modelo": {"respuesta": ["protocolo", "ronda", "version", "paquete", "clases", "arquitectura",
                                      "backbone_sha256", "pesos [{forma, datos}]", "sha256"],
                        "nota": "la app solo participa si backbone_sha256 es el de su paquete instalado"},
        "POST /update": {"cuerpo": ["protocolo", "ronda", "participant_id", "n_muestras (> 0)", "bits (8)",
                                    "tensores [{forma, escala, datos}]", "metricas_locales (opcional)"],
                         "delta": "pesos locales - pesos de GET /modelo de esa ronda",
                         "codigos": {"200": "aceptado, o duplicado (aceptado: false)", "400": "inválido",
                                     "409": "ronda vieja", "413": "más de 1 MB"}},
        "POST /muestras": {"cuerpo": ["protocolo", "participant_id", "muestras [senar-muestra-v2]"],
                           "reglas": "solo landmarks; uuid obligatorio; participant_id igual al que envía; "
                                     "etiquetas en minúsculas, sin tildes; hasta 32 MB",
                           "respuesta": ["aceptadas", "repetidas", "rechazadas", "total_aportadas"]},
        "GET /paquete": {"respuesta": ["protocolo", "paquete", "creado", "clases", "senas_nuevas",
                                       "archivos [{nombre, sha256, bytes}]", "backbone_sha256", "cabeza_sha256",
                                       "entrada", "uuids_test", "metricas", "controles"]},
        "GET /paquete/v<N>/<archivo>": {"respuesta": "los bytes del archivo; solo los listados en el manifiesto"},
    },
    "reglas": {
        "idempotencia": "un update por (participant_id, ronda); el segundo responde duplicado",
        "cierre_de_ronda": "con el quórum, o al vencer el timeout con al menos un update",
        "agregacion": "w_{r+1} = w_r + sum_k (n_k / n) * delta_k (FedAvg)",
        "control_de_calidad": "la cabeza agregada se publica solo si no baja más de la tolerancia en el test fijo",
        "paquete_nuevo": "las rondas pasan a la cabeza del paquete; los updates pendientes se descartan",
        "uuids_test": "las tomas del test fijo no se usan para entrenar ni se aportan",
        "app": "verifica cada sha256 antes de usar un archivo; conserva el modelo de fábrica para volver",
    },
}

rng = np.random.default_rng(20261018)
casos = []
for nombre, x in [("normal", rng.normal(0, 0.02, 300)), ("con_ceros", np.r_[np.zeros(20), rng.normal(0, 1, 40)]),
                  ("todo_cero", np.zeros(16)), ("un_valor", np.array([-0.5])),
                  ("matriz", rng.normal(0, 0.05, (8, 5))), ("extremos", np.array([1e-8, -3.0, 3.0, 2.999, -0.0]))]:
    x = np.asarray(x, np.float32)
    u = rng.random(x.shape, dtype=np.float32)
    q, escala = pr.cuantizar(x, u)
    casos.append({"nombre": nombre, "forma": list(x.shape), "x": x.ravel().tolist(), "uniformes": u.ravel().tolist(),
                  "q": q.ravel().astype(int).tolist(), "escala": escala,
                  "q_base64": pr.a_b64(q, np.int8),
                  "reconstruido": pr.descuantizar(q, escala).ravel().tolist()})
pesos = [rng.normal(0, 0.1, (4, 3)).astype(np.float32), rng.normal(0, 0.1, 3).astype(np.float32)]
rotaciones = []
for semilla, n in [(0, 13), (123456789, 64), (2 ** 40 + 7, 832)]:
    x = rng.normal(0, 0.05, n).astype(np.float32)
    signos = pr.signos(semilla, pr.potencia_de_2(n)).astype(int).tolist()
    rotaciones.append({"semilla": semilla, "largo": n, "signos": signos,
                       "x": x.tolist(), "rotado": pr.rotar(x, semilla).tolist(), "tolerancia": 1e-5})
fixture = {"protocolo": pr.PROTOCOLO, "cuantizacion": casos, "rotacion": rotaciones,
           "sha_pesos": {"tensores": [{"forma": list(w.shape), "datos": w.ravel().tolist()} for w in pesos],
                         "base64": [pr.a_b64(w, np.float32) for w in pesos], "sha256": pr.sha_pesos(pesos)}}

with open(os.path.join(CONTRATO, "federado_v1.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(especificacion, f, ensure_ascii=False, indent=1)
    f.write("\n")
with open(os.path.join(CONTRATO, "fixtures", "paridad_federado_v1.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(fixture, f, ensure_ascii=False, separators=(",", ":"))
    f.write("\n")

# SHA256SUMS fija el contenido de contract/ para el test de espejo de los dos repos.
lineas = []
for raiz, _, archivos in os.walk(CONTRATO):
    for a in sorted(archivos):
        if a == "SHA256SUMS":
            continue
        p = os.path.join(raiz, a)
        rel = os.path.relpath(p, CONTRATO).replace(os.sep, "/")
        with open(p, "rb") as fh:
            lineas.append(f"{hashlib.sha256(fh.read()).hexdigest()}  {rel}")
with open(os.path.join(CONTRATO, "SHA256SUMS"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(sorted(lineas, key=lambda s: s.split("  ")[1])) + "\n")
print(f"{len(casos)} casos de cuantización -> contract/. Copiar contract/ a TraductorLSA/contract/.")
