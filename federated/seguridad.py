"""TLS y autenticación del servidor federado (R7 y R8 de plan/federado_robustez.md).

Dos problemas que marcó la auditoría (X5): el federado iba por HTTP sin cifrar, y el
`participant_id` lo declaraba el cliente, así que un teléfono podía aportar a nombre
de otro.

TLS con huella de la clave pública. El certificado es autofirmado: en una red local no
hay dominio para pedir uno firmado, y la IP de la PC cambia con la Wi-Fi. La app no
confía en una autoridad sino en la huella (SHA-256 del SubjectPublicKeyInfo, lo mismo
que fija el pinning de OkHttp): la primera vez la muestra y la persona la compara con
la que imprime el servidor; después la recuerda y rechaza cualquier otra clave (como
SSH). La huella no cambia aunque cambie la IP, porque no depende del nombre.

Códigos de participación. `alta_participante.py` genera uno por persona; el servidor
guarda solo su sha256 en `participantes.json`. El teléfono lo manda en
`Authorization: Bearer` y el servidor deduce de ahí quién es: el `participant_id` del
cuerpo tiene que coincidir. Va detrás de una interfaz (`verificar(cabecera) ->
participant_id`) para que un despliegue en la nube la cambie por la verificación del
token de sesión de Clerk sin tocar el resto.

Solo biblioteca estándar: el servidor corre en cualquier PC.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import ssl
import tempfile
from datetime import datetime

# Sin I, O, 0 ni 1: se lee en la pantalla de la PC y se escribe en el teléfono.
ALFABETO = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
LARGO_CODIGO = 16  # 16 × 5 bits = 80 bits
BYTES_HUELLA_A_MOSTRAR = 10  # 80 bits para comparar a ojo; la app guarda los 32


# --- huella del certificado ---

def _tlv(der: bytes, i: int) -> tuple[int, int, int]:
    """(etiqueta, inicio del contenido, fin del contenido) del elemento DER en i."""
    etiqueta = der[i]
    largo = der[i + 1]
    j = i + 2
    if largo & 0x80:
        n = largo & 0x7F
        largo = int.from_bytes(der[j:j + n], "big")
        j += n
    return etiqueta, j, j + largo


def spki_de_certificado(der: bytes) -> bytes:
    """El SubjectPublicKeyInfo (DER) de un certificado X.509, sin bibliotecas de afuera.

    Certificate = SEQUENCE { tbsCertificate, ... } y tbsCertificate = SEQUENCE {
    [0] version (opcional), serialNumber, signature, issuer, validity, subject,
    subjectPublicKeyInfo, ... }: el SPKI es el sexto elemento después de la versión."""
    _, ini, _ = _tlv(der, 0)          # Certificate
    _, i, fin_tbs = _tlv(der, ini)    # tbsCertificate
    if der[i] == 0xA0:                # [0] version
        i = _tlv(der, i)[2]
    for _ in range(5):                # serial, signature, issuer, validity, subject
        i = _tlv(der, i)[2]
    etiqueta, _, fin = _tlv(der, i)
    if etiqueta != 0x30 or fin > fin_tbs:
        raise ValueError("no se encontró el SubjectPublicKeyInfo en el certificado")
    return der[i:fin]


def huella(pem: str) -> bytes:
    """SHA-256 del SubjectPublicKeyInfo: la misma cuenta que hace la app (Seguridad.kt)."""
    return hashlib.sha256(spki_de_certificado(ssl.PEM_cert_to_DER_cert(pem))).digest()


def huella_para_mostrar(h: bytes) -> str:
    """Los primeros 10 bytes en grupos de 4 caracteres: "3F2A 91C0 ...". Igual en la app."""
    t = h[:BYTES_HUELLA_A_MOSTRAR].hex().upper()
    return " ".join(t[i:i + 4] for i in range(0, len(t), 4))


def huella_base64(h: bytes) -> str:
    return base64.b64encode(h).decode("ascii")


def contexto_tls(cert: str, clave: str) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cert, clave)
    return ctx


# --- códigos de participación ---

def normalizar_codigo(codigo: str) -> str:
    """Lo que escribió la persona, sin guiones ni espacios y en mayúsculas."""
    return "".join(c for c in codigo.upper() if c not in " -")


def codigo_para_mostrar(codigo: str) -> str:
    c = normalizar_codigo(codigo)
    return "-".join(c[i:i + 4] for i in range(0, len(c), 4))


def _hash(codigo: str) -> str:
    return hashlib.sha256(normalizar_codigo(codigo).encode("ascii")).hexdigest()


class AutenticadorCodigos:
    """participant_id a partir de `Authorization: Bearer <código>`. Lee el archivo en cada
    pedido: dar de alta a alguien con el servidor andando no pide reiniciarlo."""

    def __init__(self, ruta: str):
        self.ruta = ruta

    def _leer(self) -> dict:
        if not os.path.exists(self.ruta):
            return {}
        with open(self.ruta, encoding="utf-8") as f:
            return json.load(f)

    def _escribir(self, datos: dict) -> None:
        # Escritura atómica: si se corta a la mitad, el archivo viejo queda entero.
        directorio = os.path.dirname(os.path.abspath(self.ruta))
        os.makedirs(directorio, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directorio, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.ruta)

    def participantes(self) -> list[str]:
        return sorted(self._leer())

    def alta(self, participante: str) -> str:
        """Un código nuevo para esa persona (reemplaza el anterior, si tenía). Se devuelve
        una sola vez: el servidor no lo guarda, solo su sha256."""
        if not participante or not all(c.isalnum() or c == "_" for c in participante):
            raise ValueError(f"participant_id {participante!r}: letras, números y guion bajo")
        codigo = "".join(secrets.choice(ALFABETO) for _ in range(LARGO_CODIGO))
        datos = self._leer()
        datos[participante] = {"sha256": _hash(codigo), "creado": datetime.now().isoformat(timespec="seconds")}
        self._escribir(datos)
        return codigo

    def revocar(self, participante: str) -> bool:
        datos = self._leer()
        if participante not in datos:
            return False
        del datos[participante]
        self._escribir(datos)
        return True

    def verificar(self, cabecera: str | None) -> str | None:
        """El participant_id dueño del código, o None si falta o no es válido."""
        if not cabecera or not cabecera.startswith("Bearer "):
            return None
        h = _hash(cabecera[len("Bearer "):])
        encontrado = None
        for pid, d in self._leer().items():
            # compare_digest contra todos, sin cortar al primero: el tiempo no dice cuál era.
            if hmac.compare_digest(h, d.get("sha256", "")):
                encontrado = pid
        return encontrado
