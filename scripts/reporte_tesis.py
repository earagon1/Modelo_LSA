"""Arma el documento de resultados para el informe de la tesis (PY-H1).

Junta en `resultados_tesis/` (Markdown + tablas LaTeX booktabs) lo que hoy sostiene
el capítulo de resultados, siempre desde los JSON de las corridas, nunca a mano:
  1. La comparativa de escenarios de evaluación (validar.py): desde el modelo original
     de la app hasta el esquema viejo, con media ± desvío entre semillas.
  2. La fuga del esquema viejo, cuantificada contra la validación por persona.
  3. Accuracy por persona y F1 por seña: modelo original contra el nuevo (LOPO).
  4. La comparativa de la sesión de captura por dispositivo (resumen_captura.py).
  5. El federado con personas reales y la calibración (sus corridas más reciente).
  6. La robustez por condición de la sesión 2 (robustez.py, su corrida más reciente).

Uso, desde la raíz de modelo_LSA:
    python scripts/reporte_tesis.py resultados_validacion_v2/<corrida>
"""
import glob
import json
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "resultados_tesis")

ESCENARIOS = [
    ("publicado", "Modelo original de la app", "solo webcam, contrato v1; sobre las tomas reales de los teléfonos"),
    ("webcam_a_telefono", "Solo webcam, contrato v2", "mismo dato de entrenamiento, preprocesamiento corregido"),
    ("persona", "Persona no vista (LOPO)", "entrena sin la persona evaluada: la cifra honesta"),
    ("sesion", "Sesión no vista", "agrupado por sesión de grabación"),
    ("estratificado", "Esquema anterior", "tomas mezcladas al azar: mide la fuga, no la generalización"),
]


def pct(v):
    return f"{100 * v:.1f}".replace(".", ",")


def fila_escenario(r):
    s = r.get("por_semilla")
    if s and len(s["semillas"]) > 1:
        return (f"{pct(s['accuracy_media'])} ± {pct(s['accuracy_desvio'])}",
                f"{pct(s['f1_media'])} ± {pct(s['f1_desvio'])}")
    return pct(r["global"]["accuracy"]), pct(r["global"]["f1_macro"])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    corrida = sys.argv[1]
    with open(os.path.join(corrida, "resultados.json"), encoding="utf-8") as f:
        res = json.load(f)
    esq = res["esquemas"]
    semillas = res.get("semillas", [0])

    md = ["# Resultados para el informe", "",
          f"Generado por `scripts/reporte_tesis.py` desde `{os.path.relpath(corrida, RAIZ)}`. No editar a mano.", "",
          f"Datos: {res['datos']}. Semillas: {semillas}.", "",
          "## 1. Escenarios de evaluación", "",
          "| Escenario | Qué mide | Accuracy (%) | F1 macro (%) |", "|---|---|---|---|"]
    tex = [r"\begin{tabular}{llrr}", r"\toprule", r"Escenario & Qué mide & Accuracy (\%) & F1 macro (\%) \\",
           r"\midrule"]
    for clave, nombre, que in ESCENARIOS:
        if clave not in esq:
            continue
        a, f1 = fila_escenario(esq[clave])
        md.append(f"| {nombre} | {que} | {a} | {f1} |")
        pm = r"$\pm$"
        fin = r" \\"
        tex.append(f"{nombre} & {que} & {a.replace('±', pm)} & {f1.replace('±', pm)}{fin}")
    tex += [r"\bottomrule", r"\end{tabular}"]

    if "publicado" in esq:
        p = esq["publicado"]
        md += ["", f"El modelo original muestra el {pct(p['mostradas'])} % de sus predicciones (confianza ≥ "
               f"{p['umbral_app']}) y de esas acierta el {pct(p['accuracy_de_las_mostradas'])} %: se equivoca "
               "con seguridad."]

    if "estratificado" in esq and "persona" in esq:
        acc = lambda k: esq[k].get("por_semilla", {}).get("accuracy_media", esq[k]["global"]["accuracy"])  # noqa: E731
        md += ["", "## 2. Fuga del esquema anterior", "",
               f"El esquema anterior da {pct(acc('estratificado'))} %; con una persona no vista, "
               f"{pct(acc('persona'))} %. La diferencia, **{pct(acc('estratificado') - acc('persona'))} puntos**, "
               "es lo que inflaba la validación original: tomas casi idénticas de la misma persona a ambos lados "
               "de la partición."]

    personas = sorted({p for k in ("publicado", "persona") if k in esq for p in esq[k]["por_persona"]})
    if personas:
        md += ["", "## 3. Por persona: modelo original contra persona no vista", "",
               "| Persona | Modelo original (%) | Persona no vista (%) |", "|---|---|---|"]
        for p in personas:
            o = esq.get("publicado", {}).get("por_persona", {}).get(p)
            n = esq.get("persona", {}).get("por_persona", {}).get(p)
            md.append(f"| {p} | {pct(o['accuracy']) if o else '—'} | {pct(n['accuracy']) if n else '—'} |")

    if "publicado" in esq and "persona" in esq:
        fo = esq["publicado"]["global"]["f1_por_clase"]
        fn = esq["persona"]["global"]["f1_por_clase"]
        md += ["", "## 4. F1 por seña", "", "| Seña | Modelo original | Persona no vista |", "|---|---|---|"]
        for c in res["clases"]:
            if c in fo or c in fn:
                md.append(f"| {c} | {pct(fo[c]) if c in fo else '—'} | {pct(fn[c]) if c in fn else '—'} |")

    captura = os.path.join(RAIZ, "resultados_captura_E3", "resumen_captura.md")
    if os.path.exists(captura):
        with open(captura, encoding="utf-8") as f:
            cuerpo = f.read().split("\n", 4)[-1]  # sin el título ni el aviso de generación
        md += ["", "## 5. Sesión de captura por dispositivo", "", cuerpo.strip()]

    # Las corridas más recientes del federado y de la calibración, si existen.
    for titulo, patron in (("6. Aprendizaje federado con personas reales",
                            os.path.join(RAIZ, "federated", "resultados_personas_*")),
                           ("7. Calibración y umbral de la app",
                            os.path.join(RAIZ, "resultados_calibracion", "20*")),
                           ("8. Robustez por condición (sesión 2)",
                            os.path.join(RAIZ, "resultados_robustez", "20*"))):
        corridas = sorted(glob.glob(patron))
        resumen = os.path.join(corridas[-1], "resumen.md") if corridas else None
        if resumen and os.path.exists(resumen):
            with open(resumen, encoding="utf-8") as f:
                cuerpo = f.read().split("\n", 1)[-1]  # sin el título propio
            md += ["", f"## {titulo}", "", f"Fuente: `{os.path.relpath(corridas[-1], RAIZ)}`.", "", cuerpo.strip()]

    md += ["", "## Tabla de escenarios en LaTeX (booktabs)", "", "```latex", *tex, "```", ""]
    os.makedirs(SALIDA, exist_ok=True)
    with open(os.path.join(SALIDA, "resultados.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))
    with open(os.path.join(SALIDA, "escenarios.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(md[:40]))
    print(f"\n-> {SALIDA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
