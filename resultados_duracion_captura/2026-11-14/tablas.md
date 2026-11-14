# Duración de la captura por toma

Generado por `scripts/duracion_captura.py`. Mediana (mín-máx; p10-p90).

## Teléfono (marcas de tiempo reales de la app)

`duracion_ms`: desde que empieza la captura hasta el último frame. «1.º→último»: entre el primer y el último frame guardado.

| Dispositivo | Persona | Tomas | Frames (tomas) | duracion_ms | 1.º→último ms | FPS |
|---|---|---|---|---|---|---|
| LENOVO TB321FU | P1 | 154 | {'15': 154} | 876 (782-1331; p10-p90 835-1133) | 819 (731-1250; p10-p90 779-1057) | 17,1 (11,3-19,2; p10-p90 13,2-18,0) |
| OnePlus CPH2663 | P2 | 153 | {'10': 104, '15': 49} | 1402 (1075-2285; p10-p90 1243-1920) | 1264 (988-2138; p10-p90 1122-1795) | 7,6 (5,6-14,0; p10-p90 6,8-8,7) |
| samsung SM-A556E | P3 | 142 | {'10': 131, '15': 11} | 1618 (825-2882; p10-p90 1404-2029) | 1452 (751-2697; p10-p90 1278-1825) | 6,3 (4,8-12,1; p10-p90 5,3-7,2) |

### Por bloque (A = señas, N = «nada», B = robustez, sesión 2)

| Dispositivo | Persona | Bloque | Tomas | Frames (tomas) | duracion_ms | FPS |
|---|---|---|---|---|---|---|
| LENOVO TB321FU | P1 | A | 96 | {'15': 96} | 874 (834-1331; p10-p90 846-1094) | 17,2 (11,3-18,0; p10-p90 13,7-17,7) |
| LENOVO TB321FU | P1 | B | 48 | {'15': 48} | 900 (782-1322; p10-p90 810-1086) | 16,7 (11,3-19,2; p10-p90 13,8-18,5) |
| LENOVO TB321FU | P1 | N | 10 | {'15': 10} | 876 (847-1320; p10-p90 848-1254) | 17,1 (11,4-17,7; p10-p90 12,0-17,7) |
| OnePlus CPH2663 | P2 | A | 96 | {'10': 52, '15': 44} | 1455 (1075-2285; p10-p90 1231-1945) | 7,8 (6,1-14,0; p10-p90 7,0-8,9) |
| OnePlus CPH2663 | P2 | B | 47 | {'10': 42, '15': 5} | 1390 (1183-2143; p10-p90 1286-1840) | 7,2 (5,6-8,5; p10-p90 6,7-7,9) |
| OnePlus CPH2663 | P2 | N | 10 | {'10': 10} | 1399 (1279-1630; p10-p90 1309-1545) | 7,1 (6,1-7,8; p10-p90 6,5-7,6) |
| samsung SM-A556E | P3 | A | 96 | {'10': 87, '15': 9} | 1608 (825-2882; p10-p90 1373-1966) | 6,3 (4,8-12,1; p10-p90 5,4-7,4) |
| samsung SM-A556E | P3 | B | 36 | {'10': 35, '15': 1} | 1620 (1395-2527; p10-p90 1496-1905) | 6,2 (4,8-7,2; p10-p90 5,4-6,7) |
| samsung SM-A556E | P3 | N | 10 | {'10': 9, '15': 1} | 1679 (1078-2482; p10-p90 1469-2118) | 6,0 (4,8-9,3; p10-p90 4,9-6,9) |

### Por cantidad de frames de la toma

| Dispositivo | Frames | Tomas | duracion_ms | FPS |
|---|---|---|---|---|
| LENOVO TB321FU | {'15': 154} | 154 | 876 (782-1331; p10-p90 835-1133) | 17,1 (11,3-19,2; p10-p90 13,2-18,0) |
| OnePlus CPH2663 | {'10': 104} | 104 | 1356 (1107-1796; p10-p90 1217-1535) | 7,4 (5,6-9,0; p10-p90 6,5-8,2) |
| OnePlus CPH2663 | {'15': 49} | 49 | 1795 (1075-2285; p10-p90 1651-2105) | 8,4 (6,6-14,0; p10-p90 7,1-9,1) |
| samsung SM-A556E | {'10': 131} | 131 | 1591 (825-2096; p10-p90 1395-1879) | 6,3 (4,8-12,1; p10-p90 5,3-7,2) |
| samsung SM-A556E | {'15': 11} | 11 | 2413 (1466-2882; p10-p90 2239-2579) | 6,2 (5,2-10,2; p10-p90 5,8-6,7) |

## Webcam

- webcam PC: 1200 tomas, frames por toma {'13': 3, '14': 6, '15': 1191}; `duracion_ms` en el archivo: 500 (433-500; p10-p90 500-500) y `fps` 30,0 (30,0-30,0; p10-p90 30,0-30,0), **valores supuestos por migrar_v1.py (30 FPS), no medidos**. Tomas con algún frame sin mano: 919.

Cota superior medida (segundos entre el guardado de una toma y el de la anterior de la misma clase, intervalos ≤ 60 s), por año de grabación:

| Año | Intervalos | Mediana s | Mín s | p10 s | p90 s |
|---|---|---|---|---|---|
| 2024 | 746 | 3,42 | 1,75 | 2,41 | 24,41 |
| 2025 | 139 | 13,17 | 2,42 | 2,81 | 32,05 |
| 2026 | 218 | 3,86 | 2,15 | 2,58 | 34,48 |
