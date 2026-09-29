"""
blob_detector.py
================
Two-pass Connected Components Labeling con Union-Find.
Deteccion de jugadores individuales por equipo.

Algoritmo identico al de la version C (avance3_video.c):
  Pasada 1 -> asignar etiquetas provisionales con vecinos arriba e izquierda
  Pasada 2 -> resolver equivalencias, acumular momentos, filtrar por area

NO usa scipy, skimage, ni ninguna libreria de vision artificial.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

# ======================== CONSTANTES ========================
MAX_LABELS       = 2048
MAX_JUGADORES    = 64
MIN_AREA_JUGADOR = 80
MAX_AREA_JUGADOR = 50000

# ======================== DETECCION DE COLOR ========================
# Rangos RGB identicos a los #define del codigo C

EQ_A_R = (170, 255)
EQ_A_G = (  0,  90)
EQ_A_B = ( 20,  90)

EQ_B_R = (  0,  90)
EQ_B_G = ( 20, 140)
EQ_B_B = (150, 255)

BALON_MIN = 180


def es_equipo_a(r: int, g: int, b: int) -> bool:
    return (EQ_A_R[0] <= r <= EQ_A_R[1] and
            EQ_A_G[0] <= g <= EQ_A_G[1] and
            EQ_A_B[0] <= b <= EQ_A_B[1])


def es_equipo_b(r: int, g: int, b: int) -> bool:
    return (EQ_B_R[0] <= r <= EQ_B_R[1] and
            EQ_B_G[0] <= g <= EQ_B_G[1] and
            EQ_B_B[0] <= b <= EQ_B_B[1])


def es_balon(r: int, g: int, b: int) -> bool:
    return r >= BALON_MIN and g >= BALON_MIN and b >= BALON_MIN


# ======================== ESTRUCTURA JUGADOR ========================
@dataclass
class Jugador:
    cx:    float = 0.0
    cy:    float = 0.0
    xmin:  int   = 0
    xmax:  int   = 0
    ymin:  int   = 0
    ymax:  int   = 0
    area:  int   = 0
    zona:  int   = 0     # 0-8 en el grid 3x3
    equipo: int  = 0     # 1 = A, 2 = B


# ======================== UNION-FIND ========================
class UnionFind:
    """
    Union-Find con compresion de camino.
    Equivalente directo de parent[], find() y unite() del codigo C.
    """
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        # Compresion de camino (path compression)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def unite(self, a: int, b: int):
        a = self.find(a)
        b = self.find(b)
        if a != b:
            self.parent[b] = a


# ======================== ALGORITMO PRINCIPAL ========================
def detectar_jugadores(
    pixels_rgb: bytearray,
    width: int,
    height: int,
    equipo_fn: Callable[[int, int, int], bool],
    equipo_id: int
) -> List[Jugador]:
    """
    Two-pass connected components para un equipo.

    Parametros:
        pixels_rgb  buffer RGB top-left (width*height*3)
        width, height  dimensiones de la imagen
        equipo_fn   funcion de deteccion de color (es_equipo_a o es_equipo_b)
        equipo_id   1 o 2

    Retorna lista de Jugador filtrados por area.
    """

    # ---- Paso 0: mapa de etiquetas, 0 = fondo ----
    labels = [0] * (width * height)
    uf     = UnionFind(MAX_LABELS)
    next_label = 1

    # ---- PASADA 1: etiquetas provisionales ----
    for y in range(height):
        for x in range(width):
            idx = (y * width + x) * 3
            r = pixels_rgb[idx]
            g = pixels_rgb[idx + 1]
            b = pixels_rgb[idx + 2]

            if not equipo_fn(r, g, b):
                continue  # pixel de fondo

            lup   = labels[(y - 1) * width + x] if y > 0 else 0
            lleft = labels[y * width + (x - 1)] if x > 0 else 0

            if lup == 0 and lleft == 0:
                if next_label < MAX_LABELS:
                    labels[y * width + x] = next_label
                    next_label += 1
            elif lup != 0 and lleft == 0:
                labels[y * width + x] = lup
            elif lup == 0 and lleft != 0:
                labels[y * width + x] = lleft
            else:
                menor = lup if lup < lleft else lleft
                labels[y * width + x] = menor
                uf.unite(lup, lleft)

    # ---- PASADA 2: resolver equivalencias ----
    remap   = [0] * MAX_LABELS
    n_blobs = 0

    for i in range(width * height):
        if labels[i] > 0:
            root = uf.find(labels[i])
            labels[i] = root
            if remap[root] == 0:
                n_blobs += 1
                remap[root] = n_blobs

    if n_blobs == 0:
        return []

    # ---- Acumular estadisticas por blob ----
    sum_x = [0] * (n_blobs + 1)
    sum_y = [0] * (n_blobs + 1)
    area  = [0] * (n_blobs + 1)
    xmin  = [width]  * (n_blobs + 1)
    xmax  = [0]      * (n_blobs + 1)
    ymin  = [height] * (n_blobs + 1)
    ymax  = [0]      * (n_blobs + 1)

    for y in range(height):
        for x in range(width):
            lbl = labels[y * width + x]
            if lbl == 0:
                continue
            bid = remap[lbl]
            sum_x[bid] += x
            sum_y[bid] += y
            area[bid]  += 1
            if x < xmin[bid]: xmin[bid] = x
            if x > xmax[bid]: xmax[bid] = x
            if y < ymin[bid]: ymin[bid] = y
            if y > ymax[bid]: ymax[bid] = y

    # ---- Construir lista de jugadores (filtrar por area) ----
    jugadores: List[Jugador] = []

    for i in range(1, n_blobs + 1):
        if len(jugadores) >= MAX_JUGADORES:
            break
        if area[i] < MIN_AREA_JUGADOR:
            continue
        if area[i] > MAX_AREA_JUGADOR:
            continue

        cx = sum_x[i] / area[i]
        cy = sum_y[i] / area[i]

        # Zona en el grid 3x3
        col  = min(int(cx * 3) // width,  2)
        fila = min(int(cy * 3) // height, 2)
        zona = fila * 3 + col

        jugadores.append(Jugador(
            cx=cx, cy=cy,
            xmin=xmin[i], xmax=xmax[i],
            ymin=ymin[i], ymax=ymax[i],
            area=area[i],
            zona=zona,
            equipo=equipo_id
        ))

    return jugadores


# ======================== ANALISIS GLOBAL DE FRAME ========================
def analizar_pixels(pixels_rgb: bytearray, width: int, height: int) -> dict:
    """
    Recorre todos los pixeles y calcula:
      - count y centroide global de cada equipo y del balon
      - zonas 3x3 (conteo de pixeles por zona)
      - lista de jugadores individuales (blobs) por equipo

    Retorna un dict con todos los resultados, equivalente
    al AppState de la version C tras la llamada a analizarPixels().
    """
    sum_x_a = sum_y_a = count_a = 0
    sum_x_b = sum_y_b = count_b = 0
    sum_x_bal = sum_y_bal = count_bal = 0
    zonas_a = [0] * 9
    zonas_b = [0] * 9

    for y in range(height):
        for x in range(width):
            idx = (y * width + x) * 3
            r = pixels_rgb[idx]
            g = pixels_rgb[idx + 1]
            b = pixels_rgb[idx + 2]

            col_z  = (x * 3) // width
            fil_z  = (y * 3) // height
            zona   = fil_z * 3 + col_z

            if es_equipo_a(r, g, b):
                sum_x_a += x; sum_y_a += y; count_a += 1
                zonas_a[zona] += 1
            elif es_equipo_b(r, g, b):
                sum_x_b += x; sum_y_b += y; count_b += 1
                zonas_b[zona] += 1
            elif es_balon(r, g, b):
                sum_x_bal += x; sum_y_bal += y; count_bal += 1

    cx_a   = sum_x_a  / count_a   if count_a   else 0.0
    cy_a   = sum_y_a  / count_a   if count_a   else 0.0
    cx_b   = sum_x_b  / count_b   if count_b   else 0.0
    cy_b   = sum_y_b  / count_b   if count_b   else 0.0
    cx_bal = sum_x_bal / count_bal if count_bal else 0.0
    cy_bal = sum_y_bal / count_bal if count_bal else 0.0

    jugadores_a = detectar_jugadores(pixels_rgb, width, height, es_equipo_a, 1)
    jugadores_b = detectar_jugadores(pixels_rgb, width, height, es_equipo_b, 2)

    return {
        "count_a": count_a,  "cx_a": cx_a,  "cy_a": cy_a,
        "count_b": count_b,  "cx_b": cx_b,  "cy_b": cy_b,
        "count_balon": count_bal, "cx_balon": cx_bal, "cy_balon": cy_bal,
        "zonas_a": zonas_a,
        "zonas_b": zonas_b,
        "jugadores_a": jugadores_a,
        "jugadores_b": jugadores_b,
    }
