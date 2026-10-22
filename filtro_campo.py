"""
filtro_campo.py
===============
Segmentacion del cesped y clasificacion del tipo de plano.

Sirve para:
  1. Descartar personas que NO estan sobre el terreno de juego
     (aficionados en la grada, fotografos, banca, publico en primer plano).
  2. Saber que tipo de plano esta mostrando la camara:
        GENERAL       -> se ve mucho campo, jugadores pequenos (juego normal)
        MEDIO         -> campo visible, jugadores medianos
        PRIMER_PLANO  -> 1-3 jugadores grandes (celebraciones, repeticiones)
        FUERA_CAMPO   -> casi no hay cesped (grada, banca, publico)

Idea principal
--------------
El color del cesped cambia entre estadios, iluminacion y sombras, asi que
no usamos un rango de verde fijo: se aprende el TONO (H de HSV) dominante
del cesped con un histograma de la mitad inferior del frame y se suaviza
con un promedio exponencial. Asi las vallas publicitarias verdes (otro tono
y mucha mas saturacion) no se confunden con el campo.
"""

import cv2
import numpy as np

PLANO_GENERAL = "GENERAL"
PLANO_MEDIO = "MEDIO"
PLANO_PRIMER = "PRIMER_PLANO"
PLANO_FUERA = "FUERA_CAMPO"

NOMBRES_PLANO = {
    PLANO_GENERAL: "Plano general",
    PLANO_MEDIO: "Plano medio",
    PLANO_PRIMER: "Primer plano",
    PLANO_FUERA: "Fuera del campo",
}


class FiltroCampo:

    # Rango amplio donde buscamos el pico de tono del cesped (OpenCV: H 0-179)
    H_BUSQUEDA = (25, 95)
    S_MIN = 40
    V_MIN = 30
    TOL_H_BAJO = 13    # tolerancia por debajo del pico
    TOL_H_ALTO = 13    # por arriba (las sombras tiran a tonos mas altos)
    ALPHA_H = 0.10     # suavizado del tono aprendido

    def __init__(self, escala_proc=0.5):
        # Se procesa a media resolucion: es suficiente y mucho mas rapido
        self.escala = escala_proc
        self.h_cesped = None
        self.mask_cesped = None   # cesped real (resolucion completa)
        self.mask_campo = None    # region del campo (cesped + huecos + jugadores)
        self.ratio_cesped = 0.0
        self.ratio_cesped_inferior = 0.0
        self.contorno_campo = None

    # ------------------------------------------------------------
    # 1) Aprender tono del cesped
    # ------------------------------------------------------------

    def _actualizar_tono(self, hsv):
        h, s, v = cv2.split(hsv)
        alto = hsv.shape[0]
        zona = slice(alto // 2, alto)   # mitad inferior: casi siempre es campo
        hh, ss, vv = h[zona], s[zona], v[zona]
        m = (ss > self.S_MIN) & (vv > self.V_MIN) & \
            (hh >= self.H_BUSQUEDA[0]) & (hh <= self.H_BUSQUEDA[1])
        if np.count_nonzero(m) < 0.08 * m.size:
            return  # casi no hay verde: no actualizamos (grada / primer plano)
        hist = np.bincount(hh[m].ravel(), minlength=180).astype(np.float32)
        hist = np.convolve(hist, np.ones(5, np.float32) / 5, mode="same")
        pico = float(np.argmax(hist))
        if self.h_cesped is None:
            self.h_cesped = pico
        else:
            self.h_cesped = (1 - self.ALPHA_H) * self.h_cesped + self.ALPHA_H * pico

    # ------------------------------------------------------------
    # 2) Procesar frame
    # ------------------------------------------------------------

    def procesar(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        peq = cv2.resize(frame_bgr, (max(1, int(W * self.escala)),
                                     max(1, int(H * self.escala))),
                         interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(peq, cv2.COLOR_BGR2HSV)
        self._actualizar_tono(hsv)

        if self.h_cesped is None:
            self.mask_cesped = np.zeros((H, W), np.uint8)
            self.mask_campo = np.zeros((H, W), np.uint8)
            self.ratio_cesped = 0.0
            self.ratio_cesped_inferior = 0.0
            self.contorno_campo = None
            return

        lo = int(max(0, self.h_cesped - self.TOL_H_BAJO))
        hi = int(min(179, self.h_cesped + self.TOL_H_ALTO))
        cesped = cv2.inRange(hsv, (lo, self.S_MIN, self.V_MIN), (hi, 255, 255))

        ph, pw = cesped.shape
        k_chico = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cesped = cv2.morphologyEx(cesped, cv2.MORPH_OPEN, k_chico)

        self.ratio_cesped = float(np.count_nonzero(cesped)) / cesped.size
        self.ratio_cesped_inferior = float(
            np.count_nonzero(cesped[ph * 2 // 3:])) / max(1, cesped[ph * 2 // 3:].size)

        # Region del campo: cerrar huecos (lineas, jugadores) y quedarse con
        # las componentes grandes; luego envolvente convexa.
        k = max(5, int(min(pw, ph) * 0.06)) | 1
        cerrado = cv2.morphologyEx(
            cesped, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(cerrado, 8)
        campo = np.zeros_like(cerrado)
        area_min = 0.03 * cerrado.size
        puntos = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= area_min:
                ys, xs = np.nonzero(etiquetas == i)
                puntos.append(np.column_stack([xs, ys]))
        self.contorno_campo = None
        if puntos:
            pts = np.vstack(puntos).astype(np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(campo, hull, 255)
            self.contorno_campo = (hull.astype(np.float32) / self.escala).astype(np.int32)

        self.mask_cesped = cv2.resize(cesped, (W, H), interpolation=cv2.INTER_NEAREST)
        self.mask_campo = cv2.resize(campo, (W, H), interpolation=cv2.INTER_NEAREST)

    # ------------------------------------------------------------
    # 3) Consultas
    # ------------------------------------------------------------

    def _ratio(self, mask, x1, y1, x2, y2):
        H, W = mask.shape
        x1, x2 = max(0, int(x1)), min(W, int(x2))
        y1, y2 = max(0, int(y1)), min(H, int(y2))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = mask[y1:y2, x1:x2]
        return float(np.count_nonzero(roi)) / roi.size

    def persona_en_campo(self, x1, y1, x2, y2):
        """
        True si la caja de una persona parece estar sobre el terreno de juego.
        Se evalua la zona de los pies, no el centro de la caja.
        """
        if self.mask_campo is None or self.h_cesped is None:
            return True  # sin informacion todavia: no filtramos
        H, W = self.mask_campo.shape
        w = x2 - x1
        h = y2 - y1
        if h < 0.035 * H:          # demasiado pequena (gente lejana / grada)
            return False

        toca_borde_inf = y2 >= H - max(4, 0.02 * H)
        if not toca_borde_inf:
            # Pies visibles: el punto de apoyo debe estar dentro del campo
            # y debe haber cesped alrededor de los pies.
            px = int((x1 + x2) / 2)
            py = int(min(H - 1, y2 - 0.02 * h))
            if self.mask_campo[py, px] == 0:
                return False
            franja = self._ratio(self.mask_cesped,
                                 x1 - 0.35 * w, y2 - 0.12 * h,
                                 x2 + 0.35 * w, y2 + 0.06 * h)
            return franja >= 0.12

        # Pies fuera de cuadro (primer plano): pedimos cesped a los lados de
        # la parte baja del cuerpo. El publico o los fotografos en primer
        # plano (manos, cabezas, camaras) no tienen cesped alrededor.
        lados_inf = max(
            self._ratio(self.mask_cesped, x1 - 0.5 * w, y1 + 0.75 * h, x1, y2),
            self._ratio(self.mask_cesped, x2, y1 + 0.75 * h, x2 + 0.5 * w, y2),
        )
        lados_med = max(
            self._ratio(self.mask_cesped, x1 - 0.5 * w, y1 + 0.5 * h, x1, y2),
            self._ratio(self.mask_cesped, x2, y1 + 0.5 * h, x2 + 0.5 * w, y2),
        )
        return lados_inf >= 0.25 or lados_med >= 0.45

    def clasificar_plano(self, alturas_jugadores, alto_frame):
        """Tipo de plano a partir del cesped visible y del tamano de los jugadores."""
        if self.ratio_cesped < 0.10:
            return PLANO_FUERA
        alt_med = float(np.median(alturas_jugadores)) / alto_frame if alturas_jugadores else 0.0
        alt_max = float(np.max(alturas_jugadores)) / alto_frame if alturas_jugadores else 0.0
        pocos = len(alturas_jugadores) <= 2
        if alt_max >= 0.36 or (pocos and alt_max >= 0.26) or \
                (self.ratio_cesped < 0.30 and alt_med >= 0.22):
            return PLANO_PRIMER
        if alt_med >= 0.17 or self.ratio_cesped < 0.40:
            return PLANO_MEDIO
        return PLANO_GENERAL

    def dibujar(self, frame, color=(0, 255, 255)):
        """Dibuja el contorno del campo detectado (para depurar)."""
        if self.contorno_campo is not None:
            cv2.polylines(frame, [self.contorno_campo], True, color, 1)
