"""
tracker_engine.py  v6
=====================
- v6: clasificador de equipos robusto a la luz (histogramas de color con
      correccion local de iluminacion), clusters asignados a cada equipo con
      los colores de camiseta de partido.toml, y votacion por ventana.
- v5: filtro de campo (descarta grada, fotografos y publico), tipo de plano,
      ids internos estables (uid) aunque el reconocedor cambie track_id por
      el dorsal, y cajas x/y/w/h reales para el reconocedor de dorsales.
- El punto sobre cada jugador se pinta con el COLOR REAL de su camiseta
- ClasificadorEquipo expone centroid_bgr(equipo) para que main.py lo use
- K-means en espacio Lab, funciona con cualquier color de playera
- Velocidad filtrada con mediana + suavizado exponencial
"""

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from filtro_campo import (FiltroCampo, PLANO_GENERAL, PLANO_MEDIO,
                          PLANO_PRIMER, PLANO_FUERA)

# ============================================================
# CONFIG
# ============================================================

COCO_PERSON = 0
COCO_BALL   = 32
MODELO_YOLO = "yolov8n.pt"   # "yolov8s.pt" detecta mejor si la PC lo aguanta
CONF_MIN    = 0.30
IMGSZ       = 640            # 960 ayuda con jugadores lejanos (mas lento)
FILTRAR_FUERA_DE_CAMPO = True

CANCHA_LARGO_M = 60.0
CANCHA_ANCHO_M = 40.0
HIST_LEN       = 15
VEL_MAX_REAL   = 36.0   # km/h maximo razonable en futbol
VEL_ALPHA      = 0.25   # suavizado exponencial

FACTOR_POSESION     = 1.8
BALON_MEMORIA_FRAMES = 12
VENTANA_POSESION    = 180
HISTERESIS_POSESION = 3
WARMUP_FRAMES       = 30
VOTOS_VENTANA       = 25     # votos de color recientes por jugador
VOTOS_MIN           = 3      # votos para asignar equipo por primera vez
VOTOS_CAMBIO        = 0.65   # fraccion de la ventana para cambiar de equipo

EQUIPO_A  = 1
EQUIPO_B  = 2
EQUIPO_NA = 0

# Color gris para jugadores sin clasificar
COLOR_NA_BGR = (160, 160, 160)


# ============================================================
# CLASIFICADOR DE COLOR  (v6: histogramas + correccion de luz)
# ============================================================
#
# Problemas de la version anterior (color Lab PROMEDIO del torso + k-means):
#   - Camisetas a rayas (Argentina): el promedio blanco/celeste se parece
#     poco a cualquier pixel real y queda cerca de otros colores claros.
#   - Sombras / exposicion: L (brillo) pesaba igual que el color, asi que un
#     jugador en sombra "cambiaba de equipo".
#   - Calibracion de solo 30 frames y luego congelada: si la luz cambiaba, o
#     en ese tramo casi solo se veia un equipo, el segundo cluster se lo
#     quedaba el portero, el arbitro o un recorte con dos jugadores.
#   - El orden de los clusters era aleatorio: EQUIPO_A podia ser Argentina
#     aunque en partido.toml sea el local (Espana).
#   - Votos acumulados para siempre: un error de ByteTrack (cambio de id
#     entre dos jugadores) quedaba pegado.
#
# Ahora:
#   1. Correccion LOCAL de iluminacion: se compara el brillo del cesped junto
#      a los pies del jugador con el brillo de referencia del cesped
#      (aprendido en la calibracion) y se escala el recorte de la camiseta.
#   2. Descriptor = histograma 2D suavizado de cromaticidad (a, b de Lab) +
#      histograma corto de brillo L con poco peso. Se usa la raiz del
#      histograma (distancia de Hellinger), asi una camiseta a rayas es
#      simplemente un histograma con dos picos.
#   3. Muestras: se ignoran jugadores superpuestos (el recorte mezclaria dos
#      camisetas) y se guarda 1 de cada 3 frames para no repetir lo mismo.
#   4. Con colores de camiseta en partido.toml: k-means con 2-5 clusters y
#      cada equipo se queda con los clusters parecidos a su color (varios
#      "prototipos": rayas blancas, rayas celestes, en sombra...); los demas
#      (porteros, arbitro, recortes malos) se ignoran. Si un equipo aun no
#      aparece en el video, se sigue clasificando solo el otro.
#      Sin colores en el toml: k = 2, emparejado con los prototipos previos.
#   5. Recalibracion CONTINUA: cada REFIT_CADA frames se rehace el k-means
#      con las ultimas MAX_MUESTRAS muestras, asi los prototipos siguen los
#      cambios de luz (sol / sombra, otra camara) y ganan variedad.
#   6. Se recorta el 10% de muestras mas lejanas de cada cluster.
#   7. Clasificacion con margen: si un jugador queda a medio camino entre los
#      dos equipos (o muy lejos de ambos) NO vota.
#   8. Votacion por ventana deslizante con histeresis (ver Track).

# Colores por nombre en BGR, para "camiseta = [...]" en partido.toml
COLORES_NOMBRE = {
    "rojo": (40, 30, 200), "granate": (40, 20, 120), "blanco": (240, 240, 240),
    "negro": (25, 25, 25), "gris": (128, 128, 128), "azul": (160, 60, 20),
    "azul marino": (90, 30, 20), "celeste": (219, 170, 117), "verde": (40, 140, 30),
    "amarillo": (30, 220, 240), "naranja": (20, 130, 240), "morado": (130, 40, 110),
    "rosa": (180, 120, 240),
}


def color_a_bgr(c):
    """'#RRGGBB', nombre ('rojo', 'celeste'...) o [b, g, r] -> tupla BGR."""
    if isinstance(c, (list, tuple)) and len(c) == 3:
        return tuple(int(v) for v in c)
    if not isinstance(c, str):
        return None
    c = c.strip().lower()
    if c.startswith("#") and len(c) == 7:
        try:
            r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
            return (b, g, r)
        except ValueError:
            return None
    return COLORES_NOMBRE.get(c)


def _bgr_a_lab(bgrs):
    px = np.array(bgrs, dtype=np.uint8).reshape(1, -1, 3)
    return cv2.cvtColor(px, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)


def cajas_solapadas(cajas, frac=0.15):
    """[bool] por caja: True si otra caja tapa mas de 'frac' de su area."""
    res = []
    for i, (ax1, ay1, ax2, ay2) in enumerate(cajas):
        area = max((ax2 - ax1) * (ay2 - ay1), 1)
        tapado = False
        for j, (bx1, by1, bx2, by2) in enumerate(cajas):
            if i == j:
                continue
            iw = min(ax2, bx2) - max(ax1, bx1)
            ih = min(ay2, by2) - max(ay1, by1)
            if iw > 0 and ih > 0 and iw * ih > frac * area:
                tapado = True
                break
        res.append(tapado)
    return res


class ClasificadorEquipo:
    """
    Aprende (y re-aprende) los colores de camiseta. Uso por frame:
        clf.nuevo_frame(frame, mask_cesped, calibrar)
        eq = clf.observar(frame, x1, y1, x2, y2, solapado)  # por jugador
    observar() -> EQUIPO_A / EQUIPO_B / EQUIPO_NA (si duda)
    colores_ref: {EQUIPO_A: [color, ...], EQUIPO_B: [color, ...]} (opcional)
    """

    BINS_AB = 8
    RANGO_AB = (80.0, 176.0)   # a, b de OpenCV (128 = neutro)
    BINS_L = 4
    PESO_L = 0.5               # el brillo pesa menos que el color
    MARGEN_MIN = 0.08          # (d_lejos - d_cerca) / (d_lejos + d_cerca)
    MIN_MUESTRAS = 40
    MAX_MUESTRAS = 800
    SUBMUESTREO = 3            # se guardan muestras 1 de cada 3 frames
    MIN_CLUSTER = 6            # muestras minimas para aceptar el cluster de un equipo
    DIST_REF_MAX = 55.0        # distancia Lab maxima cluster <-> color del toml
    REFIT_CADA = 60            # frames entre recalibraciones

    def __init__(self, colores_ref=None):
        self.colores_ref = {}
        for eq, lista in (colores_ref or {}).items():
            if isinstance(lista, str):
                lista = [lista]
            bgrs = [b for b in (color_a_bgr(c) for c in lista) if b is not None]
            if bgrs:
                self.colores_ref[eq] = _bgr_a_lab(bgrs).mean(axis=0)
        self.mapeo_por_color = EQUIPO_A in self.colores_ref and EQUIPO_B in self.colores_ref
        # Prototipos (centros de cluster) de cada equipo
        self.protos = np.zeros((0, self.BINS_AB ** 2 + self.BINS_L), np.float32)
        self.protos_eq = np.zeros(0, np.int32)        # EQUIPO_A / EQUIPO_B
        self.protos_umbral = np.zeros(0, np.float32)  # distancia maxima aceptada
        self.protos_lab = np.zeros((0, 3), np.float32)
        self.protos_n = np.zeros(0, np.float32)       # muestras de cada prototipo
        self.muestras = deque(maxlen=self.MAX_MUESTRAS)
        self.medias = deque(maxlen=self.MAX_MUESTRAS)
        self._frames = 0
        self._ultimo_fit = -10 ** 9
        self._guardar = False
        self.v_cesped_ref = None               # brillo de referencia del cesped
        self._mask_cesped = None

    def activo(self, equipo):
        return bool(np.any(self.protos_eq == equipo))

    def listo(self):
        """Al menos un equipo calibrado: ya se puede clasificar."""
        return len(self.protos_eq) > 0

    def completo(self):
        return self.activo(EQUIPO_A) and self.activo(EQUIPO_B)

    def centroid_bgr(self, equipo):
        sel = self.protos_eq == equipo
        if not np.any(sel):
            return COLOR_NA_BGR
        lab = np.average(self.protos_lab[sel], axis=0, weights=self.protos_n[sel])
        lab = np.clip(lab, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        b, g, r = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR).reshape(3)
        return (int(b), int(g), int(r))

    def nuevo_frame(self, frame_bgr, mask_cesped, calibrar):
        """
        Llamar una vez por frame ANTES de observar() a los jugadores.
        calibrar = False en planos donde no se ve juego (no se guardan muestras).
        """
        self._mask_cesped = mask_cesped
        self._guardar = False
        if not calibrar:
            return
        self._frames += 1
        self._guardar = self._frames % self.SUBMUESTREO == 0
        if not self.listo() and mask_cesped is not None:
            v = self._v_cesped(frame_bgr, mask_cesped)
            if v is not None:
                self.v_cesped_ref = v if self.v_cesped_ref is None else \
                    0.9 * self.v_cesped_ref + 0.1 * v
        espera = self.REFIT_CADA if self.completo() else self.REFIT_CADA // 2
        if (self._frames >= WARMUP_FRAMES and len(self.muestras) >= self.MIN_MUESTRAS
                and self._frames - self._ultimo_fit >= espera):
            self._ultimo_fit = self._frames
            self._fit()

    def observar(self, frame_bgr, x1, y1, x2, y2, solapado=False):
        """Guarda la muestra (si toca) y devuelve EQUIPO_A / EQUIPO_B / EQUIPO_NA."""
        if solapado:
            return EQUIPO_NA
        s = self._sample(frame_bgr, x1, y1, x2, y2)
        if s is None:
            return EQUIPO_NA
        if self._guardar:
            self.muestras.append(s[0])
            self.medias.append(s[1])
        if not self.listo():
            return EQUIPO_NA
        d = np.linalg.norm(self.protos - s[0], axis=1)
        i = int(np.argmin(d))                  # prototipo mas cercano
        eq = int(self.protos_eq[i])
        cerca, umbral = d[i], self.protos_umbral[i]
        otros = d[self.protos_eq != eq]
        if len(otros):
            lejos = float(otros.min())         # prototipo mas cercano del OTRO equipo
            margen = (lejos - cerca) / (lejos + cerca + 1e-6)
            if margen < self.MARGEN_MIN or cerca > umbral:
                return EQUIPO_NA
        elif cerca > 0.8 * umbral:             # solo un equipo calibrado: mas estricto
            return EQUIPO_NA
        return eq

    # ---- internos ----

    def _v_cesped(self, frame, mask, zona=None):
        if zona is not None:
            x1, y1, x2, y2 = zona
            frame = frame[y1:y2, x1:x2]
            mask = mask[y1:y2, x1:x2]
        if mask.size == 0:
            return None
        px = frame[mask > 0]
        if len(px) < 30:
            return None
        if len(px) > 4000:
            px = px[::len(px) // 4000]
        return float(np.median(px.max(axis=1)))   # V de HSV = max(B, G, R)

    def _ganancia_local(self, frame, x1, y1, x2, y2):
        """Brillo del cesped de referencia / brillo del cesped junto al jugador."""
        if self.v_cesped_ref is None or self._mask_cesped is None:
            return 1.0
        H, W = frame.shape[:2]
        w, h = x2 - x1, y2 - y1
        zona = (max(0, int(x1 - 0.5 * w)), max(0, int(y2 - 0.3 * h)),
                min(W, int(x2 + 0.5 * w)), min(H, int(y2 + 0.15 * h)))
        v = self._v_cesped(frame, self._mask_cesped, zona)
        if v is None or v < 5:
            return 1.0
        return float(np.clip(self.v_cesped_ref / v, 0.6, 1.8))

    def _sample(self, frame, x1, y1, x2, y2):
        """(descriptor, color Lab medio) de la camiseta, o None."""
        h, w = y2 - y1, x2 - x1
        ys, ye = int(y1 + 0.15 * h), int(y1 + 0.50 * h)
        xs, xe = int(x1 + 0.20 * w), int(x2 - 0.20 * w)
        roi = frame[max(ys, 0):max(ye, 1), max(xs, 0):max(xe, 1)]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_verde = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([90, 255, 255]))
        validos = (mask_verde == 0) & (hsv[..., 2] > 25)   # sin cesped ni sombra negra
        pixels = roi[validos]
        if len(pixels) < 20:
            return None
        g = self._ganancia_local(frame, x1, y1, x2, y2)
        if g != 1.0:
            pixels = np.clip(pixels.astype(np.float32) * g, 0, 255)
        lab = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8),
                           cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
        hab, _, _ = np.histogram2d(lab[:, 1], lab[:, 2], bins=self.BINS_AB,
                                   range=[self.RANGO_AB, self.RANGO_AB])
        # Suavizado: con pocos pixeles el histograma sale "ruidoso"
        hab = cv2.GaussianBlur(hab.astype(np.float32), (3, 3), 0.8)
        hl, _ = np.histogram(lab[:, 0], bins=self.BINS_L, range=(0, 256))
        f = np.concatenate([np.sqrt(hab.ravel() / max(float(hab.sum()), 1e-6)),
                            self.PESO_L * np.sqrt(hl / float(len(lab)))]).astype(np.float32)
        return f, lab.mean(axis=0)

    def _dist_ref(self, lab, equipo):
        pesos = np.array([self.PESO_L, 1.0, 1.0], np.float32)
        return float(np.linalg.norm((lab - self.colores_ref[equipo]) * pesos))

    def _clusters(self, k):
        """k-means sobre las muestras -> [(n, centro, lab, umbral), ...]"""
        data = np.array(self.muestras, dtype=np.float32)
        medias = np.array(self.medias, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
        _, etiquetas, centers = cv2.kmeans(data, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        etiquetas = etiquetas.ravel()
        res = []
        for j in range(k):
            sel = etiquetas == j
            n = int(np.count_nonzero(sel))
            if n < 3:
                continue
            d = np.linalg.norm(data[sel] - centers[j], axis=1)
            buenos = d <= np.percentile(d, 90)          # recorta atipicos
            centro = data[sel][buenos].mean(axis=0)
            d_b = np.linalg.norm(data[sel][buenos] - centro, axis=1)
            res.append((n, centro, medias[sel][buenos].mean(axis=0),
                        float(np.percentile(d_b, 95)) * 1.6 + 1e-3))
        return res

    def _reemplazar(self, equipo, clusters):
        """Sustituye los prototipos de un equipo por los clusters nuevos."""
        quedan = self.protos_eq != equipo
        self.protos = np.vstack([self.protos[quedan]] + [c[1][None] for c in clusters])
        self.protos_eq = np.concatenate(
            [self.protos_eq[quedan], np.full(len(clusters), equipo, np.int32)])
        self.protos_umbral = np.concatenate(
            [self.protos_umbral[quedan], np.array([c[3] for c in clusters], np.float32)])
        self.protos_lab = np.vstack([self.protos_lab[quedan]] + [c[2][None] for c in clusters])
        self.protos_n = np.concatenate(
            [self.protos_n[quedan], np.array([c[0] for c in clusters], np.float32)])

    def _fit(self):
        if not self.mapeo_por_color:
            clusters = self._clusters(2)
            if len(clusters) != 2:
                return
            # Mantener la identidad A/B entre recalibraciones
            if self.completo():
                pa = self.protos[self.protos_eq == EQUIPO_A].mean(axis=0)
                pb = self.protos[self.protos_eq == EQUIPO_B].mean(axis=0)
                c0, c1 = clusters[0][1], clusters[1][1]
                if (np.linalg.norm(c0 - pb) + np.linalg.norm(c1 - pa) <
                        np.linalg.norm(c0 - pa) + np.linalg.norm(c1 - pb)):
                    clusters = clusters[::-1]
            self._reemplazar(EQUIPO_A, [clusters[0]])
            self._reemplazar(EQUIPO_B, [clusters[1]])
            return

        k = int(np.clip(len(self.muestras) // 20, 2, 5))
        nuevos = {EQUIPO_A: [], EQUIPO_B: []}
        for c in self._clusters(k):
            n, _, lab, _ = c
            if n < self.MIN_CLUSTER:
                continue
            da, db = self._dist_ref(lab, EQUIPO_A), self._dist_ref(lab, EQUIPO_B)
            eq, d_eq, d_otro = (EQUIPO_A, da, db) if da <= db else (EQUIPO_B, db, da)
            # Debe parecerse al color del equipo, y claramente MAS que al del otro
            if d_eq < self.DIST_REF_MAX and d_eq < 0.7 * d_otro:
                nuevos[eq].append(c)
        # Si un equipo no salio en esta ventana, conserva sus prototipos previos
        for eq, lista in nuevos.items():
            if lista:
                self._reemplazar(eq, lista)


# ============================================================
# TRACK
# ============================================================

@dataclass
class Track:
    track_id: int                 # etiqueta visible (el reconocedor la cambia por el dorsal)
    uid: int = -1                 # id interno de ByteTrack, NUNCA cambia
    dorsal: object = None
    equipo: int = EQUIPO_NA
    cx: float = 0.0
    cy: float = 0.0
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    conf: float = 0.0
    hist: deque = field(default_factory=lambda: deque(maxlen=HIST_LEN))
    votos: deque = field(default_factory=lambda: deque(maxlen=VOTOS_VENTANA))
    dist_total_m: float = 0.0
    vel_actual_kmh: float = 0.0

    # Caja en formato x, y, w, h (la usa jersey_recognizer.py)
    @property
    def x(self): return self.x1
    @property
    def y(self): return self.y1
    @property
    def w(self): return self.x2 - self.x1
    @property
    def h(self): return self.y2 - self.y1

    def votar(self, eq):
        """Voto por ventana deslizante; solo cambia de equipo con mayoria clara."""
        if eq not in (EQUIPO_A, EQUIPO_B):
            return
        self.votos.append(eq)
        na = sum(1 for v in self.votos if v == EQUIPO_A)
        nb = len(self.votos) - na
        if self.equipo == EQUIPO_NA:
            if len(self.votos) >= VOTOS_MIN:
                self.equipo = EQUIPO_A if na >= nb else EQUIPO_B
        elif self.equipo == EQUIPO_A and nb > VOTOS_CAMBIO * len(self.votos):
            self.equipo = EQUIPO_B
        elif self.equipo == EQUIPO_B and na > VOTOS_CAMBIO * len(self.votos):
            self.equipo = EQUIPO_A


# ============================================================
# ESTADISTICAS POR EQUIPO
# ============================================================

@dataclass
class StatsEquipo:
    pases: int = 0
    pases_ok: int = 0
    perdidas: int = 0
    dist_m: float = 0.0
    vel_max: float = 0.0
    vel_prom: float = 0.0
    frames_pos: int = 0

    def precision(self):
        return round(100.0 * self.pases_ok / self.pases, 1) if self.pases else 0.0


# ============================================================
# MOTOR
# ============================================================

class MotorTracking:

    def __init__(self, frame_w, frame_h, modelo=MODELO_YOLO, colores_equipo=None):
        from ultralytics import YOLO
        self.model   = YOLO(modelo)
        self.frame_w = frame_w
        self.frame_h = frame_h

        self.m_por_px_x = CANCHA_LARGO_M / frame_w
        self.m_por_px_y = CANCHA_ANCHO_M / frame_h

        self.clf = ClasificadorEquipo(colores_equipo)
        self.campo = FiltroCampo()
        self.plano = PLANO_GENERAL
        self.descartados = 0
        self._etiqueta_posesion = None

        self.tracks: dict[int, Track] = {}
        self.balon         = None
        self.balon_vis     = None
        self.balon_perdido = 0

        self.pos_ventana      = deque(maxlen=VENTANA_POSESION)
        self.id_posesion      = None
        self.id_posesion_cand = None
        self.cand_frames      = 0
        self.eq_posesion      = EQUIPO_NA

        self.sa = StatsEquipo()
        self.sb = StatsEquipo()

        self.pases_a = 0
        self.pases_b = 0
        self.cambios_posesion = 0
        self.toques = 0

    # --------------------------------------------------------
    def procesar_frame(self, frame_bgr):
        t_now = time.time()

        self.campo.procesar(frame_bgr)

        res = self.model.track(
            frame_bgr,
            persist=True,
            classes=[COCO_PERSON, COCO_BALL],
            conf=CONF_MIN,
            imgsz=IMGSZ,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        r = res[0]
        ids_vistos = set()
        self.balon = None
        dets_pers  = []
        tracks_frame = []
        self.descartados = 0

        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                if cls == COCO_BALL:
                    self.balon = (cx, cy)
                    continue
                if box.id is None:
                    continue

                # Personas fuera del terreno (grada, fotografos, publico)
                if FILTRAR_FUERA_DE_CAMPO and not self.campo.persona_en_campo(x1, y1, x2, y2):
                    self.descartados += 1
                    continue

                tid = int(box.id[0])
                ids_vistos.add(tid)
                dets_pers.append((x1, y1, x2, y2))

                tr = self.tracks.get(tid)
                if tr is None:
                    tr = Track(track_id=tid, uid=tid)
                    self.tracks[tid] = tr

                if tr.hist:
                    _, px, py = tr.hist[-1]
                    dx = (cx - px) * self.m_por_px_x
                    dy = (cy - py) * self.m_por_px_y
                    tr.dist_total_m += (dx**2 + dy**2) ** 0.5

                tr.x1, tr.y1, tr.x2, tr.y2 = x1, y1, x2, y2
                tr.cx, tr.cy = cx, cy
                tr.conf = conf
                tr.hist.append((t_now, cx, cy))
                tracks_frame.append(tr)

        self.plano = self.campo.clasificar_plano(
            [y2 - y1 for (_, y1, _, y2) in dets_pers], self.frame_h)

        # Equipo por color. Solo se calibra en planos donde se ve juego, y los
        # jugadores superpuestos no votan (su recorte mezcla dos camisetas).
        self.clf.nuevo_frame(frame_bgr, self.campo.mask_cesped,
                             calibrar=self.plano in (PLANO_GENERAL, PLANO_MEDIO))
        for tr, (x1, y1, x2, y2), tapado in zip(tracks_frame, dets_pers,
                                                cajas_solapadas(dets_pers)):
            tr.votar(self.clf.observar(frame_bgr, x1, y1, x2, y2, tapado))

        for tid in list(self.tracks.keys()):
            if tid not in ids_vistos:
                self._volcar_dist(self.tracks.pop(tid))

        if self.balon is not None:
            self.balon_vis    = self.balon
            self.balon_perdido = 0
        else:
            self.balon_perdido += 1
            if self.balon_perdido > BALON_MEMORIA_FRAMES:
                self.balon_vis = None

        self._calc_vel()
        # En primeros planos / grada no hay juego que medir: congelamos posesion
        if self.juego_activo():
            self._actualizar_posesion()
        self._actualizar_stats()

        return self._estado()

    def juego_activo(self):
        return self.plano in (PLANO_GENERAL, PLANO_MEDIO)

    # --------------------------------------------------------
    def _volcar_dist(self, tr):
        if tr.equipo == EQUIPO_A:   self.sa.dist_m += tr.dist_total_m
        elif tr.equipo == EQUIPO_B: self.sb.dist_m += tr.dist_total_m

    def _calc_vel(self):
        for tr in self.tracks.values():
            if len(tr.hist) < 4:
                tr.vel_actual_kmh = 0.0
                continue
            hist = list(tr.hist)
            vels = []
            for i in range(1, len(hist)):
                t0,x0,y0 = hist[i-1]; t1,x1,y1 = hist[i]
                dt = t1 - t0
                if dt <= 0: continue
                v = (((x1-x0)*self.m_por_px_x)**2 +
                     ((y1-y0)*self.m_por_px_y)**2)**0.5 / dt * 3.6
                vels.append(v)
            if not vels:
                tr.vel_actual_kmh = 0.0
                continue
            v_med = float(np.median(vels))
            if v_med > VEL_MAX_REAL:
                v_med = tr.vel_actual_kmh
            tr.vel_actual_kmh = VEL_ALPHA*v_med + (1-VEL_ALPHA)*tr.vel_actual_kmh

    def _altura_media(self):
        alt = [tr.y2-tr.y1 for tr in self.tracks.values()]
        return sum(alt)/len(alt) if alt else 0.0

    def _actualizar_posesion(self):
        if self.balon_vis is None or not self.tracks: return
        bx, by = self.balon_vis
        radio  = max(self._altura_media() * FACTOR_POSESION, 30.0)

        mejor_d, mejor_tr = 1e9, None
        for tr in self.tracks.values():
            d = ((tr.cx-bx)**2 + (tr.cy-by)**2)**0.5
            if d < mejor_d: mejor_d, mejor_tr = d, tr

        nuevo_id = (mejor_tr.uid
                    if mejor_tr and mejor_d <= radio else None)

        if nuevo_id == self.id_posesion:
            self.id_posesion_cand = None; self.cand_frames = 0
        else:
            if nuevo_id == self.id_posesion_cand: self.cand_frames += 1
            else: self.id_posesion_cand = nuevo_id; self.cand_frames = 1
            if self.cand_frames >= HISTERESIS_POSESION:
                self._registrar_cambio(nuevo_id)
                self.id_posesion = nuevo_id
                self.id_posesion_cand = None; self.cand_frames = 0

        if self.id_posesion in self.tracks:
            eq = self.tracks[self.id_posesion].equipo
            self.eq_posesion = eq
            if eq == EQUIPO_A: self.pos_ventana.append(eq); self.sa.frames_pos += 1
            elif eq == EQUIPO_B: self.pos_ventana.append(eq); self.sb.frames_pos += 1

    def _registrar_cambio(self, nuevo_id):
        prev_id = self.id_posesion
        if prev_id is None or nuevo_id is None:
            if nuevo_id is not None: self.toques += 1
            return
        if prev_id not in self.tracks or nuevo_id not in self.tracks:
            if nuevo_id is not None: self.toques += 1
            return
        eq_p = self.tracks[prev_id].equipo
        eq_n = self.tracks[nuevo_id].equipo
        self.toques += 1
        if eq_p == eq_n and eq_p in (EQUIPO_A, EQUIPO_B):
            if eq_n == EQUIPO_A:
                self.pases_a += 1; self.sa.pases += 1; self.sa.pases_ok += 1
            else:
                self.pases_b += 1; self.sb.pases += 1; self.sb.pases_ok += 1
        elif eq_p in (EQUIPO_A,EQUIPO_B) and eq_n in (EQUIPO_A,EQUIPO_B):
            self.cambios_posesion += 1
            if eq_p == EQUIPO_A:
                self.sa.perdidas += 1
                self.sa.pases_ok = max(0, self.sa.pases_ok-1)
            else:
                self.sb.perdidas += 1
                self.sb.pases_ok = max(0, self.sb.pases_ok-1)

    def _actualizar_stats(self):
        va, vb = [], []
        for tr in self.tracks.values():
            v = tr.vel_actual_kmh
            if tr.equipo == EQUIPO_A:
                va.append(v)
                if v > self.sa.vel_max: self.sa.vel_max = v
            elif tr.equipo == EQUIPO_B:
                vb.append(v)
                if v > self.sb.vel_max: self.sb.vel_max = v
        if va: self.sa.vel_prom = sum(va)/len(va)
        if vb: self.sb.vel_prom = sum(vb)/len(vb)

    def _zona_3x3(self):
        nombres = [
            ["Def. Izq.","Defensa","Def. Der."],
            ["Med. Izq.","Centro","Med. Der."],
            ["Ata. Izq.","Ataque","Ata. Der."],
        ]
        xs, ys = [], []
        for tr in self.tracks.values(): xs.append(tr.cx); ys.append(tr.cy)
        if self.balon_vis: xs.append(self.balon_vis[0]); ys.append(self.balon_vis[1])
        if not xs: return "SIN ACCION"
        col = min(int(sum(xs)/len(xs)*3/self.frame_w), 2)
        row = min(int(sum(ys)/len(ys)*3/self.frame_h), 2)
        return nombres[row][col]

    def _estado(self):
        ca = sum(1 for t in self.tracks.values() if t.equipo == EQUIPO_A)
        cb = sum(1 for t in self.tracks.values() if t.equipo == EQUIPO_B)
        na = sum(1 for e in self.pos_ventana if e == EQUIPO_A)
        nb = sum(1 for e in self.pos_ventana if e == EQUIPO_B)
        tot = na + nb
        pos_a = round(100*na/tot) if tot else 50
        pos_b = 100 - pos_a
        dist_a = self.sa.dist_m + sum(t.dist_total_m for t in self.tracks.values() if t.equipo==EQUIPO_A)
        dist_b = self.sb.dist_m + sum(t.dist_total_m for t in self.tracks.values() if t.equipo==EQUIPO_B)

        # Colores BGR reales de cada equipo (del K-means)
        color_a_bgr = self.clf.centroid_bgr(EQUIPO_A)
        color_b_bgr = self.clf.centroid_bgr(EQUIPO_B)

        tr_pos = self.tracks.get(self.id_posesion)
        if tr_pos is not None:
            self._etiqueta_posesion = tr_pos.track_id
        elif self.id_posesion is None:
            self._etiqueta_posesion = None
        return {
            "tracks":           list(self.tracks.values()),
            "balon":            self.balon_vis,
            "balon_detectado":  self.balon is not None,
            "plano":            self.plano,
            "juego_activo":     self.juego_activo(),
            "ratio_cesped":     self.campo.ratio_cesped,
            "ratio_cesped_inf": self.campo.ratio_cesped_inferior,
            "descartados":      self.descartados,
            "uid_posesion":     self.id_posesion,
            "count_a":          ca,
            "count_b":          cb,
            "pos_a":            pos_a,
            "pos_b":            pos_b,
            "zona":             self._zona_3x3(),
            # etiqueta visible del poseedor (dorsal si ya se reconocio)
            "id_posesion":      self._etiqueta_posesion,
            "eq_posesion":      self.eq_posesion,
            "pases_a":          self.pases_a,
            "pases_b":          self.pases_b,
            "cambios_posesion": self.cambios_posesion,
            "toques":           self.toques,
            "precision_a":      self.sa.precision(),
            "precision_b":      self.sb.precision(),
            "perdidas_a":       self.sa.perdidas,
            "perdidas_b":       self.sb.perdidas,
            "dist_a":           dist_a,
            "dist_b":           dist_b,
            "vel_max_a":        self.sa.vel_max,
            "vel_max_b":        self.sb.vel_max,
            "vel_prom_a":       self.sa.vel_prom,
            "vel_prom_b":       self.sb.vel_prom,
            "warmup":           not self.clf.listo(),
            # NUEVO: colores reales de camiseta para pintar los puntos
            "color_a_bgr":      color_a_bgr,
            "color_b_bgr":      color_b_bgr,
        }

import csv

def exportar_estadisticas_csv(motor, archivo="estadisticas_partido.csv"):
    with open(archivo,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["Jugador","Equipo","Distancia_m","VelMax_kmh","Sprints"])
        for tr in motor.tracks.values():
            w.writerow([
                tr.track_id,
                getattr(tr,"equipo",""),
                round(getattr(tr,"dist_total_m",0),2),
                round(getattr(tr,"vel_max_personal",0),2),
                getattr(tr,"sprints",0)
            ])
