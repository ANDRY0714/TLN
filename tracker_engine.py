"""
tracker_engine.py  v4
=====================
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

# ============================================================
# CONFIG
# ============================================================

COCO_PERSON = 0
COCO_BALL   = 32
MODELO_YOLO = "yolov8n.pt"
CONF_MIN    = 0.30

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

EQUIPO_A  = 1
EQUIPO_B  = 2
EQUIPO_NA = 0

# Color gris para jugadores sin clasificar
COLOR_NA_BGR = (160, 160, 160)


# ============================================================
# CLASIFICADOR DE COLOR  (K-means, cualquier color de camiseta)
# ============================================================

class ClasificadorEquipo:
    """
    Aprende los dos colores dominantes de camiseta en warm-up y:
      - clasificar()     -> EQUIPO_A / EQUIPO_B
      - centroid_bgr()   -> color BGR real del centroide (para pintar el punto)
    """

    def __init__(self):
        self.centroids_lab = None   # (2, 3) en Lab
        self.centroids_bgr = None   # (2, 3) en BGR uint8
        self.muestras      = []
        self._listo        = False
        self._frames       = 0

    def listo(self):
        return self._listo

    def centroid_bgr(self, equipo):
        """Devuelve el color BGR del centroide del equipo, o gris si no esta listo."""
        if not self._listo:
            return COLOR_NA_BGR
        idx = 0 if equipo == EQUIPO_A else 1
        b, g, r = self.centroids_bgr[idx]
        return (int(b), int(g), int(r))

    def warm_up(self, frame_bgr, detecciones):
        for x1, y1, x2, y2 in detecciones:
            c = self._sample(frame_bgr, x1, y1, x2, y2)
            if c is not None:
                self.muestras.append(c)
        self._frames += 1
        if self._frames >= WARMUP_FRAMES and len(self.muestras) >= 6:
            self._fit()

    def clasificar(self, frame_bgr, x1, y1, x2, y2):
        if not self._listo:
            return EQUIPO_NA
        c = self._sample(frame_bgr, x1, y1, x2, y2)
        if c is None:
            return EQUIPO_NA
        d0 = np.linalg.norm(c - self.centroids_lab[0])
        d1 = np.linalg.norm(c - self.centroids_lab[1])
        return EQUIPO_A if d0 <= d1 else EQUIPO_B

    # ---- internos ----

    def _sample(self, frame, x1, y1, x2, y2):
        """Color Lab promedio de la zona de camiseta, excluyendo verde del campo."""
        h = y2 - y1
        ys = int(y1 + 0.15 * h)
        ye = int(y1 + 0.50 * h)
        roi = frame[max(ys,0):max(ye,1), max(x1,0):max(x2,1)]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_verde = cv2.inRange(hsv, np.array([35,40,40]), np.array([90,255,255]))
        pixels = roi[cv2.bitwise_not(mask_verde) > 0]
        if len(pixels) < 20:
            return None
        lab = cv2.cvtColor(pixels.reshape(-1,1,3).astype(np.uint8),
                           cv2.COLOR_BGR2Lab).reshape(-1, 3)
        return lab.mean(axis=0)

    def _fit(self):
        data = np.array(self.muestras, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, _, centers = cv2.kmeans(data, 2, None, crit, 10,
                                   cv2.KMEANS_RANDOM_CENTERS)
        self.centroids_lab = centers  # (2, 3) Lab float32

        # Convertir centroides Lab -> BGR para usarlos directamente en cv2
        bgr = cv2.cvtColor(
            centers.reshape(1, 2, 3).astype(np.uint8),
            cv2.COLOR_Lab2BGR
        ).reshape(2, 3)
        self.centroids_bgr = bgr
        self._listo = True


# ============================================================
# TRACK
# ============================================================

@dataclass
class Track:
    track_id: int
    equipo: int = EQUIPO_NA
    cx: float = 0.0
    cy: float = 0.0
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    conf: float = 0.0
    hist: deque = field(default_factory=lambda: deque(maxlen=HIST_LEN))
    votos_a: int = 0
    votos_b: int = 0
    dist_total_m: float = 0.0
    vel_actual_kmh: float = 0.0

    def equipo_estable(self):
        if self.votos_a == 0 and self.votos_b == 0:
            return EQUIPO_NA
        return EQUIPO_A if self.votos_a >= self.votos_b else EQUIPO_B


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

    def __init__(self, frame_w, frame_h, modelo=MODELO_YOLO):
        from ultralytics import YOLO
        self.model   = YOLO(modelo)
        self.frame_w = frame_w
        self.frame_h = frame_h

        self.m_por_px_x = CANCHA_LARGO_M / frame_w
        self.m_por_px_y = CANCHA_ANCHO_M / frame_h

        self.clf = ClasificadorEquipo()

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

        res = self.model.track(
            frame_bgr,
            persist=True,
            classes=[COCO_PERSON, COCO_BALL],
            conf=CONF_MIN,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        r = res[0]
        ids_vistos = set()
        self.balon = None
        dets_pers  = []

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

                tid = int(box.id[0])
                ids_vistos.add(tid)
                dets_pers.append((x1, y1, x2, y2))

                tr = self.tracks.get(tid)
                if tr is None:
                    tr = Track(track_id=tid)
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

                if self.clf.listo():
                    eq = self.clf.clasificar(frame_bgr, x1, y1, x2, y2)
                    if eq == EQUIPO_A:   tr.votos_a += 1
                    elif eq == EQUIPO_B: tr.votos_b += 1
                    tr.equipo = tr.equipo_estable()

        if not self.clf.listo():
            self.clf.warm_up(frame_bgr, dets_pers)

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
        self._actualizar_posesion()
        self._actualizar_stats()

        return self._estado()

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

        nuevo_id = (mejor_tr.track_id
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

        return {
            "tracks":           list(self.tracks.values()),
            "balon":            self.balon_vis,
            "count_a":          ca,
            "count_b":          cb,
            "pos_a":            pos_a,
            "pos_b":            pos_b,
            "zona":             self._zona_3x3(),
            "id_posesion":      self.id_posesion,
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
