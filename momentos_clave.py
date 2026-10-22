"""
momentos_clave.py
=================
Deteccion de GOL y CELEBRACION sin entrenar un modelo propio.

En vez de confiar en una sola senal (que falla con mala luz, camara muy
cercana o angulos raros) se combinan varias senales DEBILES e
independientes y se razona sobre su orden en el tiempo:

  Senales visuales (siempre activas)
    - Red de porteria: textura de malla blanca sobre el cesped.
      "red_cercana" = la camara encuadra la porteria (remate / gol).
    - Tipo de plano (filtro_campo.py): las celebraciones casi siempre se
      muestran en primer plano y sin balon en juego.
    - Publico en primer plano: la parte inferior del frame deja de ser cesped.
    - Agrupamiento: varios jugadores del mismo equipo juntos (abrazo).
    - Brazos arriba (opcional, modelo de pose de YOLO).

  Senales de audio (opcionales)
    - Energia del audio: el estadio y el narrador suben de volumen en un gol.
      Si el audio es musica (clips editados) se detecta como "plano" y se ignora.
    - Transcripcion del narrador con Whisper (faster-whisper) y busqueda de
      palabras clave ("gol", "golazo", "goool"...). Es la senal mas fuerte
      en transmisiones reales y es procesamiento de lenguaje natural.

  Razonamiento temporal
    OCASION (porteria encuadrada / balon cerca de la red)
        -> en los siguientes ~10 s aparece una CELEBRACION sostenida
        => GOL (confianza media)
    + palabra "gol" del narrador o pico de audio => confianza alta
    CELEBRACION sin ninguna evidencia previa => solo se narra la celebracion.
"""

import os
import re
import shutil
import subprocess
import threading
from collections import deque

import cv2
import numpy as np

from filtro_campo import PLANO_GENERAL, PLANO_MEDIO, PLANO_PRIMER, PLANO_FUERA

try:
    from tracker_engine import EQUIPO_A, EQUIPO_B
except ImportError:
    EQUIPO_A, EQUIPO_B = 1, 2


# ============================================================
# CONFIGURACION
# ============================================================

USAR_AUDIO = True           # energia del audio (requiere ffmpeg o imageio-ffmpeg)
USAR_TRANSCRIPCION = False  # Whisper (pip install faster-whisper). Lento en CPU.
MODELO_WHISPER = "small"    # tiny / base / small / medium
USAR_POSE = True            # brazos arriba con yolov8n-pose.pt (se descarga solo)
MODELO_POSE = "yolov8n-pose.pt"

VENTANA_OCASION_S = 12.0    # segundos entre ocasion y celebracion para inferir gol
CELEB_MIN_S = 1.2           # duracion minima para confirmar celebracion
CELEB_UMBRAL_ON = 0.50
CELEB_UMBRAL_OFF = 0.30
ENFRIAMIENTO_GOL_S = 25.0   # no detectar dos goles seguidos

PATRON_GOL = re.compile(r"\bg+o+l+(azo|es)?\b|\bgo{2,}l*\b|\bgolazo\b", re.IGNORECASE)


# ============================================================
# 1) AUDIO: ENERGIA (EMOCION DEL ESTADIO / NARRADOR)
# ============================================================

def _ruta_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def extraer_audio_mono(ruta_video, sr=16000):
    """Devuelve el audio del video como np.float32 mono a `sr` Hz, o None."""
    ffmpeg = _ruta_ffmpeg()
    if ffmpeg is None:
        return None
    cmd = [ffmpeg, "-v", "error", "-i", ruta_video, "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "s16le", "-"]
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             timeout=300, check=False).stdout
    except Exception:
        return None
    if not out:
        return None
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


class AnalizadorAudio:
    """
    Calcula la "excitacion" del audio en cada instante:
    energia RMS comparada con la mediana de los ~20 s alrededor.
    """

    PASO_S = 0.25

    def __init__(self, ruta_video, sr=16000):
        self.disponible = False
        self.motivo = ""
        self.excitacion = None
        audio = extraer_audio_mono(ruta_video, sr) if USAR_AUDIO else None
        if audio is None or len(audio) < sr:
            self.motivo = "sin audio" if USAR_AUDIO else "desactivado"
            return
        paso = int(sr * self.PASO_S)
        n = len(audio) // paso
        rms = np.sqrt((audio[:n * paso].reshape(n, paso) ** 2).mean(axis=1) + 1e-9)
        db = 20 * np.log10(rms)

        # Audio "plano" (musica de fondo, clip editado): no sirve como senal
        if np.std(db) < 1.5:
            self.motivo = "audio sin variacion (musica/editado)"
            return

        ventana = max(3, int(20.0 / self.PASO_S))
        base = np.array([np.median(db[max(0, i - ventana // 2): i + ventana // 2 + 1])
                         for i in range(n)])
        mad = np.median(np.abs(db - np.median(db))) + 1e-6
        z = (db - base) / (1.4826 * mad)
        # z >= 2.5 se considera emocion fuerte; se lleva a 0..1
        self.excitacion = np.clip((z - 1.0) / 2.5, 0.0, 1.0)
        self.disponible = True

    def en(self, t):
        if not self.disponible:
            return 0.0
        i = int(t / self.PASO_S)
        if i < 0 or i >= len(self.excitacion):
            return 0.0
        return float(self.excitacion[i])

    def maximo_en(self, t0, t1):
        if not self.disponible:
            return 0.0
        i0 = max(0, int(t0 / self.PASO_S))
        i1 = min(len(self.excitacion), int(t1 / self.PASO_S) + 1)
        return float(self.excitacion[i0:i1].max()) if i1 > i0 else 0.0


# ============================================================
# 2) TRANSCRIPCION DEL NARRADOR (WHISPER) EN SEGUNDO PLANO
# ============================================================

class TranscriptorNarrador:
    """
    Transcribe el audio con faster-whisper en un hilo y guarda los instantes
    donde el narrador dice "gol", "golazo", "gooool", etc.
    """

    def __init__(self, ruta_video):
        self.disponible = False
        self.terminado = False
        self.instantes_gol = []     # lista de (t, palabra)
        self.segmentos = []         # lista de (t0, t1, texto)
        self.motivo = ""
        if not USAR_TRANSCRIPCION:
            self.motivo = "desactivado"
            return
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            self.motivo = "falta faster-whisper"
            return
        self.disponible = True
        threading.Thread(target=self._trabajar, args=(ruta_video,), daemon=True).start()

    def _trabajar(self, ruta_video):
        try:
            from faster_whisper import WhisperModel
            modelo = WhisperModel(MODELO_WHISPER, device="cpu", compute_type="int8")
            segmentos, _ = modelo.transcribe(ruta_video, language="es",
                                             word_timestamps=True, vad_filter=True)
            for seg in segmentos:
                self.segmentos.append((seg.start, seg.end, seg.text.strip()))
                for w in (seg.words or []):
                    if PATRON_GOL.search(w.word):
                        self.instantes_gol.append((w.start, w.word.strip()))
        except Exception as exc:
            self.motivo = f"error: {exc}"
        finally:
            self.terminado = True

    def gol_entre(self, t0, t1):
        for t, palabra in list(self.instantes_gol):
            if t0 <= t <= t1:
                return palabra
        return None


# ============================================================
# 3) RED DE PORTERIA (TEXTURA DE MALLA)
# ============================================================

class DetectorRed:
    """
    La red es una malla de lineas blancas finas con muchos "huecos" pequenos.
    Se busca por celdas: lineas finas claras (top-hat) + varios huecos +
    algo de cesped detras (descarta la grada, que tambien tiene textura).
    """

    CELDA = 12           # px, sobre la imagen a media resolucion

    def __init__(self):
        self._k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.ratio = 0.0
        self.caja = None     # (x1, y1, x2, y2) en coordenadas del frame

    def procesar(self, frame_bgr, mask_cesped):
        H, W = frame_bgr.shape[:2]
        peq = cv2.resize(frame_bgr, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        ces = cv2.resize(mask_cesped, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
        gris = cv2.cvtColor(peq, cv2.COLOR_BGR2GRAY)
        sat = cv2.cvtColor(peq, cv2.COLOR_BGR2HSV)[..., 1]
        th = cv2.morphologyEx(gris, cv2.MORPH_TOPHAT, self._k)
        lineas = ((th > 28) & (sat < 80)).astype(np.uint8)

        C = self.CELDA
        fh, fw = lineas.shape[0] // C, lineas.shape[1] // C
        if fh == 0 or fw == 0:
            self.ratio, self.caja = 0.0, None
            return
        l = lineas[:fh * C, :fw * C].reshape(fh, C, fw, C).mean(axis=(1, 3))
        c = (ces[:fh * C, :fw * C] > 0).reshape(fh, C, fw, C).mean(axis=(1, 3))
        candidatas = np.argwhere((l > 0.08) & (l < 0.6) & (c > 0.15))

        celdas = np.zeros((fh, fw), np.uint8)
        for i, j in candidatas:
            bloque = (1 - lineas[i * C:(i + 1) * C, j * C:(j + 1) * C]) * 255
            cs, _ = cv2.findContours(bloque.astype(np.uint8), cv2.RETR_LIST,
                                     cv2.CHAIN_APPROX_SIMPLE)
            huecos = sum(1 for k in cs if 1 < cv2.contourArea(k) < C * C * 0.3)
            if huecos >= 3:
                celdas[i, j] = 1

        self.ratio = float(celdas.sum()) / celdas.size
        self.caja = None
        if celdas.sum() >= 6:
            n, lab, st, _ = cv2.connectedComponentsWithStats(
                cv2.dilate(celdas, np.ones((3, 3), np.uint8)), 8)
            if n > 1:
                k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                x, y, w, h = st[k, :4]
                esc = 2 * C
                self.caja = (int(x * esc), int(y * esc), int((x + w) * esc), int((y + h) * esc))

    @property
    def visible(self):
        return self.ratio >= 0.015

    @property
    def cercana(self):
        return self.ratio >= 0.05

    def balon_en_red(self, balon, margen=0.10):
        if balon is None or self.caja is None:
            return False
        x1, y1, x2, y2 = self.caja
        mx, my = (x2 - x1) * margen, (y2 - y1) * margen
        return (x1 - mx) <= balon[0] <= (x2 + mx) and (y1 - my) <= balon[1] <= (y2 + my)


# ============================================================
# 4) POSE: BRAZOS ARRIBA (OPCIONAL)
# ============================================================

class DetectorBrazosArriba:

    def __init__(self):
        self.modelo = None
        if not USAR_POSE:
            return
        try:
            from ultralytics import YOLO
            self.modelo = YOLO(MODELO_POSE)
        except Exception:
            self.modelo = None

    def contar(self, frame_bgr, cajas_validas):
        """Cuantas personas (sobre el campo) tienen al menos un brazo arriba."""
        if self.modelo is None or not cajas_validas:
            return 0
        try:
            r = self.modelo(frame_bgr, verbose=False, conf=0.35)[0]
        except Exception:
            return 0
        if r.keypoints is None or r.boxes is None:
            return 0
        kp = r.keypoints.data.cpu().numpy()      # (n, 17, 3)
        cajas = r.boxes.xyxy.cpu().numpy()
        n = 0
        for caja, k in zip(cajas, kp):
            cx, cy = (caja[0] + caja[2]) / 2, (caja[1] + caja[3]) / 2
            # solo personas que coinciden con un jugador aceptado por el filtro de campo
            if not any(x1 <= cx <= x2 and y1 <= cy <= y2 for (x1, y1, x2, y2) in cajas_validas):
                continue
            alto = caja[3] - caja[1]
            for hombro, muneca in ((5, 9), (6, 10)):
                if k[hombro, 2] > 0.4 and k[muneca, 2] > 0.4 and \
                        k[muneca, 1] < k[hombro, 1] - 0.12 * alto:
                    n += 1
                    break
        return n


# ============================================================
# 5) DETECTOR PRINCIPAL
# ============================================================

class DetectorMomentosClave:

    def __init__(self):
        self.red = DetectorRed()
        self.pose = DetectorBrazosArriba()
        self.audio = None
        self.asr = None
        self.reset()

    # ---------------- ciclo de vida ----------------

    def reset(self):
        self.puntaje_celeb = 0.0
        self.celebrando = False
        self.t_inicio_celeb_cand = None
        self.t_bajo_umbral = None
        self.t_ultima_ocasion = -1e9
        self.ocasion_fuerte = False
        self.t_ultimo_gol = -1e9
        self.t_ultimo_balon = 0.0
        self.frame_i = 0
        self.brazos = 0
        self.equipos_celeb = deque(maxlen=90)
        self.protagonistas = deque(maxlen=90)
        self.senales = {}

    def cargar_video(self, ruta_video):
        """Prepara las senales de audio (en segundo plano si es Whisper)."""
        self.reset()
        self.audio = AnalizadorAudio(ruta_video)
        self.asr = TranscriptorNarrador(ruta_video)

    def descripcion_senales(self):
        partes = []
        if self.audio is not None:
            partes.append("audio: " + ("activo" if self.audio.disponible else self.audio.motivo))
        if self.asr is not None:
            if self.asr.disponible:
                partes.append("narrador: " + ("listo" if self.asr.terminado else "transcribiendo..."))
            else:
                partes.append("narrador: " + self.asr.motivo)
        partes.append("pose: " + ("activa" if self.pose.modelo is not None else "no disponible"))
        return " | ".join(partes)

    # ---------------- por frame ----------------

    def procesar(self, frame_bgr, estado, t, mask_cesped):
        """
        frame_bgr: frame original (sin dibujos)
        estado: dict de MotorTracking
        t: tiempo del video en segundos
        Devuelve una lista de eventos (puede estar vacia).
        """
        self.frame_i += 1
        eventos = []
        plano = estado.get("plano", PLANO_GENERAL)
        tracks = estado.get("tracks", [])
        balon = estado.get("balon")
        if estado.get("balon_detectado"):
            self.t_ultimo_balon = t

        # ---- red de porteria (cada 3 frames) ----
        if self.frame_i % 3 == 0 and mask_cesped is not None:
            self.red.procesar(frame_bgr, mask_cesped)
        if self.red.cercana or (self.red.visible and self.red.balon_en_red(balon)):
            self.t_ultima_ocasion = t
            self.ocasion_fuerte = self.ocasion_fuerte or self.red.balon_en_red(balon)

        # ---- pose (cada 4 frames, solo en planos cercanos) ----
        if plano in (PLANO_PRIMER, PLANO_MEDIO) and self.frame_i % 4 == 0:
            cajas = [(tr.x1, tr.y1, tr.x2, tr.y2) for tr in tracks]
            self.brazos = self.pose.contar(frame_bgr, cajas)
        elif plano == PLANO_GENERAL:
            self.brazos = 0

        # ---- puntaje de celebracion ----
        s = {}
        s["plano"] = {PLANO_PRIMER: 0.35, PLANO_FUERA: 0.20, PLANO_MEDIO: 0.10}.get(plano, 0.0)
        s["sin_balon"] = 0.20 if (t - self.t_ultimo_balon) > 1.5 else 0.0
        ces_tot = estado.get("ratio_cesped", 0.0)
        ces_inf = estado.get("ratio_cesped_inf", ces_tot)
        s["publico"] = 0.15 if (ces_tot > 0.25 and ces_inf < 0.55 * ces_tot) else 0.0
        s["brazos"] = min(0.30, 0.20 * self.brazos)
        s["abrazo"] = 0.15 if self._hay_agrupamiento(tracks) else 0.0
        s["audio"] = 0.30 * (self.audio.en(t) if self.audio else 0.0)
        palabra = self.asr.gol_entre(t - 8.0, t) if (self.asr and self.asr.disponible) else None
        s["narrador"] = 0.40 if palabra else 0.0
        # Contexto: hubo ocasion (porteria encuadrada) hace poco. Sin ese
        # contexto, un primer plano por si solo pesa menos (puede ser una
        # falta, un cambio, el entrenador...).
        if (t - self.t_ultima_ocasion) <= VENTANA_OCASION_S:
            s["contexto"] = 0.15
        elif not palabra:
            s["plano"] *= 0.6
        self.senales = s

        bruto = min(1.0, sum(s.values()))
        self.puntaje_celeb = 0.85 * self.puntaje_celeb + 0.15 * bruto

        # Equipo y jugador protagonista durante la celebracion
        if plano in (PLANO_PRIMER, PLANO_MEDIO) and tracks:
            eqs = [tr.equipo for tr in tracks if tr.equipo in (EQUIPO_A, EQUIPO_B)]
            if eqs:
                self.equipos_celeb.append(max(set(eqs), key=eqs.count))
            mayor = max(tracks, key=lambda tr: tr.h)
            if getattr(mayor, "dorsal", None) is not None:
                self.protagonistas.append((mayor.dorsal, mayor.equipo))

        # ---- maquina de estados ----
        if not self.celebrando:
            if self.puntaje_celeb >= CELEB_UMBRAL_ON:
                if self.t_inicio_celeb_cand is None:
                    self.t_inicio_celeb_cand = t
                elif t - self.t_inicio_celeb_cand >= CELEB_MIN_S:
                    self.celebrando = True
                    self.t_bajo_umbral = None
                    eventos.extend(self._al_iniciar_celebracion(t, palabra))
            elif self.puntaje_celeb < CELEB_UMBRAL_ON - 0.10:
                self.t_inicio_celeb_cand = None
        else:
            if self.puntaje_celeb < CELEB_UMBRAL_OFF:
                if self.t_bajo_umbral is None:
                    self.t_bajo_umbral = t
                elif t - self.t_bajo_umbral > 1.5:
                    self.celebrando = False
                    self.t_inicio_celeb_cand = None
                    self.ocasion_fuerte = False
            else:
                self.t_bajo_umbral = None

        # Gol anunciado por el narrador aunque la imagen no ayude
        if palabra and not self.celebrando and (t - self.t_ultimo_gol) > ENFRIAMIENTO_GOL_S \
                and (self.audio is None or not self.audio.disponible or
                     self.audio.maximo_en(t - 6, t + 0.5) > 0.4):
            eventos.append(self._crear_gol(t, "alta", f"narrador: '{palabra}'"))

        return eventos

    # ---------------- auxiliares ----------------

    def _hay_agrupamiento(self, tracks):
        """3+ jugadores del mismo equipo muy juntos (abrazo / pina)."""
        por_eq = {}
        for tr in tracks:
            if tr.equipo in (EQUIPO_A, EQUIPO_B):
                por_eq.setdefault(tr.equipo, []).append(tr)
        for lista in por_eq.values():
            if len(lista) < 3:
                continue
            alt = np.median([tr.h for tr in lista])
            xs = np.array([tr.cx for tr in lista])
            ys = np.array([tr.cy for tr in lista])
            if np.hypot(xs - xs.mean(), ys - ys.mean()).max() < 0.9 * alt:
                return True
        return False

    def _equipo_celebrando(self):
        if not self.equipos_celeb:
            return None
        l = list(self.equipos_celeb)
        return max(set(l), key=l.count)

    def _protagonista(self, equipo):
        cand = [d for d, e in self.protagonistas if e == equipo or equipo is None]
        if not cand:
            return None
        return max(set(cand), key=cand.count)

    def _crear_gol(self, t, confianza, motivo):
        equipo = self._equipo_celebrando()
        self.t_ultimo_gol = t
        return {
            "tipo": "GOL",
            "equipo": equipo,
            "jugador": self._protagonista(equipo),
            "jugador_anterior": None,
            "equipo_anterior": None,
            "zona": None,
            "confianza": confianza,
            "motivo": motivo,
            "t_video": t,
        }

    def _al_iniciar_celebracion(self, t, palabra):
        eventos = []
        hubo_ocasion = (t - self.t_ultima_ocasion) <= VENTANA_OCASION_S
        audio_alto = self.audio.maximo_en(t - 10, t) > 0.6 if self.audio else False
        puede_gol = (t - self.t_ultimo_gol) > ENFRIAMIENTO_GOL_S

        if puede_gol and (hubo_ocasion or palabra or audio_alto):
            motivos = []
            if hubo_ocasion:
                motivos.append("balon en la red" if self.ocasion_fuerte else "porteria encuadrada")
            if palabra:
                motivos.append(f"narrador: '{palabra}'")
            if audio_alto:
                motivos.append("pico de audio")
            fuertes = int(self.ocasion_fuerte) + int(bool(palabra)) + int(audio_alto)
            confianza = "alta" if fuertes >= 1 and len(motivos) >= 2 else "media"
            eventos.append(self._crear_gol(t, confianza, " + ".join(motivos) + " + celebracion"))

        equipo = self._equipo_celebrando()
        eventos.append({
            "tipo": "CELEBRACION",
            "equipo": equipo,
            "jugador": self._protagonista(equipo),
            "jugador_anterior": None,
            "equipo_anterior": None,
            "zona": None,
            "confianza": "media",
            "motivo": ", ".join(k for k, v in self.senales.items() if v > 0),
            "t_video": t,
        })
        return eventos
