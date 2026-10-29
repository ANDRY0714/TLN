"""
main.py  v6
===========
- Video limpio: solo puntos de jugadores, etiqueta (dorsal / velocidad) y balon.
  Las narraciones ya no se dibujan encima del video.
- Panel lateral con selector:
      [ Eventos ]  -> marcador, estado del plano, narracion del balon,
                      ultimo evento e historial (GOL / CELEBRACION incluidos)
      [ Analisis tactico ] -> el panel de estadisticas de siempre
- Filtro de campo (filtro_campo.py): se descartan personas fuera del terreno.
- Deteccion de GOL y CELEBRACION (momentos_clave.py).
- Casilla "Ver campo" para depurar: contorno del campo detectado y red.
- v7: panel lateral mas ancho, letra grande y con barra de desplazamiento;
      colores de camiseta de partido.toml (EQUIPO_A = local = rojo);
      voz del narrador (voz.py) con casilla "Voz" para silenciarla.
"""

import time
import tkinter as tk
from tkinter import filedialog

import cv2
from PIL import Image, ImageTk

from tracker_engine import MotorTracking, EQUIPO_A, EQUIPO_B
from event_detector import DetectorEventos
from narrator import NarradorFutbol
from jersey_recognizer import ReconocedorDorsal
from momentos_clave import DetectorMomentosClave
from filtro_campo import NOMBRES_PLANO
from voz import VozNarrador, RELLENO, EVENTO, URGENTE

VIDEO_WIDTH = 960
VIDEO_HEIGHT = 540
PANEL_W = 580
WRAP = PANEL_W - 70   # ancho de texto dentro de las tarjetas

# Tipografia del panel de eventos (antes 7-11 pt, no se leia bien)
F_TITULO = ("Segoe UI", 20, "bold")
F_MARCADOR = ("Segoe UI", 24, "bold")
F_SECCION = ("Segoe UI", 13, "bold")
F_ESTADO = ("Segoe UI", 13)
F_SENALES = ("Segoe UI", 11)
F_TEXTO = ("Segoe UI", 16)
F_EVENTO_TIT = ("Segoe UI", 20, "bold")
F_EVENTO_TXT = ("Segoe UI", 17)
F_HIST = ("Segoe UI", 14)
F_HIST_TIPO = ("Segoe UI", 14, "bold")
F_HIST_T = ("Consolas", 12)
F_HIST_MOTIVO = ("Segoe UI", 12, "italic")

# Paleta
BG = "#0F172A"
BG_PANEL = "#111827"
BG_CARD = "#1F2937"
TXT = "#E5E7EB"
TXT_DIM = "#9CA3AF"
ACENTO = "#2563EB"

COLOR_EVENTO = {
    "GOL": "#F59E0B",
    "CELEBRACION": "#F472B6",
    "PASE": "#34D399",
    "CAMBIO_POSESION": "#60A5FA",
    "POSESION_INICIAL": "#A78BFA",
    "BALON_LIBRE": "#9CA3AF",
}
TITULO_EVENTO = {
    "GOL": "GOL",
    "CELEBRACION": "CELEBRACION",
    "PASE": "PASE",
    "CAMBIO_POSESION": "CAMBIO DE POSESION",
    "POSESION_INICIAL": "POSESION",
    "BALON_LIBRE": "BALON EN DISPUTA",
}

INTERVALO_NARRACION_BALON_S = 1.5
SILENCIO_ANTES_DE_RELLENO_S = 4.0   # la voz narra el balon si lleva rato callada
PRIORIDAD_VOZ = {"GOL": URGENTE, "CELEBRACION": URGENTE, "BALON_LIBRE": RELLENO}


def mmss(t):
    t = max(0, int(t))
    return f"{t // 60:02d}:{t % 60:02d}"


class DetectorFutbol:
    def __init__(self, root):
        self.root = root
        self.root.title("PRO - Sistema PRO - Analisis de Futbol con IA")
        self.root.geometry("1600x900")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar)

        self.cap = None
        self.ruta_video = None
        self.running = False
        self.paused = False
        self.motor = None

        self.detector_eventos = DetectorEventos()
        self.narrador_futbol = NarradorFutbol()
        self.reconocedor_dorsales = ReconocedorDorsal()
        self.reconocedor_dorsales.sincronizar_con_narrador(self.narrador_futbol)
        self.momentos = DetectorMomentosClave()
        self.voz = VozNarrador()
        self.colores_equipo = self._colores_camiseta()

        self.proc_w = VIDEO_WIDTH
        self.proc_h = VIDEO_HEIGHT
        self.last_time = time.time()
        self.fps = 0
        self.t_video = 0.0
        self.estado = {}
        self.tkimg = None

        self._reset_narracion()

        self.ver_campo = tk.BooleanVar(value=False)
        self.voz_activa = tk.BooleanVar(value=True)
        self.panel_activo = "eventos"
        self.crear_ui()

    def _colores_camiseta(self):
        """{EQUIPO_A: [...], EQUIPO_B: [...]} con la clave 'camiseta' de partido.toml."""
        red = self.narrador_futbol.red_semantica
        datos = getattr(red, "_datos_partido_cargados", None) or {}
        colores = {}
        for clave, info in (datos.get("equipos") or {}).items():
            if isinstance(info, dict) and info.get("camiseta"):
                eq = red._resolver_equipo_id(info.get("id", clave))
                if eq in (EQUIPO_A, EQUIPO_B):
                    colores[eq] = info["camiseta"]
        return colores

    def cerrar(self):
        self.running = False
        self.voz.cerrar()
        self.root.destroy()

    def _reset_narracion(self):
        self.historial_eventos = []
        self.ultimo_evento = None
        self.ultima_narracion = ""
        self._balon_ref = None
        self._t_narr_balon = -1e9
        self._t_destacado = -1e9

    # =========================================================
    # UI
    # =========================================================

    def crear_ui(self):
        top = tk.Frame(self.root, bg=BG_PANEL, height=55)
        top.pack(fill=tk.X, side=tk.TOP)
        top.pack_propagate(False)

        btn_s = dict(bg=ACENTO, fg="white", relief=tk.FLAT, padx=14, pady=8,
                     font=("Segoe UI", 10, "bold"), cursor="hand2",
                     activebackground="#1D4ED8")

        tk.Button(top, text="Abrir Video", command=self.abrir_video, **btn_s).pack(
            side=tk.LEFT, padx=8, pady=8)
        self.btn_pausa = tk.Button(top, text="Pausa", command=self.toggle_pausa, **btn_s)
        self.btn_pausa.pack(side=tk.LEFT, padx=4, pady=8)
        tk.Button(top, text="Reiniciar", command=self.reiniciar_video, **btn_s).pack(
            side=tk.LEFT, padx=4, pady=8)
        tk.Checkbutton(top, text="Ver campo", variable=self.ver_campo,
                       bg=BG_PANEL, fg=TXT, selectcolor=BG_CARD,
                       activebackground=BG_PANEL, activeforeground=TXT,
                       font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(top, text="Voz", variable=self.voz_activa,
                       command=lambda: self.voz.set_activa(self.voz_activa.get()),
                       bg=BG_PANEL, fg=TXT, selectcolor=BG_CARD,
                       activebackground=BG_PANEL, activeforeground=TXT,
                       font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=4)
        self.lbl_status = tk.Label(top, text="", bg=BG_PANEL, fg="#FBBF24",
                                   font=("Segoe UI", 10, "bold"))
        self.lbl_status.pack(side=tk.LEFT, padx=12)

        # Selector de panel (derecha)
        sel = tk.Frame(top, bg=BG_PANEL)
        sel.pack(side=tk.RIGHT, padx=10)
        self.btn_tab = {}
        for clave, texto in (("eventos", "Eventos"), ("tactico", "Analisis tactico")):
            b = tk.Button(sel, text=texto, relief=tk.FLAT, padx=12, pady=6,
                          font=("Segoe UI", 9, "bold"), cursor="hand2",
                          command=lambda c=clave: self.mostrar_panel(c))
            b.pack(side=tk.LEFT, padx=2)
            self.btn_tab[clave] = b

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True)

        self.video_canvas = tk.Canvas(main, bg="black", highlightthickness=0)
        self.video_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Panel lateral con barra de desplazamiento (el contenido ya no cabe)
        lateral = tk.Frame(main, bg=BG_PANEL, width=PANEL_W)
        lateral.pack(side=tk.RIGHT, fill=tk.Y)
        lateral.pack_propagate(False)
        barra = tk.Scrollbar(lateral, orient=tk.VERTICAL)
        barra.pack(side=tk.RIGHT, fill=tk.Y)
        self.cv_lateral = tk.Canvas(lateral, bg=BG_PANEL, highlightthickness=0,
                                    yscrollcommand=barra.set)
        self.cv_lateral.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        barra.config(command=self.cv_lateral.yview)
        cont = tk.Frame(self.cv_lateral, bg=BG_PANEL)
        ventana = self.cv_lateral.create_window(0, 0, window=cont, anchor="nw")
        cont.bind("<Configure>", lambda e: self.cv_lateral.configure(
            scrollregion=self.cv_lateral.bbox("all")))
        self.cv_lateral.bind("<Configure>", lambda e: self.cv_lateral.itemconfig(
            ventana, width=e.width))
        self.lateral = lateral
        self.root.bind_all("<MouseWheel>", self._rueda)

        self.panel_eventos = tk.Frame(cont, bg=BG_PANEL)
        self.panel_tactico = tk.Frame(cont, bg=BG_PANEL)
        self._build_panel_eventos(self.panel_eventos)
        self._build_panel(self.panel_tactico)
        self.mostrar_panel("eventos")

    def mostrar_panel(self, clave):
        self.panel_activo = clave
        for c, frame in (("eventos", self.panel_eventos), ("tactico", self.panel_tactico)):
            if c == clave:
                frame.pack(fill=tk.BOTH, expand=True)
            else:
                frame.pack_forget()
        self.cv_lateral.yview_moveto(0)
        for c, b in self.btn_tab.items():
            activo = c == clave
            b.config(bg=ACENTO if activo else BG_CARD,
                     fg="white" if activo else TXT_DIM,
                     activebackground="#1D4ED8" if activo else "#374151")

    def _rueda(self, event):
        # El historial tiene su propio scroll: ahi la rueda mueve solo el historial
        w = self.root.winfo_containing(event.x_root, event.y_root)
        if w is None or w is self.txt_hist or not str(w).startswith(str(self.lateral)):
            return
        self.cv_lateral.yview_scroll(int(-event.delta / 120), "units")

    def _nombre(self, equipo):
        return self.narrador_futbol.nombre_equipo(equipo)

    def _sep(self, p):
        tk.Frame(p, bg="#374151", height=1).pack(fill=tk.X, padx=8, pady=3)

    def _titulo_seccion(self, p, texto):
        tk.Label(p, text=texto, bg=BG_PANEL, fg=TXT_DIM,
                 font=F_SECCION, anchor="w").pack(fill=tk.X, padx=14, pady=(14, 4))

    # ---------------- Panel de eventos ----------------

    def _build_panel_eventos(self, p):
        tk.Label(p, text="EVENTOS DEL PARTIDO", bg=BG_PANEL, fg="white",
                 font=F_TITULO).pack(pady=(14, 6))

        # Marcador
        self.lbl_marcador = tk.Label(p, text=self.narrador_futbol.marcador_texto(),
                                     bg=BG_CARD, fg="white", font=F_MARCADOR, pady=12)
        self.lbl_marcador.pack(fill=tk.X, padx=14, pady=(2, 8))

        # Ultimo evento (lo mas importante va arriba)
        self._titulo_seccion(p, "ULTIMO EVENTO")
        self.card_evento = tk.Frame(p, bg=BG_CARD, highlightthickness=3,
                                    highlightbackground=BG_CARD)
        self.card_evento.pack(fill=tk.X, padx=14)
        self.lbl_evento_tit = tk.Label(self.card_evento, text="--", bg=BG_CARD, fg="#FBBF24",
                                       font=F_EVENTO_TIT, anchor="w", justify="left",
                                       wraplength=WRAP, padx=14)
        self.lbl_evento_tit.pack(fill=tk.X, pady=(12, 0))
        self.lbl_evento_txt = tk.Label(self.card_evento, text="", bg=BG_CARD, fg="white",
                                       font=F_EVENTO_TXT, anchor="nw", justify="left",
                                       wraplength=WRAP, padx=14, pady=10, height=5)
        self.lbl_evento_txt.pack(fill=tk.X)

        # Balon
        self._titulo_seccion(p, "BALON")
        self.lbl_balon = tk.Label(p, text="--", bg=BG_CARD, fg=TXT, font=F_TEXTO,
                                  anchor="nw", justify="left", wraplength=WRAP,
                                  padx=14, pady=10, height=3)
        self.lbl_balon.pack(fill=tk.X, padx=14)

        # Estado de la camara / juego
        self._titulo_seccion(p, "ESTADO")
        fila = tk.Frame(p, bg=BG_PANEL)
        fila.pack(fill=tk.X, padx=14)
        self.lbl_plano = tk.Label(fila, text="Plano: --", bg=BG_PANEL, fg=TXT,
                                  font=F_ESTADO, anchor="w", justify="left",
                                  wraplength=WRAP - 90)
        self.lbl_plano.pack(side=tk.LEFT)
        self.lbl_tiempo = tk.Label(fila, text="00:00", bg=BG_PANEL, fg="white",
                                   font=("Consolas", 16, "bold"))
        self.lbl_tiempo.pack(side=tk.RIGHT)

        # Medidor de celebracion
        fila2 = tk.Frame(p, bg=BG_PANEL)
        fila2.pack(fill=tk.X, padx=14, pady=(8, 0))
        tk.Label(fila2, text="Celebracion", bg=BG_PANEL, fg=TXT,
                 font=F_ESTADO).pack(side=tk.LEFT)
        self.cv_celeb = tk.Canvas(fila2, bg=BG_CARD, height=18, highlightthickness=0)
        self.cv_celeb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        self.lbl_senales = tk.Label(p, text="", bg=BG_PANEL, fg=TXT_DIM,
                                    font=F_SENALES, anchor="w", justify="left",
                                    wraplength=WRAP + 20)
        self.lbl_senales.pack(fill=tk.X, padx=14, pady=(6, 0))
        self.lbl_voz = tk.Label(p, text="", bg=BG_PANEL, fg="#34D399",
                                font=F_SENALES, anchor="w")
        self.lbl_voz.pack(fill=tk.X, padx=14)

        # Historial
        self._titulo_seccion(p, "HISTORIAL")
        marco = tk.Frame(p, bg=BG_PANEL)
        marco.pack(fill=tk.X, padx=14, pady=(0, 14))
        barra = tk.Scrollbar(marco)
        barra.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_hist = tk.Text(marco, bg=BG_CARD, fg=TXT, font=F_HIST, wrap="word",
                                relief=tk.FLAT, padx=12, pady=10, yscrollcommand=barra.set,
                                state="disabled", cursor="arrow", height=16, spacing1=2,
                                spacing3=2)
        self.txt_hist.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        barra.config(command=self.txt_hist.yview)
        self.txt_hist.tag_configure("tiempo", foreground=TXT_DIM, font=F_HIST_T)
        for tipo, col in COLOR_EVENTO.items():
            self.txt_hist.tag_configure(tipo, foreground=col, font=F_HIST_TIPO)
        self.txt_hist.tag_configure("motivo", foreground=TXT_DIM, font=F_HIST_MOTIVO)

    # ---------------- Panel tactico (el de siempre) ----------------

    def _build_panel(self, p):
        tk.Label(p, text="ANALISIS TACTICO", bg=BG_PANEL, fg="white",
                 font=("Segoe UI", 14, "bold")).pack(pady=(12, 2))
        self._sep(p)

        self.lbl_fps = tk.Label(p, text="FPS: --", bg=BG_PANEL, fg="#00FFAA",
                                font=("Consolas", 12, "bold"), anchor="w")
        self.lbl_fps.pack(fill=tk.X, padx=10, pady=2)
        self._sep(p)

        tk.Label(p, text="POSESION DEL BALON", bg=BG_PANEL, fg=TXT_DIM,
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(fill=tk.X, padx=10)
        self.bar_c = tk.Canvas(p, bg=BG_CARD, height=26, highlightthickness=0)
        self.bar_c.pack(fill=tk.X, padx=10, pady=3)
        self.lbl_pos = tk.Label(p, text="--", bg=BG_PANEL, fg="white",
                                font=("Segoe UI", 10, "bold"), anchor="center")
        self.lbl_pos.pack(fill=tk.X, padx=10)
        self._sep(p)

        ley = tk.Frame(p, bg=BG_PANEL)
        ley.pack(fill=tk.X, padx=10, pady=2)
        tk.Canvas(ley, bg="#DC2626", width=14, height=14, highlightthickness=0).pack(side=tk.LEFT)
        tk.Label(ley, text=f" {self._nombre(EQUIPO_A)} (rojo)  ", bg=BG_PANEL, fg="#FCA5A5",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Canvas(ley, bg=ACENTO, width=14, height=14, highlightthickness=0).pack(side=tk.LEFT)
        tk.Label(ley, text=f" {self._nombre(EQUIPO_B)} (azul)", bg=BG_PANEL, fg="#93C5FD",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self._sep(p)

        tk.Label(p, text="ESTADISTICAS COMPARATIVAS", bg=BG_PANEL, fg=TXT_DIM,
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(fill=tk.X, padx=10)
        cab = tk.Frame(p, bg=BG_CARD)
        cab.pack(fill=tk.X, padx=10, pady=(2, 0))
        tk.Label(cab, text="Metrica", bg=BG_CARD, fg="#6B7280", font=("Segoe UI", 8, "bold"),
                 width=14, anchor="w").grid(row=0, column=0)
        tk.Label(cab, text=self._nombre(EQUIPO_A)[:10], bg=BG_CARD, fg="#F87171", font=("Segoe UI", 8, "bold"),
                 width=9, anchor="center").grid(row=0, column=1)
        tk.Label(cab, text=self._nombre(EQUIPO_B)[:10], bg=BG_CARD, fg="#60A5FA", font=("Segoe UI", 8, "bold"),
                 width=9, anchor="center").grid(row=0, column=2)

        self._rows = {}
        metricas = [("Jugadores", "count"), ("Pases", "pases"), ("Precision %", "precision"),
                    ("Perdidas", "perdidas"), ("Vel.max km/h", "vel_max"),
                    ("Vel.prom km/h", "vel_prom"), ("Distancia m", "dist")]
        bgs = (BG_PANEL, "#161D2B")
        for i, (nom, key) in enumerate(metricas):
            bg = bgs[i % 2]
            f = tk.Frame(p, bg=bg)
            f.pack(fill=tk.X, padx=10)
            tk.Label(f, text=nom, bg=bg, fg="#D1D5DB", font=("Segoe UI", 8), width=14,
                     anchor="w").pack(side=tk.LEFT)
            va = tk.Label(f, text="--", bg=bg, fg="#FCA5A5", font=("Consolas", 9, "bold"),
                          width=9, anchor="center")
            va.pack(side=tk.LEFT)
            vb = tk.Label(f, text="--", bg=bg, fg="#93C5FD", font=("Consolas", 9, "bold"),
                          width=9, anchor="center")
            vb.pack(side=tk.LEFT)
            self._rows[key] = (va, vb)

        self._sep(p)
        tk.Label(p, text="EVENTOS DE JUEGO", bg=BG_PANEL, fg=TXT_DIM,
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(fill=tk.X, padx=10)
        self.lbl_quien = tk.Label(p, text="En posesion: --", bg=BG_PANEL, fg="#FBBF24",
                                  font=("Segoe UI", 10, "bold"), anchor="w")
        self.lbl_quien.pack(fill=tk.X, padx=10, pady=1)
        self.lbl_cambios = tk.Label(p, text="Cambios: 0", bg=BG_PANEL, fg="white",
                                    font=("Segoe UI", 9), anchor="w")
        self.lbl_cambios.pack(fill=tk.X, padx=10, pady=1)
        self.lbl_toques = tk.Label(p, text="Toques: 0", bg=BG_PANEL, fg="white",
                                   font=("Segoe UI", 9), anchor="w")
        self.lbl_toques.pack(fill=tk.X, padx=10, pady=1)
        self._sep(p)

        tk.Label(p, text="ZONA DE JUEGO", bg=BG_PANEL, fg=TXT_DIM,
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(fill=tk.X, padx=10)
        self.lbl_zona = tk.Label(p, text="--", bg=BG_PANEL, fg="#A78BFA",
                                 font=("Segoe UI", 12, "bold"), anchor="w")
        self.lbl_zona.pack(fill=tk.X, padx=10, pady=2)
        self._sep(p)

        tk.Label(p, text="RADAR", bg=BG_PANEL, fg=TXT_DIM, font=("Segoe UI", 8, "bold"),
                 anchor="w").pack(fill=tk.X, padx=10)
        self.radar_c = tk.Canvas(p, bg="#064E3B", width=PANEL_W - 30, height=90,
                                 highlightthickness=1, highlightbackground="#374151")
        self.radar_c.pack(padx=10, pady=4)

    # =========================================================
    # CONTROL DE VIDEO
    # =========================================================

    def _reiniciar_modulos(self):
        self.motor = MotorTracking(self.proc_w, self.proc_h,
                                   colores_equipo=self.colores_equipo)
        self.voz.callar()
        self.detector_eventos.reset()
        self.narrador_futbol.reset()
        self.reconocedor_dorsales.reset()
        self.momentos.reset()
        self._reset_narracion()
        self._limpiar_panel_eventos()

    def abrir_video(self):
        path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        self.ruta_video = path
        self.cap = cv2.VideoCapture(path)
        sw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        sh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if sw <= 0 or sh <= 0:
            sw, sh = VIDEO_WIDTH, VIDEO_HEIGHT
        esc = min(960 / sw, 960 / sh, 1.0)
        self.proc_w = max(int(sw * esc), 1)
        self.proc_h = max(int(sh * esc), 1)

        self.lbl_status.config(text="Analizando audio...")
        self.root.update_idletasks()
        self._reiniciar_modulos()
        self.momentos.cargar_video(path)

        self.running = True
        self.paused = False
        self.btn_pausa.config(text="Pausa")
        self.update_video()

    def toggle_pausa(self):
        if not self.cap:
            return
        self.paused = not self.paused
        self.btn_pausa.config(text=("Reanudar" if self.paused else "Pausa"))
        if self.paused:
            self.voz.callar()
        if not self.paused and self.running:
            self.update_video()

    def reiniciar_video(self):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        audio, asr = self.momentos.audio, self.momentos.asr
        self._reiniciar_modulos()
        self.momentos.audio, self.momentos.asr = audio, asr   # no recalcular audio
        self.paused = False
        self.btn_pausa.config(text="Pausa")
        if not self.running:
            self.running = True
            self.update_video()

    # =========================================================
    # DIBUJO SOBRE EL VIDEO (solo marcadores)
    # =========================================================

    def _bgr(self, equipo):
        if equipo == EQUIPO_A:
            return (0, 0, 220)
        if equipo == EQUIPO_B:
            return (220, 80, 0)
        return (160, 160, 160)

    def dibujar_jugadores(self, frame, tracks):
        for tr in tracks:
            col = self._bgr(tr.equipo)
            cx, cy = int(tr.cx), int(tr.cy)
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), 8, col, -1)
            vel = getattr(tr, "vel_actual_kmh", 0.0)
            dorsal = getattr(tr, "dorsal", None)
            label = f"#{dorsal}" if dorsal is not None else f"J{tr.uid}"
            if vel > 1.0:
                label += f" {vel:.1f}km/h"
            cv2.putText(frame, label, (cx + 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 3)
            cv2.putText(frame, label, (cx + 12, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    def dibujar_balon(self, frame, balon):
        if balon is None:
            return
        bx, by = int(balon[0]), int(balon[1])
        cv2.circle(frame, (bx, by), 10, (255, 255, 255), 2)
        cv2.circle(frame, (bx, by), 7, (0, 220, 220), -1)
        cv2.putText(frame, "BALON", (bx + 13, by + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3)
        cv2.putText(frame, "BALON", (bx + 13, by + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 220, 220), 1)

    def dibujar_depuracion(self, frame):
        """Contorno del campo detectado y caja de la red (casilla 'Ver campo')."""
        self.motor.campo.dibujar(frame, (0, 255, 255))
        caja = self.momentos.red.caja
        if caja is not None and self.momentos.red.visible:
            cv2.rectangle(frame, caja[:2], caja[2:], (255, 0, 255), 2)
            cv2.putText(frame, "RED", (caja[0], max(15, caja[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # =========================================================
    # NARRACION
    # =========================================================

    def _movimiento_balon(self, balon):
        if balon is None:
            return None
        if self._balon_ref is None:
            self._balon_ref = balon
            return "empieza a moverse"
        dx = balon[0] - self._balon_ref[0]
        dy = balon[1] - self._balon_ref[1]
        self._balon_ref = balon
        umbral = 0.05 * max(self.proc_w, self.proc_h)
        if abs(dx) < umbral and abs(dy) < umbral:
            return "se mantiene estable"
        if abs(dx) > abs(dy):
            return "avanza hacia la derecha" if dx > 0 else "se mueve hacia la izquierda"
        return "baja por el campo" if dy > 0 else "sube por el campo"

    def actualizar_narracion_balon(self, e):
        if self.t_video - self._t_narr_balon < INTERVALO_NARRACION_BALON_S:
            return
        self._t_narr_balon = self.t_video
        if not e.get("juego_activo", True):
            plano = NOMBRES_PLANO.get(e.get("plano"), "")
            self.lbl_balon.config(text=f"Juego no visible ({plano.lower()}).", fg=TXT_DIM)
            return
        balon = e.get("balon")
        if balon is None:
            self.lbl_balon.config(text="Balon sin seguimiento.", fg=TXT_DIM)
            return
        texto = self.narrador_futbol.narrar_estado_balon(
            e.get("eq_posesion", 0), self._movimiento_balon(balon), e.get("zona"))
        self.lbl_balon.config(text=texto, fg=TXT)
        if self.voz.segundos_callada() > SILENCIO_ANTES_DE_RELLENO_S:
            self.voz.decir(texto, RELLENO)

    def registrar_evento(self, evento):
        texto = self.narrador_futbol.narrar(evento)
        if not texto:
            return
        tipo = evento.get("tipo", "")
        self.ultimo_evento = evento
        self.ultima_narracion = texto
        self.historial_eventos.append({
            "tipo": tipo, "texto": texto, "t_video": self.t_video,
            "jugador": evento.get("jugador"), "equipo": evento.get("equipo"),
            "confianza": evento.get("confianza"), "motivo": evento.get("motivo"),
        })

        # Tarjeta de ultimo evento
        col = COLOR_EVENTO.get(tipo, "#FBBF24")
        titulo = TITULO_EVENTO.get(tipo, "EVENTO")
        if evento.get("confianza"):
            titulo += f"  (confianza {evento['confianza']})"
        destacado = tipo in ("GOL", "CELEBRACION")
        bg = "#78350F" if tipo == "GOL" else ("#831843" if tipo == "CELEBRACION" else BG_CARD)
        for w in (self.card_evento, self.lbl_evento_tit, self.lbl_evento_txt):
            w.config(bg=bg)
        self.lbl_evento_tit.config(text=titulo, fg="white" if destacado else col)
        self.lbl_evento_txt.config(text=texto)
        self.card_evento.config(highlightbackground=col)
        self.voz.decir(texto, PRIORIDAD_VOZ.get(tipo, EVENTO))
        if destacado:
            self._t_destacado = time.time()

        # Historial (lo mas reciente arriba)
        h = self.txt_hist
        h.config(state="normal")
        h.insert("1.0", "\n")
        if evento.get("motivo"):
            h.insert("1.0", f"   {evento['motivo']}\n", "motivo")
        h.insert("1.0", texto + "\n")
        h.insert("1.0", TITULO_EVENTO.get(tipo, tipo) + "  ", tipo)
        h.insert("1.0", f"[{mmss(self.t_video)}] ", "tiempo")
        h.config(state="disabled")

        if tipo == "GOL":
            self.lbl_marcador.config(text=self.narrador_futbol.marcador_texto())

    def _limpiar_panel_eventos(self):
        self.txt_hist.config(state="normal")
        self.txt_hist.delete("1.0", tk.END)
        self.txt_hist.config(state="disabled")
        for w in (self.card_evento, self.lbl_evento_tit, self.lbl_evento_txt):
            w.config(bg=BG_CARD)
        self.lbl_evento_tit.config(text="--", fg="#FBBF24")
        self.card_evento.config(highlightbackground=BG_CARD)
        self.lbl_evento_txt.config(text="")
        self.lbl_balon.config(text="--")
        self.lbl_marcador.config(text=self.narrador_futbol.marcador_texto())

    # =========================================================
    # BUCLE PRINCIPAL
    # =========================================================

    def update_video(self):
        if not self.running or not self.cap or self.paused:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.running = False
            self.lbl_status.config(text="Video terminado")
            return
        self.t_video = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        frame = cv2.resize(frame, (self.proc_w, self.proc_h))
        original = frame.copy()

        estado = self.motor.procesar_frame(frame)
        self.estado = estado
        self.reconocedor_dorsales.actualizar_estado(original, estado)

        # Eventos de juego (posesion / pases) y momentos clave (gol / celebracion)
        evento = self.detector_eventos.procesar(estado)
        if evento is not None:
            self.registrar_evento(evento)
        for ev in self.momentos.procesar(original, estado, self.t_video,
                                         self.motor.campo.mask_cesped):
            self.registrar_evento(ev)
        self.actualizar_narracion_balon(estado)

        # Dibujo (solo marcadores)
        if self.ver_campo.get():
            self.dibujar_depuracion(frame)
        self.dibujar_jugadores(frame, estado["tracks"])
        self.dibujar_balon(frame, estado["balon"])

        now = time.time()
        self.fps = 1.0 / max(now - self.last_time, 0.001)
        self.last_time = now

        self._update_panel(estado)
        self._update_panel_eventos(estado)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()
        if cw < 2 or ch < 2:
            self.root.after(15, self.update_video)
            return
        fw, fh = img.size
        esc = min(cw / fw, ch / fh)
        nw, nh = max(int(fw * esc), 1), max(int(fh * esc), 1)
        img = img.resize((nw, nh))
        fondo = Image.new("RGB", (cw, ch), (0, 0, 0))
        fondo.paste(img, ((cw - nw) // 2, (ch - nh) // 2))
        self.tkimg = ImageTk.PhotoImage(fondo)
        self.video_canvas.delete("all")
        self.video_canvas.create_image(0, 0, image=self.tkimg, anchor=tk.NW)
        self.root.after(15, self.update_video)

    def _update_panel_eventos(self, e):
        self.lbl_tiempo.config(text=mmss(self.t_video))
        plano = NOMBRES_PLANO.get(e.get("plano"), "--")
        juego = "juego activo" if e.get("juego_activo") else "juego no visible"
        self.lbl_plano.config(
            text=f"Plano: {plano}  |  {juego}  |  descartados: {e.get('descartados', 0)}")

        c = self.cv_celeb
        c.delete("all")
        w = c.winfo_width() or 200
        v = self.momentos.puntaje_celeb
        col = "#F472B6" if self.momentos.celebrando else "#6B7280"
        c.create_rectangle(0, 0, int(w * min(1.0, v)), 18, fill=col, outline="")
        c.create_line(int(w * 0.55), 0, int(w * 0.55), 18, fill="#F9FAFB", width=2)

        activas = [k for k, val in self.momentos.senales.items() if val > 0]
        self.lbl_senales.config(
            text=("Senales: " + (", ".join(activas) if activas else "ninguna")
                  + "\n" + self.momentos.descripcion_senales()))
        self.lbl_voz.config(text=self.voz.estado_texto())

        # El color destacado del GOL / CELEBRACION se apaga a los 4 s
        if self._t_destacado and time.time() - self._t_destacado > 4.0:
            self._t_destacado = 0
            for wdg in (self.card_evento, self.lbl_evento_tit, self.lbl_evento_txt):
                wdg.config(bg=BG_CARD)
            tipo = (self.ultimo_evento or {}).get("tipo", "")
            self.lbl_evento_tit.config(fg=COLOR_EVENTO.get(tipo, "#FBBF24"))

    def _draw_radar(self, tracks, balon):
        c = self.radar_c
        c.delete("all")
        rw = c.winfo_width() or 310
        rh = c.winfo_height() or 90
        c.create_line(rw // 2, 0, rw // 2, rh, fill="#166534", width=2)
        for tr in tracks:
            rx = int((tr.cx / self.proc_w) * rw)
            ry = int((tr.cy / self.proc_h) * rh)
            col = ("#EF4444" if tr.equipo == EQUIPO_A
                   else "#3B82F6" if tr.equipo == EQUIPO_B else "#9CA3AF")
            c.create_oval(rx - 4, ry - 4, rx + 4, ry + 4, fill=col, outline="")
        if balon:
            rx = int((balon[0] / self.proc_w) * rw)
            ry = int((balon[1] / self.proc_h) * rh)
            c.create_oval(rx - 5, ry - 5, rx + 5, ry + 5, fill="#FDE047", outline="#FBBF24",
                          width=2)

    def _draw_bar(self, pa, pb):
        c = self.bar_c
        c.delete("all")
        w = c.winfo_width() or 310
        h = 26
        xa = int(w * pa / 100)
        c.create_rectangle(0, 0, xa, h, fill="#DC2626", outline="")
        c.create_rectangle(xa, 0, w, h, fill=ACENTO, outline="")
        if xa > 28:
            c.create_text(xa // 2, h // 2, text=f"{pa}%", fill="white",
                          font=("Segoe UI", 8, "bold"))
        if w - xa > 28:
            c.create_text(xa + (w - xa) // 2, h // 2, text=f"{pb}%", fill="white",
                          font=("Segoe UI", 8, "bold"))

    def _update_panel(self, e: dict):
        self.lbl_status.config(
            text=("Calibrando colores..." if e.get("warmup") else "Sistema activo")
            + f"   FPS {self.fps:.1f}")
        if self.panel_activo != "tactico":
            return
        self.lbl_fps.config(text=f"FPS: {self.fps:.1f}")
        pa, pb = e["pos_a"], e["pos_b"]
        self._draw_bar(pa, pb)
        self.lbl_pos.config(
            text=f"{self._nombre(EQUIPO_A)} {pa}%   vs   {self._nombre(EQUIPO_B)} {pb}%")

        def s(key, va, vb):
            self._rows[key][0].config(text=str(va))
            self._rows[key][1].config(text=str(vb))

        s("count", e["count_a"], e["count_b"])
        s("pases", e["pases_a"], e["pases_b"])
        s("precision", f"{e['precision_a']}%", f"{e['precision_b']}%")
        s("perdidas", e["perdidas_a"], e["perdidas_b"])
        s("vel_max", f"{e['vel_max_a']:.1f}", f"{e['vel_max_b']:.1f}")
        s("vel_prom", f"{e['vel_prom_a']:.1f}", f"{e['vel_prom_b']:.1f}")
        s("dist", f"{e['dist_a']:.0f}", f"{e['dist_b']:.0f}")

        eq = e.get("eq_posesion", 0)
        quien = self._nombre(eq) if eq in (EQUIPO_A, EQUIPO_B) else "-- libre"
        self.lbl_quien.config(text=f"En posesion: {quien}")
        self.lbl_cambios.config(text=f"Cambios de posesion: {e['cambios_posesion']}")
        self.lbl_toques.config(text=f"Toques totales: {e['toques']}")
        self.lbl_zona.config(text=e["zona"])
        self._draw_radar(e["tracks"], e["balon"])


if __name__ == "__main__":
    root = tk.Tk()
    app = DetectorFutbol(root)
    root.mainloop()
