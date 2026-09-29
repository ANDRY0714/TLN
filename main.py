"""
main.py  v4
===========
- Sin cuadros sobre jugadores (clasificacion interna sigue igual)
- Solo punto de color (rojo=Eq.A, azul=Eq.B) + etiqueta con velocidad
- BALON sin tilde (cv2 no soporta Unicode)
- Panel completo con estadisticas comparativas
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import textwrap
import time

from tracker_engine import MotorTracking, EQUIPO_A, EQUIPO_B

VIDEO_WIDTH  = 960
VIDEO_HEIGHT = 540


class DetectorFutbol:

    def __init__(self, root):
        self.root = root
        self.root.title("PRO - Sistema PRO - Analisis de Futbol con IA")
        self.root.geometry("1600x900")
        self.root.configure(bg="#0F172A")

        self.cap     = None
        self.running = False
        self.paused  = False
        self.motor   = None

        self.proc_w = VIDEO_WIDTH
        self.proc_h = VIDEO_HEIGHT

        self.last_time = time.time()
        self.fps    = 0
        self.estado = {}
        self.tkimg  = None
        self._ultimo_balon = None

        self.crear_ui()

    # =========================================================
    # UI
    # =========================================================

    def crear_ui(self):
        top = tk.Frame(self.root, bg="#111827", height=55)
        top.pack(fill=tk.X, side=tk.TOP)
        top.pack_propagate(False)

        btn_s = dict(bg="#2563EB", fg="white", relief=tk.FLAT,
                     padx=14, pady=8, font=("Segoe UI", 10, "bold"),
                     cursor="hand2", activebackground="#1D4ED8")

        tk.Button(top, text="Abrir Video",
                  command=self.abrir_video, **btn_s).pack(side=tk.LEFT, padx=8, pady=8)
        self.btn_pausa = tk.Button(top, text="Pausa",
                                   command=self.toggle_pausa, **btn_s)
        self.btn_pausa.pack(side=tk.LEFT, padx=4, pady=8)
        tk.Button(top, text="Reiniciar",
                  command=self.reiniciar_video, **btn_s).pack(side=tk.LEFT, padx=4, pady=8)

        self.lbl_status = tk.Label(top, text="", bg="#111827", fg="#FBBF24",
                                    font=("Segoe UI", 10, "bold"))
        self.lbl_status.pack(side=tk.LEFT, padx=16)

        main = tk.Frame(self.root, bg="#0F172A")
        main.pack(fill=tk.BOTH, expand=True)

        self.video_canvas = tk.Canvas(main, bg="black", highlightthickness=0)
        self.video_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        panel = tk.Frame(main, bg="#111827", width=340)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        self._build_panel(panel)

    # --------------------------------------------------------
    def _sep(self, p):
        tk.Frame(p, bg="#374151", height=1).pack(fill=tk.X, padx=8, pady=3)

    def _build_panel(self, p):
        tk.Label(p, text="ANALISIS TACTICO", bg="#111827", fg="white",
                 font=("Segoe UI", 14, "bold")).pack(pady=(12, 2))

        self._sep(p)

        self.lbl_fps = tk.Label(p, text="FPS: --", bg="#111827", fg="#00FFAA",
                                 font=("Consolas", 12, "bold"), anchor="w")
        self.lbl_fps.pack(fill=tk.X, padx=10, pady=2)

        self._sep(p)

        tk.Label(p, text="POSESION DEL BALON", bg="#111827", fg="#9CA3AF",
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(fill=tk.X, padx=10)

        self.bar_c = tk.Canvas(p, bg="#1F2937", height=26, highlightthickness=0)
        self.bar_c.pack(fill=tk.X, padx=10, pady=3)

        self.lbl_pos = tk.Label(p, text="Rojo 50%  vs  Azul 50%",
                                 bg="#111827", fg="white",
                                 font=("Segoe UI", 10, "bold"), anchor="center")
        self.lbl_pos.pack(fill=tk.X, padx=10)

        self._sep(p)

        # Leyenda de colores
        ley = tk.Frame(p, bg="#111827")
        ley.pack(fill=tk.X, padx=10, pady=2)
        tk.Canvas(ley, bg="#DC2626", width=14, height=14,
                  highlightthickness=0).pack(side=tk.LEFT)
        tk.Label(ley, text=" Equipo A  ", bg="#111827", fg="#FCA5A5",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Canvas(ley, bg="#2563EB", width=14, height=14,
                  highlightthickness=0).pack(side=tk.LEFT)
        tk.Label(ley, text=" Equipo B", bg="#111827", fg="#93C5FD",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)

        self._sep(p)

        tk.Label(p, text="ESTADISTICAS COMPARATIVAS", bg="#111827", fg="#9CA3AF",
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(fill=tk.X, padx=10)

        cab = tk.Frame(p, bg="#1F2937")
        cab.pack(fill=tk.X, padx=10, pady=(2, 0))
        tk.Label(cab, text="Metrica",  bg="#1F2937", fg="#6B7280",
                 font=("Segoe UI", 8, "bold"), width=14, anchor="w").grid(row=0, column=0)
        tk.Label(cab, text="Eq. ROJO", bg="#1F2937", fg="#F87171",
                 font=("Segoe UI", 8, "bold"), width=9,  anchor="center").grid(row=0, column=1)
        tk.Label(cab, text="Eq. AZUL", bg="#1F2937", fg="#60A5FA",
                 font=("Segoe UI", 8, "bold"), width=9,  anchor="center").grid(row=0, column=2)

        self._rows = {}
        metricas = [
            ("Jugadores",     "count"),
            ("Pases",         "pases"),
            ("Precision %",   "precision"),
            ("Perdidas",      "perdidas"),
            ("Vel.max km/h",  "vel_max"),
            ("Vel.prom km/h", "vel_prom"),
            ("Distancia m",   "dist"),
        ]
        bgs = ("#111827", "#161D2B")
        for i, (nom, key) in enumerate(metricas):
            bg = bgs[i % 2]
            f = tk.Frame(p, bg=bg)
            f.pack(fill=tk.X, padx=10)
            tk.Label(f, text=nom, bg=bg, fg="#D1D5DB",
                     font=("Segoe UI", 8), width=14, anchor="w").pack(side=tk.LEFT)
            va = tk.Label(f, text="--", bg=bg, fg="#FCA5A5",
                          font=("Consolas", 9, "bold"), width=9, anchor="center")
            va.pack(side=tk.LEFT)
            vb = tk.Label(f, text="--", bg=bg, fg="#93C5FD",
                          font=("Consolas", 9, "bold"), width=9, anchor="center")
            vb.pack(side=tk.LEFT)
            self._rows[key] = (va, vb)

        self._sep(p)

        tk.Label(p, text="EVENTOS DE JUEGO", bg="#111827", fg="#9CA3AF",
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(fill=tk.X, padx=10)

        self.lbl_quien   = tk.Label(p, text="En posesion: --",
                                     bg="#111827", fg="#FBBF24",
                                     font=("Segoe UI", 10, "bold"), anchor="w")
        self.lbl_quien.pack(fill=tk.X, padx=10, pady=1)

        self.lbl_cambios = tk.Label(p, text="Cambios: 0",
                                     bg="#111827", fg="white",
                                     font=("Segoe UI", 9), anchor="w")
        self.lbl_cambios.pack(fill=tk.X, padx=10, pady=1)

        self.lbl_toques  = tk.Label(p, text="Toques: 0",
                                     bg="#111827", fg="white",
                                     font=("Segoe UI", 9), anchor="w")
        self.lbl_toques.pack(fill=tk.X, padx=10, pady=1)

        self._sep(p)

        tk.Label(p, text="ZONA DE JUEGO", bg="#111827", fg="#9CA3AF",
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(fill=tk.X, padx=10)
        self.lbl_zona = tk.Label(p, text="--", bg="#111827", fg="#A78BFA",
                                  font=("Segoe UI", 12, "bold"), anchor="w")
        self.lbl_zona.pack(fill=tk.X, padx=10, pady=2)

        self._sep(p)

        tk.Label(p, text="RADAR", bg="#111827", fg="#9CA3AF",
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(fill=tk.X, padx=10)
        self.radar_c = tk.Canvas(p, bg="#064E3B", width=310, height=90,
                                  highlightthickness=1,
                                  highlightbackground="#374151")
        self.radar_c.pack(padx=10, pady=4)

    # =========================================================
    # CONTROLES
    # =========================================================

    def abrir_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        sw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        sh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if sw <= 0 or sh <= 0:
            sw, sh = VIDEO_WIDTH, VIDEO_HEIGHT
        esc = min(960/sw, 960/sh, 1.0)
        self.proc_w = max(int(sw*esc), 1)
        self.proc_h = max(int(sh*esc), 1)
        self.motor   = MotorTracking(self.proc_w, self.proc_h)
        self._ultimo_balon = None
        self.running = True
        self.paused  = False
        self.btn_pausa.config(text="Pausa")
        self.update_video()

    def toggle_pausa(self):
        if not self.cap:
            return
        self.paused = not self.paused
        self.btn_pausa.config(text="Reanudar" if self.paused else "Pausa")
        if not self.paused and self.running:
            self.update_video()

    def reiniciar_video(self):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.motor  = MotorTracking(self.proc_w, self.proc_h)
        self._ultimo_balon = None
        self.paused = False
        self.btn_pausa.config(text="Pausa")
        if not self.running:
            self.running = True
            self.update_video()

    # =========================================================
    # COLORES  BGR para cv2
    # =========================================================

    def _bgr(self, equipo):
        if equipo == EQUIPO_A: return (0, 0, 220)    # rojo
        if equipo == EQUIPO_B: return (220, 80, 0)   # azul
        return (160, 160, 160)                        # gris (sin clasificar)

    # =========================================================
    # DIBUJO — solo puntos, sin cuadros
    # =========================================================

    def dibujar_jugadores(self, frame, tracks):
        for tr in tracks:
            col = self._bgr(tr.equipo)
            cx, cy = int(tr.cx), int(tr.cy)

            # Punto con borde blanco para contraste sobre cualquier fondo
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy),  8, col, -1)

            vel = getattr(tr, "vel_actual_kmh", 0.0)
            label = f"J{tr.track_id}"
            if vel > 1.0:
                label += f" {vel:.1f}km/h"

            # Sombra negra + texto de color
            cv2.putText(frame, label, (cx + 12, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(frame, label, (cx + 12, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    def dibujar_balon(self, frame, balon):
        if balon is None:
            return
        bx, by = int(balon[0]), int(balon[1])
        cv2.circle(frame, (bx, by), 10, (255, 255, 255), 2)
        cv2.circle(frame, (bx, by),  7, (0, 220, 220), -1)
        cv2.putText(frame, "BALON", (bx + 13, by + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(frame, "BALON", (bx + 13, by + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)

    def _sanear_texto(self, texto):
        if texto is None:
            return ""
        texto = str(texto)
        texto = texto.replace("?", "").replace("¿", "").replace("¡", "").replace("!", "")
        texto = texto.replace("  ", " ").strip()
        return texto

    def narracion_balon(self, e):
        balon = e.get("balon")
        if balon is None:
            return self._sanear_texto("Balon sin seguimiento")

        eq = e.get("eq_posesion", 0)
        if eq == EQUIPO_A:
            equipo = "equipo rojo"
        elif eq == EQUIPO_B:
            equipo = "equipo azul"
        else:
            equipo = "juego abierto"

        zona = e.get("zona", "SIN ACCION").lower()
        if "ata" in zona:
            fase = "en ataque"
        elif "def" in zona:
            fase = "en defensa"
        elif "cent" in zona:
            fase = "en el centro"
        else:
            fase = "en juego"

        if self._ultimo_balon is not None:
            dx = balon[0] - self._ultimo_balon[0]
            dy = balon[1] - self._ultimo_balon[1]
            if abs(dx) > abs(dy):
                if dx > 12:
                    movimiento = "avanza hacia la derecha"
                elif dx < -12:
                    movimiento = "se mueve hacia la izquierda"
                else:
                    movimiento = "se mantiene estable"
            else:
                if dy > 12:
                    movimiento = "baja por el campo"
                elif dy < -12:
                    movimiento = "sube por el campo"
                else:
                    movimiento = "se mantiene estable"
        else:
            movimiento = "empieza a moverse"

        self._ultimo_balon = balon

        if eq in (EQUIPO_A, EQUIPO_B):
            texto = f"El balon esta con el {equipo}, {movimiento} {fase}."
        else:
            texto = f"El balon esta en juego abierto, {movimiento} {fase}."
        return self._sanear_texto(texto)

    def dibujar_panel_balon(self, frame, narracion):
        if not narracion:
            return

        narracion = self._sanear_texto(narracion)
        if not narracion:
            return

        x = max(10, self.proc_w - 330)
        y = 18
        ancho = 310
        alto = 110

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + ancho, y + alto), (18, 24, 35), -1)
        cv2.rectangle(overlay, (x, y), (x + ancho, y + alto), (0, 200, 255), 2)
        cv2.line(overlay, (x + 12, y + 28), (x + ancho - 12, y + 28), (0, 200, 255), 1)
        frame[:] = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        cv2.putText(frame, "BALON", (x + 18, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        lineas = textwrap.wrap(narracion, width=30)
        for i, linea in enumerate(lineas[:3]):
            y_texto = y + 52 + i * 20
            cv2.putText(frame, linea, (x + 18, y_texto),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def dibujar_grid(self, frame):
        h, w = frame.shape[:2]
        c = (50, 50, 50)
        for i in (1, 2):
            cv2.line(frame, (w*i//3, 0),  (w*i//3, h), c, 1)
            cv2.line(frame, (0, h*i//3),  (w, h*i//3), c, 1)

    # =========================================================
    # RADAR
    # =========================================================

    def _draw_radar(self, tracks, balon):
        c  = self.radar_c
        c.delete("all")
        rw = c.winfo_width()  or 310
        rh = c.winfo_height() or 90
        c.create_line(rw//2, 0, rw//2, rh, fill="#166534", width=2)
        for tr in tracks:
            rx = int((tr.cx / self.proc_w) * rw)
            ry = int((tr.cy / self.proc_h) * rh)
            col = ("#EF4444" if tr.equipo == EQUIPO_A else
                   "#3B82F6" if tr.equipo == EQUIPO_B else "#9CA3AF")
            c.create_oval(rx-4, ry-4, rx+4, ry+4, fill=col, outline="")
        if balon:
            rx = int((balon[0] / self.proc_w) * rw)
            ry = int((balon[1] / self.proc_h) * rh)
            c.create_oval(rx-5, ry-5, rx+5, ry+5,
                          fill="#FDE047", outline="#FBBF24", width=2)

    # =========================================================
    # BARRA POSESION
    # =========================================================

    def _draw_bar(self, pa, pb):
        c = self.bar_c
        c.delete("all")
        w = c.winfo_width() or 310
        h = 26
        xa = int(w * pa / 100)
        c.create_rectangle(0,  0, xa, h, fill="#DC2626", outline="")
        c.create_rectangle(xa, 0, w,  h, fill="#2563EB", outline="")
        if xa > 28:
            c.create_text(xa//2, h//2, text=f"{pa}%",
                          fill="white", font=("Segoe UI", 8, "bold"))
        if w - xa > 28:
            c.create_text(xa + (w-xa)//2, h//2, text=f"{pb}%",
                          fill="white", font=("Segoe UI", 8, "bold"))

    # =========================================================
    # LOOP
    # =========================================================

    def update_video(self):
        if not self.running or not self.cap or self.paused:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.running = False
            return

        frame = cv2.resize(frame, (self.proc_w, self.proc_h))

        estado = self.motor.procesar_frame(frame)
        self.estado = estado

        self.dibujar_grid(frame)
        self.dibujar_jugadores(frame, estado["tracks"])
        self.dibujar_balon(frame, estado["balon"])

        narracion = self.narracion_balon(estado)
        self.dibujar_panel_balon(frame, narracion)

        if estado.get("warmup"):
            cv2.putText(frame, "Calibrando colores...",
                        (10, self.proc_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        now = time.time()
        self.fps = 1.0 / max(now - self.last_time, 0.001)
        self.last_time = now
        cv2.putText(frame, f"FPS: {self.fps:.1f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        self._update_panel(estado)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        cw  = self.video_canvas.winfo_width()
        ch  = self.video_canvas.winfo_height()
        if cw < 2 or ch < 2:
            self.root.after(15, self.update_video)
            return
        fw, fh = img.size
        esc = min(cw/fw, ch/fh)
        nw, nh = max(int(fw*esc), 1), max(int(fh*esc), 1)
        img   = img.resize((nw, nh))
        fondo = Image.new("RGB", (cw, ch), (0, 0, 0))
        fondo.paste(img, ((cw-nw)//2, (ch-nh)//2))
        self.tkimg = ImageTk.PhotoImage(fondo)
        self.video_canvas.create_image(0, 0, image=self.tkimg, anchor=tk.NW)

        self.root.after(15, self.update_video)

    # =========================================================
    # PANEL
    # =========================================================

    def _update_panel(self, e: dict):
        self.lbl_status.config(
            text=("Calibrando colores..." if e.get("warmup") else "Sistema activo"))

        self.lbl_fps.config(text=f"FPS: {self.fps:.1f}")

        pa, pb = e["pos_a"], e["pos_b"]
        self._draw_bar(pa, pb)
        self.lbl_pos.config(text=f"Rojo {pa}%   vs   Azul {pb}%")

        def s(key, va, vb):
            self._rows[key][0].config(text=str(va))
            self._rows[key][1].config(text=str(vb))

        s("count",     e["count_a"],              e["count_b"])
        s("pases",     e["pases_a"],              e["pases_b"])
        s("precision", f"{e['precision_a']}%",   f"{e['precision_b']}%")
        s("perdidas",  e["perdidas_a"],            e["perdidas_b"])
        s("vel_max",   f"{e['vel_max_a']:.1f}",  f"{e['vel_max_b']:.1f}")
        s("vel_prom",  f"{e['vel_prom_a']:.1f}", f"{e['vel_prom_b']:.1f}")
        s("dist",      f"{e['dist_a']:.0f}",     f"{e['dist_b']:.0f}")

        eq = e.get("eq_posesion", 0)
        quien = ("ROJO Eq.A" if eq == EQUIPO_A else
                 "AZUL Eq.B"  if eq == EQUIPO_B else "-- libre")
        self.lbl_quien.config(text=f"En posesion: {quien}")
        self.lbl_cambios.config(text=f"Cambios de posesion: {e['cambios_posesion']}")
        self.lbl_toques.config(text=f"Toques totales: {e['toques']}")
        self.lbl_zona.config(text=e["zona"])
        self._draw_radar(e["tracks"], e["balon"])


# =========================================================
root = tk.Tk()
DetectorFutbol(root)
root.mainloop()

# === Extensiones PRO ===
def resumen_profesional(motor):
    try:
        topv = motor._top_velocidad()
        topd = motor._top_distancia()
        print("\n===== RESUMEN PROFESIONAL =====")
        print("TOP VELOCIDAD:", topv)
        print("TOP DISTANCIA:", topd)
    except Exception:
        pass
