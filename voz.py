"""
voz.py
======
Voz del narrador (texto a voz).

Backend: el sintetizador SAPI de Windows (System.Speech) dentro de un
proceso de PowerShell que se queda abierto y recibe las frases por stdin.
No hay que instalar nada; si existe una voz en espanol (p. ej. "Microsoft
Sabina", es-MX) se elige automaticamente.

Como los eventos llegan mas rapido de lo que se pueden decir, no hay cola
larga: solo se guarda UNA frase pendiente (la mas reciente). Prioridades:
    0 RELLENO  -> estado del balon cuando la voz lleva rato callada
    1 EVENTO   -> pases, cambios de posesion...
    2 URGENTE  -> GOL / CELEBRACION: corta lo que se este diciendo
Una frase pendiente solo se reemplaza por otra de prioridad igual o mayor.
"""

import base64
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

RELLENO, EVENTO, URGENTE = 0, 1, 2

# Velocidad (-10..10) y volumen (0..100) de SAPI por prioridad
VELOCIDAD = {RELLENO: 1, EVENTO: 2, URGENTE: 4}
VOLUMEN = {RELLENO: 80, EVENTO: 100, URGENTE: 100}

_SCRIPT_PS = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$v = $s.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'es*' } | Select-Object -First 1
if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }
$s.SetOutputToDefaultAudioDevice()
[Console]::Out.WriteLine('LISTO|' + $s.Voice.Name)
while ($true) {
    $l = [Console]::In.ReadLine()
    if ($l -eq $null -or $l -eq 'SALIR') { break }
    $p = $l.Split('|')
    $s.Rate = [int]$p[0]
    $s.Volume = [int]$p[1]
    $s.Speak([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($p[2])))
    [Console]::Out.WriteLine('OK')
}
"""

# El narrador escribe sin tildes; la voz las necesita para pronunciar bien
_TILDES = {
    "espana": "España", "espanola": "española", "espanol": "español",
    "balon": "balón", "posesion": "posesión", "presion": "presión",
    "celebracion": "celebración", "transicion": "transición", "accion": "acción",
    "direccion": "dirección", "linea": "línea", "area": "área", "rapido": "rápido",
    "rapida": "rápida", "ultimo": "último", "tactica": "táctica", "tactico": "táctico",
    "cubarsi": "Cubarsí", "martinez": "Martínez", "fernandez": "Fernández",
    "gonzalez": "González", "alvarez": "Álvarez", "simon": "Simón", "fabian": "Fabián",
    "alex": "Álex", "nicolas": "Nicolás", "julian": "Julián", "despues": "después",
    "tambien": "también", "jugadon": "jugadón",
}


def limpiar_para_voz(texto):
    """Adapta el texto del panel para que suene natural al leerlo en voz alta."""
    t = texto
    t = re.sub(r"\s*\(#(\d+)\)", r", dorsal \1,", t)
    t = re.sub(r"#(\d+)", r"número \1", t)
    t = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1 a \2", t)
    t = t.replace("km/h", " kilómetros por hora")
    t = re.sub(r"\s*[()\[\]]\s*", ", ", t)
    t = re.sub(r"\s*,\s*([,.;:!?])", r"\1", t)
    t = re.sub(r",\s*,", ",", t)

    def tilde(m):
        w = m.group(0)
        nueva = _TILDES.get(w.lower())
        if nueva is None:
            return w
        return nueva[0].upper() + nueva[1:] if w[0].isupper() else nueva
    t = re.sub(r"[A-Za-z]+", tilde, t)
    return re.sub(r"\s+", " ", t).strip(" ,")


class VozNarrador:

    def __init__(self, activa=True):
        self.activa = activa
        self.disponible = sys.platform == "win32"
        self.nombre_voz = None
        self.error = None
        self._cond = threading.Condition()
        self._pendiente = None          # (prioridad, texto)
        self._hablando = None           # prioridad de lo que suena ahora
        self._proc = None
        self._cerrar = False
        self._t_fin = time.time()
        if self.disponible:
            threading.Thread(target=self._trabajar, daemon=True).start()

    # ---------------- API ----------------

    def decir(self, texto, prioridad=EVENTO):
        if not (self.activa and self.disponible) or not texto:
            return
        texto = limpiar_para_voz(texto)
        with self._cond:
            if self._pendiente and self._pendiente[0] > prioridad:
                return
            if self._hablando is not None and prioridad == URGENTE and self._hablando < URGENTE:
                self._pendiente = None
                self._cortar()
            self._pendiente = (prioridad, texto)
            self._cond.notify()

    def ocupada(self):
        return self._hablando is not None or self._pendiente is not None

    def segundos_callada(self):
        return 0.0 if self.ocupada() else time.time() - self._t_fin

    def callar(self):
        with self._cond:
            self._pendiente = None
            if self._hablando is not None:
                self._cortar()

    def set_activa(self, activa):
        self.activa = activa
        if not activa:
            self.callar()

    def estado_texto(self):
        if not self.disponible:
            return "voz: no disponible (solo Windows)"
        if self.error:
            return f"voz: error ({self.error})"
        if not self.activa:
            return "voz: silenciada"
        nombre = self.nombre_voz or "iniciando..."
        return f"voz: {nombre}" + ("  - hablando" if self._hablando is not None else "")

    def cerrar(self):
        with self._cond:
            self._cerrar = True
            self._pendiente = None
            self._cond.notify()
        p = self._proc
        if p is not None:
            try:
                p.stdin.write("SALIR\n")
                p.stdin.flush()
            except OSError:
                pass
            p.kill()

    # ---------------- internos ----------------

    def _cortar(self):
        # SAPI no se puede interrumpir desde stdin mientras habla: se mata el
        # proceso y el hilo lo vuelve a crear para la siguiente frase.
        p = self._proc
        if p is not None:
            p.kill()

    def _arrancar(self):
        # Se usa -File y no -Command / -EncodedCommand: con -Command el propio
        # PowerShell lee el stdin redirigido y se "come" las frases.
        ruta = os.path.join(tempfile.gettempdir(), "tln_voz_narrador.ps1")
        with open(ruta, "w", encoding="utf-8-sig") as f:
            f.write(_SCRIPT_PS)
        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", ruta],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        linea = self._proc.stdout.readline().strip()
        if not linea.startswith("LISTO"):
            raise RuntimeError(f"no arranco el sintetizador: {linea[:30]!r}")
        self.nombre_voz = linea.split("|", 1)[1] if "|" in linea else "SAPI"

    def _trabajar(self):
        while True:
            with self._cond:
                while self._pendiente is None and not self._cerrar:
                    self._cond.wait()
                if self._cerrar:
                    return
                prioridad, texto = self._pendiente
                self._pendiente = None
                self._hablando = prioridad
            try:
                if self._proc is None or self._proc.poll() is not None:
                    self._arrancar()
                b64 = base64.b64encode(texto.encode("utf-8")).decode("ascii")
                self._proc.stdin.write(f"{VELOCIDAD[prioridad]}|{VOLUMEN[prioridad]}|{b64}\n")
                self._proc.stdin.flush()
                if self._proc.stdout.readline().strip() != "OK":
                    self._proc = None          # lo cortamos (o murio): se reinicia
                self.error = None
            except (OSError, RuntimeError, ValueError) as e:
                self._proc = None
                self.error = str(e)[:40]
                time.sleep(1.0)
            finally:
                with self._cond:
                    self._hablando = None
                    self._t_fin = time.time()


if __name__ == "__main__":
    v = VozNarrador()
    v.decir("Cambio de posesion en la medular: La Furia Espanola corta el avance "
            "de Argentina con el zaguero central Pau Cubarsi (#22).")
    time.sleep(1.5)
    v.decir("Gol de Espana! Marcador 1-0.", URGENTE)
    time.sleep(6)
    print(v.estado_texto())
    v.cerrar()
