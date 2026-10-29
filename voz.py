"""
voz.py
======
Voz del narrador (texto a voz), multiplataforma.

Motores, en orden de preferencia (se usa el primero que funcione):
    1. Piper   -> voz neuronal local, sin internet (pip install piper-tts).
                  El modelo va en la carpeta voces/ (p. ej. es_MX-ald-medium.onnx
                  + .onnx.json). Reproduce con winsound (Windows), afplay (macOS)
                  o paplay / pw-play / aplay (Linux).
    2. SAPI    -> sintetizador de Windows (System.Speech) en un PowerShell que
                  se queda abierto y recibe las frases por stdin.
    3. espeak  -> espeak-ng / espeak en Linux (sudo apt install espeak-ng).
Si ninguno esta disponible la app sigue funcionando sin voz.

Como los eventos llegan mas rapido de lo que se pueden decir, no hay cola
larga: solo se guarda UNA frase pendiente (la mas reciente). Prioridades:
    0 RELLENO  -> estado del balon cuando la voz lleva rato callada
    1 EVENTO   -> pases, cambios de posesion...
    2 URGENTE  -> GOL / CELEBRACION: corta lo que se este diciendo
Una frase pendiente solo se reemplaza por otra de prioridad igual o mayor.

Descargar la voz de Piper (una sola vez, necesita internet):
    python -m piper.download_voices --download-dir voces es_MX-ald-medium
"""

import base64
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave

RELLENO, EVENTO, URGENTE = 0, 1, 2

DIR_VOCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voces")
VOZ_PIPER = "es_MX-ald-medium"      # si no esta, se usa cualquier .onnx de voces/

# Piper: length_scale (<1 = mas rapido) y volumen (0..1) por prioridad
PIPER_RITMO = {RELLENO: 1.0, EVENTO: 0.92, URGENTE: 0.82}
PIPER_VOLUMEN = {RELLENO: 0.8, EVENTO: 1.0, URGENTE: 1.0}

# SAPI: velocidad (-10..10) y volumen (0..100) por prioridad
VELOCIDAD = {RELLENO: 1, EVENTO: 2, URGENTE: 4}
VOLUMEN = {RELLENO: 80, EVENTO: 100, URGENTE: 100}

# espeak: palabras por minuto y amplitud (0..200) por prioridad
ESPEAK_PPM = {RELLENO: 160, EVENTO: 175, URGENTE: 200}
ESPEAK_AMP = {RELLENO: 80, EVENTO: 100, URGENTE: 120}

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


# ---------------- motores ----------------
# Todos tienen: nombre, hablar(texto, prioridad) (bloquea hasta terminar o
# ser cortado), cortar() (desde otro hilo) y cerrar().

class _MotorPiper:

    def __init__(self):
        from piper import PiperVoice  # ImportError -> se prueba el siguiente motor
        modelo = os.path.join(DIR_VOCES, VOZ_PIPER + ".onnx")
        if not os.path.exists(modelo):
            modelos = sorted(glob.glob(os.path.join(DIR_VOCES, "*.onnx")))
            if not modelos:
                raise RuntimeError(f"no hay modelos .onnx en {DIR_VOCES}")
            modelo = modelos[0]
        self._reproductor = None
        if sys.platform not in ("win32", "darwin"):
            for cmd in (["paplay"], ["pw-play"], ["aplay", "-q"]):
                if shutil.which(cmd[0]):
                    self._reproductor = cmd
                    break
            else:
                raise RuntimeError("no hay reproductor de audio (paplay / pw-play / aplay)")
        self._voz = PiperVoice.load(modelo)
        self.nombre = "Piper " + os.path.basename(modelo)[:-5]
        self._wav = os.path.join(tempfile.gettempdir(), f"tln_voz_{os.getpid()}.wav")
        self._cortado = threading.Event()
        self._proc = None

    def hablar(self, texto, prioridad):
        from piper import SynthesisConfig
        self._cortado.clear()
        cfg = SynthesisConfig(length_scale=PIPER_RITMO[prioridad],
                              volume=PIPER_VOLUMEN[prioridad])
        with wave.open(self._wav, "wb") as w:
            self._voz.synthesize_wav(texto, w, cfg)
        if self._cortado.is_set():
            return
        if sys.platform == "win32":
            import winsound
            with wave.open(self._wav, "rb") as w:
                duracion = w.getnframes() / w.getframerate()
            winsound.PlaySound(self._wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
            if self._cortado.wait(duracion + 0.1):
                winsound.PlaySound(None, 0)
        else:
            cmd = ["afplay"] if sys.platform == "darwin" else self._reproductor
            self._proc = subprocess.Popen(cmd + [self._wav], stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)
            if self._cortado.is_set():
                self._proc.kill()
            self._proc.wait()
            self._proc = None

    def cortar(self):
        self._cortado.set()
        p = self._proc
        if p is not None:
            p.kill()

    def cerrar(self):
        self.cortar()
        try:
            os.remove(self._wav)
        except OSError:
            pass


class _MotorSAPI:

    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("SAPI solo existe en Windows")
        self._proc = None
        self.nombre = "SAPI"
        self._arrancar()

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
        self.nombre = linea.split("|", 1)[1] if "|" in linea else "SAPI"

    def hablar(self, texto, prioridad):
        if self._proc is None or self._proc.poll() is not None:
            self._arrancar()
        b64 = base64.b64encode(texto.encode("utf-8")).decode("ascii")
        self._proc.stdin.write(f"{VELOCIDAD[prioridad]}|{VOLUMEN[prioridad]}|{b64}\n")
        self._proc.stdin.flush()
        if self._proc.stdout.readline().strip() != "OK":
            self._proc = None          # lo cortamos (o murio): se reinicia

    def cortar(self):
        # SAPI no se puede interrumpir desde stdin mientras habla: se mata el
        # proceso y se vuelve a crear para la siguiente frase.
        p = self._proc
        if p is not None:
            p.kill()

    def cerrar(self):
        p = self._proc
        if p is not None:
            try:
                p.stdin.write("SALIR\n")
                p.stdin.flush()
            except OSError:
                pass
            p.kill()


class _MotorEspeak:

    def __init__(self):
        self._exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if self._exe is None:
            raise RuntimeError("no esta instalado espeak-ng")
        self.nombre = os.path.basename(self._exe) + " (es)"
        self._proc = None

    def hablar(self, texto, prioridad):
        self._proc = subprocess.Popen(
            [self._exe, "-v", "es", "-s", str(ESPEAK_PPM[prioridad]),
             "-a", str(ESPEAK_AMP[prioridad]), "--", texto],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._proc.wait()
        self._proc = None

    def cortar(self):
        p = self._proc
        if p is not None:
            p.kill()

    def cerrar(self):
        self.cortar()


_MOTORES = (_MotorPiper, _MotorSAPI, _MotorEspeak)


class VozNarrador:

    def __init__(self, activa=True):
        self.activa = activa
        self.disponible = True          # pasa a False si ningun motor arranca
        self.nombre_voz = None
        self.error = None
        self.motivos = []               # por que fallo cada motor descartado
        self._motor = None
        self._cond = threading.Condition()
        self._pendiente = None          # (prioridad, texto)
        self._hablando = None           # prioridad de lo que suena ahora
        self._cerrar = False
        self._t_fin = time.time()
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
            return "voz: no disponible (instala piper-tts o espeak-ng)"
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
        m = self._motor
        if m is not None:
            m.cerrar()

    # ---------------- internos ----------------

    def _cortar(self):
        m = self._motor
        if m is not None:
            m.cortar()

    def _elegir_motor(self):
        for clase in _MOTORES:
            try:
                self._motor = clase()
                self.nombre_voz = self._motor.nombre
                return
            except Exception as e:      # falta paquete, modelo o programa
                self.motivos.append(f"{clase.__name__[6:]}: {e}")
        self.disponible = False
        with self._cond:
            self._pendiente = None

    def _trabajar(self):
        self._elegir_motor()
        if self._motor is None:
            return
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
                self._motor.hablar(texto, prioridad)
                self.error = None
            except Exception as e:
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
    time.sleep(4)
    v.decir("Gol de Espana! Marcador 1-0.", URGENTE)
    time.sleep(5)
    print(v.estado_texto())
    for m in v.motivos:
        print("  descartado ->", m)
    v.cerrar()
