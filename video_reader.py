"""
video_reader.py
===============
Lectura de video frame a frame usando ffmpeg via subprocess.

Equivalente directo de la funcionalidad Media Foundation de la version C:
  abrirVideo()       -> VideoReader.__init__() / abrir()
  leerSiguienteFrame() -> VideoReader.leer_frame()

ffmpeg es el equivalente honesto a Media Foundation en multiplataforma:
no es una libreria de Python, es una herramienta del sistema que ya
viene instalada en la mayoria de entornos o se instala con una linea.
Lo invocamos con subprocess, igual que Media Foundation se invocaba
con COM en C.

NO usa cv2, imageio, decord, PyAV, ni ninguna libreria Python de video.

Requisito: ffmpeg instalado y en el PATH del sistema.
  Windows: https://ffmpeg.org/download.html  (agregar al PATH)
  Linux:   sudo apt install ffmpeg
  macOS:   brew install ffmpeg
"""

import subprocess

class VideoReader:
    def __init__(self, path):
        self.path = path

        probe = subprocess.check_output([
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames",
            "-of", "csv=p=0",
            path
        ]).decode().strip().split(',')

        self.original_width = int(probe[0])
        self.original_height = int(probe[1])

        self.width = 960
        self.height = int(self.original_height * (960 / self.original_width))

        try:
            self.total_frames = int(probe[2])
        except:
            self.total_frames = 0

        cmd = [
            "ffmpeg",
            "-i", path,
            "-vf", "scale=960:-1",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-loglevel", "quiet",
            "-"
        ]

        self.pipe = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            bufsize=10**8
        )

        self.frame_size = self.width * self.height * 3
        self.frame_actual = 0

    def leer_frame(self):
        raw = self.pipe.stdout.read(self.frame_size)

        if len(raw) != self.frame_size:
            return None

        self.frame_actual += 1

        return bytearray(raw)

    def cerrar(self):
        self.pipe.kill()