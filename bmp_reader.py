"""
bmp_reader.py
=============
Lectura manual de archivos BMP de 24bpp.
NO usa PIL, imageio, ni ninguna libreria de imagen externa.
El parseo es byte a byte, identico a la logica C del proyecto.

Retorna el buffer de pixeles como bytearray en formato RGB top-left,
igual que g_state.pixels_rgb en la version C.
"""

import struct
import os


# Tamanios de cabeceras BMP
_BITMAPFILEHEADER_SIZE = 14   # type(2) + size(4) + res1(2) + res2(2) + offset(4)
_BITMAPINFOHEADER_SIZE = 40


def leer_bmp(path: str):
    """
    Parsea un BMP de 24bpp.

    Retorna:
        (pixels_rgb: bytearray, width: int, height: int)
        pixels_rgb tiene largo width*height*3, en orden RGB, origen top-left.

    Lanza:
        ValueError si el archivo no es un BMP valido o no es 24bpp.
        FileNotFoundError si el archivo no existe.
    """
    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) < _BITMAPFILEHEADER_SIZE + _BITMAPINFOHEADER_SIZE:
        raise ValueError("Archivo demasiado pequeno para ser un BMP valido.")

    # --- BITMAPFILEHEADER ---
    # type(2s) size(I) res1(H) res2(H) offset(I)
    bfType, bfSize, _, _, bfOffBits = struct.unpack_from("<2sIHHI", raw, 0)

    if bfType != b"BM":
        raise ValueError(f"No es un BMP valido (magic bytes: {bfType!r})")

    # --- BITMAPINFOHEADER ---
    # size(I) width(i) height(i) planes(H) bpp(H) compression(I) ...
    (biSize, biWidth, biHeight,
     biPlanes, biBitCount,
     biCompression) = struct.unpack_from("<IiiHHI", raw, _BITMAPFILEHEADER_SIZE)

    if biBitCount != 24:
        raise ValueError(
            f"Solo se soportan BMP de 24bpp. Este archivo tiene {biBitCount}bpp."
        )

    width  = biWidth
    height = abs(biHeight)
    # biHeight negativo = top-down; positivo = bottom-up (lo mas comun)
    top_down = biHeight < 0

    # Stride: cada fila esta alineada a multiplos de 4 bytes
    stride = ((width * 3 + 3) // 4) * 4
    padding = stride - width * 3

    # Verificar que hay suficientes datos
    expected_data = stride * height
    if len(raw) < bfOffBits + expected_data:
        raise ValueError("El archivo BMP esta truncado o corrupto.")

    # --- Leer pixeles ---
    # BMP guarda en BGR. Nosotros queremos RGB, origen top-left.
    pixels_rgb = bytearray(width * height * 3)
    offset = bfOffBits

    for y_file in range(height):
        # Convertir y_file (orden de lectura del archivo) a y_real (top-left)
        if top_down:
            y_real = y_file
        else:
            y_real = height - 1 - y_file  # BMP bottom-up: invertir Y

        row_offset = offset + y_file * stride
        dst_row    = y_real * width * 3

        for x in range(width):
            b = raw[row_offset + x * 3 + 0]
            g = raw[row_offset + x * 3 + 1]
            r = raw[row_offset + x * 3 + 2]
            pixels_rgb[dst_row + x * 3 + 0] = r
            pixels_rgb[dst_row + x * 3 + 1] = g
            pixels_rgb[dst_row + x * 3 + 2] = b

    return pixels_rgb, width, height


def generar_bmp_prueba(path: str, width: int = 640, height: int = 480):
    """
    Genera una imagen BMP de prueba con 8 jugadores (4 rojos, 4 azules)
    y un balon blanco sobre fondo verde, identica a la de la version C.
    """
    stride   = ((width * 3 + 3) // 4) * 4
    padding  = stride - width * 3
    img_size = stride * height

    # Cabeceras
    file_header_size = _BITMAPFILEHEADER_SIZE
    info_header_size = _BITMAPINFOHEADER_SIZE
    offset = file_header_size + info_header_size
    total  = offset + img_size

    # Construir buffer de imagen (fondo verde, bottom-up)
    img = bytearray(img_size)
    for y in range(height):
        row = y * stride
        for x in range(width):
            img[row + x * 3 + 0] = 34   # B
            img[row + x * 3 + 1] = 139  # G
            img[row + x * 3 + 2] = 34   # R

    # Objetos a dibujar: (x, y, w, h, r, g, b) en coordenadas top-left
    objetos = [
        # Equipo A - rojo
        ( 80, 300, 15, 20, 210,  30,  30),
        (160, 200, 15, 20, 210,  30,  30),
        (160, 380, 15, 20, 210,  30,  30),
        (250, 150, 15, 20, 210,  30,  30),
        # Equipo B - azul
        (380, 300, 15, 20,  20,  30, 210),
        (460, 200, 15, 20,  20,  30, 210),
        (460, 380, 15, 20,  20,  30, 210),
        (550, 150, 15, 20,  20,  30, 210),
        # Balon - blanco
        (310, 240,  8,  8, 220, 220, 220),
    ]

    for ox, oy, ow, oh, cr, cg, cb in objetos:
        for dy in range(oh):
            for dx in range(ow):
                px = ox + dx
                py = oy + dy
                if px >= width or py >= height:
                    continue
                # Convertir a coordenada bottom-up para el archivo BMP
                yr  = height - 1 - py
                idx = yr * stride + px * 3
                img[idx + 0] = cb
                img[idx + 1] = cg
                img[idx + 2] = cr

    with open(path, "wb") as f:
        # BITMAPFILEHEADER
        f.write(struct.pack("<2sIHHI", b"BM", total, 0, 0, offset))
        # BITMAPINFOHEADER
        f.write(struct.pack(
            "<IiiHHIIiiII",
            info_header_size, width, height,  # height positivo = bottom-up
            1, 24, 0, img_size, 2835, 2835, 0, 0
        ))
        f.write(img)

    print(f"[OK] Imagen de prueba generada: {path} ({width}x{height})")
