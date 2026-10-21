"""
jersey_recognizer.py
====================
Modulo de vision por computadora para identificar los numeros de las camisetas
(dorsales) de los jugadores en el analisis de video y asignarlos como la
etiqueta principal ('track_id' / 'id') que utiliza todo el sistema
(tracker_engine.py, event_detector.py, narrator.py y main.py).

Soporta tanto objetos Track (en estado["tracks"] con atributos .track_id,
.equipo, .cx, .cy y opcionalmente .x, .y, .w, .h) como diccionarios de
jugadores (en estado["jugadores_a"] / estado["jugadores_b"]).
"""

from collections import Counter, defaultdict
import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from tracker_engine import EQUIPO_A, EQUIPO_B
except ImportError:
    EQUIPO_A = 1
    EQUIPO_B = 2


class ReconocedorDorsal:
    """
    Identifica el numero de camiseta (1-99) a partir del recorte BGR de cada
    jugador y estabiliza la etiqueta a lo largo del video para que el sistema
    use el dorsal real como identificador ('track_id' e 'id').
    """

    ANCHO_NORM = 28
    ALTO_NORM = 40

    HUECOS_POR_DIGITO = {
        0: {1},
        1: {0},
        2: {0},
        3: {0},
        4: {0, 1},
        5: {0},
        6: {1},
        7: {0},
        8: {2},
        9: {1},
    }

    def __init__(
        self,
        min_votos_confirmacion=2,
        umbral_confianza=0.52,
        intervalo_frames_lectura=2,
        dorsales_validos_por_equipo=None,
        usar_tesseract=False,
    ):
        self.min_votos_confirmacion = max(1, int(min_votos_confirmacion))
        self.umbral_confianza = float(umbral_confianza)
        self.intervalo_frames_lectura = max(1, int(intervalo_frames_lectura))
        self.usar_tesseract = bool(usar_tesseract)

        self.dorsales_validos = {
            EQUIPO_A: set(),
            EQUIPO_B: set(),
        }
        if isinstance(dorsales_validos_por_equipo, dict):
            for eq_id, lista in dorsales_validos_por_equipo.items():
                if eq_id in self.dorsales_validos and lista:
                    self.dorsales_validos[eq_id] = {
                        int(d) for d in lista if str(d).isdigit()
                    }

        self._votos_track = defaultdict(Counter)
        self._dorsal_confirmado = {}
        self._frames_track = defaultdict(int)

        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        self._plantillas_digitos = self._generar_plantillas_sinteticas()

    def _generar_plantillas_sinteticas(self):
        plantillas = defaultdict(list)
        fuentes = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
        ]
        grosores = [2, 3, 4]

        for digito in range(10):
            texto = str(digito)
            for fuente in fuentes:
                for grosor in grosores:
                    lienzo = np.zeros((80, 60), dtype=np.uint8)
                    escala = 2.2 if fuente != cv2.FONT_HERSHEY_PLAIN else 3.2
                    (tw, th), _ = cv2.getTextSize(texto, fuente, escala, grosor)
                    px = max(2, (60 - tw) // 2)
                    py = max(th + 2, (80 + th) // 2)
                    cv2.putText(lienzo, texto, (px, py), fuente, escala, 255, grosor, cv2.LINE_AA)

                    coords = cv2.findNonZero(lienzo)
                    if coords is None:
                        continue
                    x, y, w, h = cv2.boundingRect(coords)
                    recorte = lienzo[y:y + h, x:x + w]
                    norm = cv2.resize(
                        recorte,
                        (self.ANCHO_NORM, self.ALTO_NORM),
                        interpolation=cv2.INTER_AREA,
                    )
                    _, norm_bin = cv2.threshold(norm, 127, 255, cv2.THRESH_BINARY)
                    plantillas[digito].append(norm_bin)

        return plantillas

    @staticmethod
    def _obtener_caja_jugador(jugador):
        if isinstance(jugador, dict):
            if all(k in jugador for k in ("x", "y", "w", "h")):
                return int(jugador["x"]), int(jugador["y"]), int(jugador["w"]), int(jugador["h"])
            cx = int(jugador.get("cx", 0))
            cy = int(jugador.get("cy", 0))
            w = int(jugador.get("w", 48))
            h = int(jugador.get("h", 84))
            return cx - w // 2, cy - h // 2, w, h

        if all(hasattr(jugador, k) for k in ("x", "y", "w", "h")):
            w = int(getattr(jugador, "w", 0))
            h = int(getattr(jugador, "h", 0))
            if w > 0 and h > 0:
                return int(jugador.x), int(jugador.y), w, h

        cx = int(getattr(jugador, "cx", 0))
        cy = int(getattr(jugador, "cy", 0))
        w = int(getattr(jugador, "w", 48) or 48)
        h = int(getattr(jugador, "h", 84) or 84)
        return cx - w // 2, cy - h // 2, w, h

    def _extraer_roi_torso(self, frame, jugador):
        if frame is None or frame.size == 0:
            return None

        fh, fw = frame.shape[:2]
        x, y, w, h = self._obtener_caja_jugador(jugador)

        if w < 12 or h < 20:
            return None

        y1 = max(0, int(y + 0.12 * h))
        y2 = min(fh, int(y + 0.68 * h))
        x1 = max(0, int(x + 0.12 * w))
        x2 = min(fw, int(x + 0.88 * w))

        if y2 - y1 < 10 or x2 - x1 < 8:
            return None

        return frame[y1:y2, x1:x2]

    def _obtener_mascaras_candidatas(self, roi_bgr):
        gris = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        rh, rw = gris.shape[:2]
        if rh < 64:
            factor = 64.0 / float(rh)
            gris = cv2.resize(
                gris,
                (max(24, int(rw * factor)), 64),
                interpolation=cv2.INTER_CUBIC,
            )

        gris_eq = self._clahe.apply(gris)
        suave = cv2.GaussianBlur(gris_eq, (3, 3), 0)

        _, bin_claro = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, bin_oscuro = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return [bin_claro, bin_oscuro]

    def _segmentar_cajas_digitos(self, mascara_bin):
        mh, mw = mascara_bin.shape[:2]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara_bin, connectivity=8)

        candidatos = []
        for lbl in range(1, num_labels):
            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            w = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            cx, cy = centroids[lbl]

            if x <= 1 or y <= 1 or (x + w) >= (mw - 1) or (y + h) >= (mh - 1):
                continue

            rel_h = h / float(mh)
            rel_w = w / float(mw)
            aspect = w / float(max(1, h))
            lleno = area / float(max(1, w * h))

            if not (0.20 <= rel_h <= 0.90):
                continue
            if not (0.08 <= rel_w <= 0.75):
                continue
            if not (0.12 <= aspect <= 0.98):
                continue
            if not (0.15 <= lleno <= 0.98):
                continue
            if not (0.20 * mh <= cy <= 0.80 * mh):
                continue

            sub_mask = np.where(labels[y:y + h, x:x + w] == lbl, 255, 0).astype(np.uint8)
            candidatos.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": cx,
                "cy": cy,
                "mask": sub_mask,
            })

        if not candidatos:
            return []

        candidatos.sort(key=lambda c: abs(c["cy"] - mh / 2.0) - 0.3 * c["h"])
        principal = candidatos[0]
        seleccionados = [principal]

        for c in candidatos[1:]:
            dif_y = abs(c["cy"] - principal["cy"])
            dif_h = abs(c["h"] - principal["h"]) / float(max(principal["h"], c["h"]))
            dist_x = abs(c["cx"] - principal["cx"])
            if dif_y <= 0.25 * mh and dif_h <= 0.35 and (0.35 * principal["w"] <= dist_x <= 1.25 * max(principal["h"], c["h"])):
                seleccionados.append(c)
                break

        seleccionados.sort(key=lambda c: c["x"])
        return seleccionados

    @staticmethod
    def _contar_huecos_internos(mask_digito):
        pad = cv2.copyMakeBorder(mask_digito, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        contornos, jerarquia = cv2.findContours(pad, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if jerarquia is None:
            return 0

        huecos = 0
        area_total = float(mask_digito.shape[0] * mask_digito.shape[1])
        for idx, h_info in enumerate(jerarquia[0]):
            if h_info[3] != -1:
                area_hueco = cv2.contourArea(contornos[idx])
                if area_hueco >= 0.025 * area_total:
                    huecos += 1
        return min(huecos, 2)

    def _clasificar_digito_individual(self, item_digito):
        w, h = item_digito["w"], item_digito["h"]
        aspect = w / float(max(1, h))
        mask = item_digito["mask"]

        huecos = self._contar_huecos_internos(mask)
        if aspect < 0.45 and huecos == 0:
            densidad = np.count_nonzero(mask) / float(max(1, w * h))
            if densidad > 0.48:
                return 1, 0.85

        norm = cv2.resize(
            mask,
            (self.ANCHO_NORM, self.ALTO_NORM),
            interpolation=cv2.INTER_NEAREST,
        )

        mejor_digito = None
        mejor_puntaje = -1.0

        for digito, lista_plantillas in self._plantillas_digitos.items():
            penalizacion_huecos = 0.0 if huecos in self.HUECOS_POR_DIGITO[digito] else 0.18

            max_sim_digito = -1.0
            for tpl in lista_plantillas:
                res = cv2.matchTemplate(norm, tpl, cv2.TM_CCOEFF_NORMED)
                corr = float(res[0, 0])

                inter = np.logical_and(norm > 0, tpl > 0).sum()
                union = np.logical_or(norm > 0, tpl > 0).sum()
                iou = float(inter) / float(max(1, union))

                puntaje = 0.55 * max(0.0, corr) + 0.45 * iou - penalizacion_huecos
                if puntaje > max_sim_digito:
                    max_sim_digito = puntaje

            if max_sim_digito > mejor_puntaje:
                mejor_puntaje = max_sim_digito
                mejor_digito = digito

        if mejor_digito in (6, 9) and huecos == 1:
            mitad = self.ALTO_NORM // 2
            masa_sup = np.count_nonzero(norm[:mitad, :])
            masa_inf = np.count_nonzero(norm[mitad:, :])
            if masa_sup > masa_inf * 1.08:
                mejor_digito = 9
            elif masa_inf > masa_sup * 1.08:
                mejor_digito = 6

        return mejor_digito, float(max(0.0, min(1.0, mejor_puntaje)))

    def detectar_dorsal_en_roi(self, roi_bgr, equipo_id=None):
        if roi_bgr is None or roi_bgr.size == 0:
            return None, 0.0

        dorsales_permitidos = self.dorsales_validos.get(equipo_id) or set()

        if self.usar_tesseract and pytesseract is not None:
            try:
                gris = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                txt = pytesseract.image_to_string(
                    gris,
                    config="--psm 7 -c tessedit_char_whitelist=0123456789",
                ).strip()
                if txt.isdigit() and 1 <= int(txt) <= 99:
                    num_ocr = int(txt)
                    if not dorsales_permitidos or num_ocr in dorsales_permitidos:
                        return num_ocr, 0.90
            except Exception:
                pass

        mejor_numero = None
        mejor_confianza = 0.0

        for mascara in self._obtener_mascaras_candidatas(roi_bgr):
            cajas = self._segmentar_cajas_digitos(mascara)
            if not cajas:
                continue

            digitos_leidos = []
            confianzas = []
            for c in cajas:
                dig, conf = self._clasificar_digito_individual(c)
                if dig is None:
                    break
                digitos_leidos.append(str(dig))
                confianzas.append(conf)

            if not digitos_leidos or len(digitos_leidos) != len(cajas):
                continue

            cadena_num = "".join(digitos_leidos)
            if not cadena_num.isdigit():
                continue

            numero = int(cadena_num)
            if not (1 <= numero <= 99):
                continue

            conf_promedio = float(sum(confianzas) / len(confianzas))

            if dorsales_permitidos:
                if numero in dorsales_permitidos:
                    conf_promedio = min(1.0, conf_promedio + 0.15)
                else:
                    conf_promedio *= 0.75

            if conf_promedio > mejor_confianza:
                mejor_confianza = conf_promedio
                mejor_numero = numero

        if mejor_numero is not None and mejor_confianza >= self.umbral_confianza:
            return mejor_numero, mejor_confianza

        return None, 0.0

    def identificar_y_etiquetar_objeto_track(self, frame, tr, dorsales_ocupados=None):
        equipo_id = getattr(tr, "equipo", None)
        raw_id = getattr(tr, "_raw_track_id", getattr(tr, "track_id", None))
        setattr(tr, "_raw_track_id", raw_id)

        clave = (equipo_id, raw_id)
        self._frames_track[clave] += 1

        dorsal_actual = self._dorsal_confirmado.get(clave)
        debe_leer = (
            dorsal_actual is None
            or (self._frames_track[clave] % self.intervalo_frames_lectura == 0)
        )

        if debe_leer:
            roi_torso = self._extraer_roi_torso(frame, tr)
            dorsal_leido, conf = self.detectar_dorsal_en_roi(roi_torso, equipo_id=equipo_id)
            if dorsal_leido is not None:
                self._votos_track[clave][dorsal_leido] += conf
                candidato_top, puntaje_top = self._votos_track[clave].most_common(1)[0]
                if puntaje_top >= (self.min_votos_confirmacion * self.umbral_confianza):
                    if dorsales_ocupados is None or candidato_top not in dorsales_ocupados:
                        self._dorsal_confirmado[clave] = candidato_top
                        dorsal_actual = candidato_top

        if dorsal_actual is not None:
            setattr(tr, "dorsal", dorsal_actual)
            setattr(tr, "dorsal_confirmado", True)
            setattr(tr, "track_id", dorsal_actual)
            if dorsales_ocupados is not None:
                dorsales_ocupados.add(dorsal_actual)
        elif self._votos_track[clave]:
            candidato_prov, _ = self._votos_track[clave].most_common(1)[0]
            if dorsales_ocupados is None or candidato_prov not in dorsales_ocupados:
                setattr(tr, "dorsal", candidato_prov)
                setattr(tr, "dorsal_confirmado", False)
                setattr(tr, "track_id", candidato_prov)

        return tr

    def identificar_y_etiquetar_jugador(self, frame, jugador_dict, equipo_id, dorsales_ocupados=None):
        track_id = jugador_dict.get("track_id", jugador_dict.get("id"))
        jugador_dict["track_id"] = track_id
        clave = (equipo_id, track_id)
        self._frames_track[clave] += 1

        dorsal_actual = self._dorsal_confirmado.get(clave)
        debe_leer = (
            dorsal_actual is None
            or (self._frames_track[clave] % self.intervalo_frames_lectura == 0)
        )

        if debe_leer:
            roi_torso = self._extraer_roi_torso(frame, jugador_dict)
            dorsal_leido, conf = self.detectar_dorsal_en_roi(roi_torso, equipo_id=equipo_id)
            if dorsal_leido is not None:
                self._votos_track[clave][dorsal_leido] += conf
                candidato_top, puntaje_top = self._votos_track[clave].most_common(1)[0]
                if puntaje_top >= (self.min_votos_confirmacion * self.umbral_confianza):
                    if not dorsales_ocupados or candidato_top not in dorsales_ocupados:
                        self._dorsal_confirmado[clave] = candidato_top
                        dorsal_actual = candidato_top

        if dorsal_actual is not None:
            jugador_dict["dorsal"] = dorsal_actual
            jugador_dict["dorsal_confirmado"] = True
            jugador_dict["id"] = dorsal_actual
            if dorsales_ocupados is not None:
                dorsales_ocupados.add(dorsal_actual)
        elif self._votos_track[clave]:
            candidato_prov, _ = self._votos_track[clave].most_common(1)[0]
            if not dorsales_ocupados or candidato_prov not in dorsales_ocupados:
                jugador_dict["dorsal"] = candidato_prov
                jugador_dict["dorsal_confirmado"] = False
                jugador_dict["id"] = candidato_prov

        return jugador_dict

    def actualizar_estado(self, frame_original, estado, frame_dibujo=None):
        if not estado or frame_original is None:
            return estado

        tracks = estado.get("tracks", [])
        if tracks:
            ocupados_por_equipo = {EQUIPO_A: set(), EQUIPO_B: set()}
            for tr in tracks:
                eq_id = getattr(tr, "equipo", None)
                ocupados = ocupados_por_equipo.get(eq_id)
                self.identificar_y_etiquetar_objeto_track(
                    frame_original, tr, dorsales_ocupados=ocupados
                )

        for clave_lista, eq_id in (("jugadores_a", EQUIPO_A), ("jugadores_b", EQUIPO_B)):
            jugadores = estado.get(clave_lista, [])
            ocupados = set()
            for jug in jugadores:
                if isinstance(jug, dict):
                    self.identificar_y_etiquetar_jugador(
                        frame_original,
                        jug,
                        equipo_id=eq_id,
                        dorsales_ocupados=ocupados,
                    )

        return estado

    def sincronizar_con_narrador(self, narrador):
        if not hasattr(narrador, "red_semantica"):
            return
        plantilla = getattr(narrador.red_semantica, "plantilla_equipos", {})
        for eq_id in (EQUIPO_A, EQUIPO_B):
            dorsales = set()
            for info in plantilla.get(eq_id, {}).values():
                num = info.get("numero")
                if num is not None and str(num).isdigit():
                    dorsales.add(int(num))
            if dorsales:
                self.dorsales_validos[eq_id] = dorsales

    def reset(self):
        self._votos_track.clear()
        self._dorsal_confirmado.clear()
        self._frames_track.clear()
