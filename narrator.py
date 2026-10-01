"""
narrator.py
===========
Generador de narraciones futbolísticas.

Recibe eventos estructurados desde event_detector.py
y los transforma en frases para mostrar en pantalla.
"""

from tracker_engine import EQUIPO_A, EQUIPO_B


class NarradorFutbol:

    def __init__(self):

        self.frases = {

            "POSESION_INICIAL": [
                "El equipo {equipo} toma la posesion del balon.",
                "El equipo {equipo} comienza con el control del balon.",
                "El balon queda bajo control del equipo {equipo}.",
            ],

            "PASE": [
                "Pase del jugador {anterior} al jugador {jugador}.",
                "El jugador {anterior} conecta con el jugador {jugador}.",
                "El equipo {equipo} mueve el balon hacia el jugador {jugador}.",
            ],

            "CAMBIO_POSESION": [
                "El equipo {equipo} recupera el balon.",
                "Cambio de posesion. El equipo {equipo} toma el control.",
                "El equipo {equipo} recupera la posesion.",
            ],

            "BALON_LIBRE": [
                "El balon queda sin una posesion clara.",
                "El balon esta en disputa.",
                "Ningun jugador tiene una posesion clara del balon.",
            ],
        }

    # =========================================================
    # NOMBRE DEL EQUIPO
    # =========================================================

    def nombre_equipo(self, equipo):

        if equipo == EQUIPO_A:
            return "rojo"

        if equipo == EQUIPO_B:
            return "azul"

        return "desconocido"

    # =========================================================
    # GENERAR FRASE
    # =========================================================

    def narrar(self, evento):

        if not evento:
            return ""

        tipo = evento.get("tipo")

        opciones = self.frases.get(tipo)

        if not opciones:
            return ""

        equipo = self.nombre_equipo(
            evento.get("equipo")
        )

        jugador = evento.get("jugador")

        anterior = evento.get(
            "jugador_anterior"
        )

        # Selección sencilla y estable.
        # Posteriormente podemos hacerla dinámica.
        frase = opciones[0]

        try:

            texto = frase.format(
                equipo=equipo,
                jugador=jugador if jugador is not None else "--",
                anterior=anterior if anterior is not None else "--",
            )

        except Exception:

            return ""

        return self._sanear(texto)

    # =========================================================
    # LIMPIEZA
    # =========================================================

    def _sanear(self, texto):

        if texto is None:
            return ""

        texto = str(texto)

        texto = (
            texto
            .replace("¿", "")
            .replace("?", "")
            .replace("¡", "")
            .replace("!", "")
        )

        return " ".join(texto.split())