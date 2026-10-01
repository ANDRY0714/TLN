"""
event_detector.py
=================
Detector de eventos futbolísticos a partir del estado
producido por tracker_engine.py.

Eventos actuales:
- POSESION_INICIAL
- PASE
- CAMBIO_POSESION
- BALON_LIBRE

No modifica el motor de tracking.
"""

import time

from tracker_engine import EQUIPO_A, EQUIPO_B


class DetectorEventos:
    """
    Convierte cambios en el estado del tracking en eventos futbolísticos.

    Arquitectura:

        tracker_engine
              ↓
          estado{}
              ↓
        DetectorEventos
              ↓
            evento
    """

    def __init__(self):
        # Estado anterior
        self.ultimo_id_posesion = None
        self.ultimo_equipo = 0

        # Para evitar eventos duplicados
        self.ultimo_tipo_evento = None
        self.ultimo_evento_time = 0.0

        # Tiempo mínimo entre eventos iguales
        self.cooldowns = {
            "PASE": 1.2,
            "CAMBIO_POSESION": 1.5,
            "POSESION_INICIAL": 2.0,
            "BALON_LIBRE": 2.0,
        }

    # =========================================================
    # EQUIPO
    # =========================================================

    def _nombre_equipo(self, equipo):
        if equipo == EQUIPO_A:
            return "rojo"

        if equipo == EQUIPO_B:
            return "azul"

        return None

    # =========================================================
    # COOLDOWN
    # =========================================================

    def _puede_generar(self, tipo):
        ahora = time.time()

        cooldown = self.cooldowns.get(tipo, 1.0)

        if (
            self.ultimo_tipo_evento == tipo
            and ahora - self.ultimo_evento_time < cooldown
        ):
            return False

        return True

    def _registrar_evento(self, tipo):
        self.ultimo_tipo_evento = tipo
        self.ultimo_evento_time = time.time()

    # =========================================================
    # EVENTO
    # =========================================================

    def _crear_evento(
        self,
        tipo,
        equipo=None,
        jugador=None,
        jugador_anterior=None,
        equipo_anterior=None,
        zona=None,
    ):
        return {
            "tipo": tipo,
            "equipo": equipo,
            "jugador": jugador,
            "jugador_anterior": jugador_anterior,
            "equipo_anterior": equipo_anterior,
            "zona": zona,
            "timestamp": time.time(),
        }

    # =========================================================
    # PROCESAMIENTO PRINCIPAL
    # =========================================================

    def procesar(self, estado):
        """
        Recibe el diccionario producido por MotorTracking._estado()
        y devuelve un evento o None.
        """

        if not estado:
            return None

        id_actual = estado.get("id_posesion")
        equipo_actual = estado.get("eq_posesion", 0)
        zona = estado.get("zona", "SIN ACCION")

        # -----------------------------------------------------
        # CASO 1: NO HAY JUGADOR CON EL BALON
        # -----------------------------------------------------

        if id_actual is None or equipo_actual not in (EQUIPO_A, EQUIPO_B):

            # Solo informamos de balón libre si anteriormente
            # sí había posesión.
            if (
                self.ultimo_id_posesion is not None
                and self._puede_generar("BALON_LIBRE")
            ):
                evento = self._crear_evento(
                    tipo="BALON_LIBRE",
                    equipo=None,
                    jugador=None,
                    jugador_anterior=self.ultimo_id_posesion,
                    equipo_anterior=self.ultimo_equipo,
                    zona=zona,
                )

                self._registrar_evento("BALON_LIBRE")

                self.ultimo_id_posesion = None
                self.ultimo_equipo = 0

                return evento

            return None

        # -----------------------------------------------------
        # CASO 2: PRIMERA POSESION DETECTADA
        # -----------------------------------------------------

        if self.ultimo_id_posesion is None:

            self.ultimo_id_posesion = id_actual
            self.ultimo_equipo = equipo_actual

            if self._puede_generar("POSESION_INICIAL"):

                evento = self._crear_evento(
                    tipo="POSESION_INICIAL",
                    equipo=equipo_actual,
                    jugador=id_actual,
                    zona=zona,
                )

                self._registrar_evento("POSESION_INICIAL")

                return evento

            return None

        # -----------------------------------------------------
        # CASO 3: SIGUE EL MISMO JUGADOR
        # -----------------------------------------------------

        if id_actual == self.ultimo_id_posesion:

            return None

        # -----------------------------------------------------
        # A PARTIR DE AQUÍ CAMBIO EL JUGADOR
        # -----------------------------------------------------

        jugador_anterior = self.ultimo_id_posesion
        equipo_anterior = self.ultimo_equipo

        # Actualizamos inmediatamente el estado
        self.ultimo_id_posesion = id_actual
        self.ultimo_equipo = equipo_actual

        # -----------------------------------------------------
        # CASO 4: CAMBIO ENTRE JUGADORES DEL MISMO EQUIPO
        # → POSIBLE PASE
        # -----------------------------------------------------

        if (
            equipo_anterior in (EQUIPO_A, EQUIPO_B)
            and equipo_actual == equipo_anterior
        ):

            if not self._puede_generar("PASE"):
                return None

            evento = self._crear_evento(
                tipo="PASE",
                equipo=equipo_actual,
                jugador=id_actual,
                jugador_anterior=jugador_anterior,
                equipo_anterior=equipo_anterior,
                zona=zona,
            )

            self._registrar_evento("PASE")

            return evento

        # -----------------------------------------------------
        # CASO 5: CAMBIO ENTRE EQUIPOS
        # → CAMBIO DE POSESION
        # -----------------------------------------------------

        if (
            equipo_anterior in (EQUIPO_A, EQUIPO_B)
            and equipo_actual in (EQUIPO_A, EQUIPO_B)
            and equipo_actual != equipo_anterior
        ):

            if not self._puede_generar("CAMBIO_POSESION"):
                return None

            evento = self._crear_evento(
                tipo="CAMBIO_POSESION",
                equipo=equipo_actual,
                jugador=id_actual,
                jugador_anterior=jugador_anterior,
                equipo_anterior=equipo_anterior,
                zona=zona,
            )

            self._registrar_evento("CAMBIO_POSESION")

            return evento

        return None

    # =========================================================
    # REINICIO
    # =========================================================

    def reset(self):
        """
        Reinicia el estado cuando se abre/reinicia un video.
        """

        self.ultimo_id_posesion = None
        self.ultimo_equipo = 0

        self.ultimo_tipo_evento = None
        self.ultimo_evento_time = 0.0