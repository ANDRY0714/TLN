"""
narrator.py
===========
Generador de narraciones futbolisticas basado en Redes Semanticas,
Tablas de Reescritura y Carga de Conocimiento en JSON / TOML (TLN).

Arquitectura de Representacion del Conocimiento y NLG:
  1. Red Semantica (RedSemanticaFutbol):
     - Grafo dirigido etiquetado con nodos conceptuales (clases) e instancias.
     - Relaciones taxonomicas (ES_UN, INSTANCIA_DE) con herencia de propiedades
       por recorrido en anchura (BFS).
     - Relaciones de dominio y estado (PERTENECE_A, UBICADO_EN, POSEE, AGENTE,
       RECEPTOR, PASA_A, RECUPERA_DE, TRANSICION_HACIA, CAPITAN_DE, SOCIO_DE).
     - Carga de archivos JSON o TOML con datos y relaciones del partido
       (nombres de equipos, jugadores con su numero/dorsal, posiciones y tripletas).
  2. Motor de Reescritura (MotorReescritura):
     - Transforma los eventos enriquecidos por la red semantica en lenguaje
       natural variado mediante tablas estructurales, lexicas y morfologicas.
"""

from collections import deque
import json
import os
import re

try:
    import tomllib
except ImportError:
    tomllib = None

try:
    from tracker_engine import EQUIPO_A, EQUIPO_B
except ImportError:
    EQUIPO_A = 1
    EQUIPO_B = 2


# =====================================================================
# 0. PARSER Y CARGADOR DE ARCHIVOS JSON / TOML DEL PARTIDO
# =====================================================================

def _parsear_valor_toml_simple(valor_str):
    """Convierte un valor escalar o lista simple de TOML a tipos de Python."""
    v = valor_str.strip()
    if not v:
        return ""
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.startswith("[") and v.endswith("]"):
        interior = v[1:-1].strip()
        if not interior:
            return []
        partes = [p.strip() for p in interior.split(",") if p.strip()]
        return [_parsear_valor_toml_simple(p) for p in partes]
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _parsear_toml_fallback(contenido_toml):
    """
    Parser ligero de respaldo para archivos TOML (tablas [seccion],
    subtablas [a.b] y listas de tablas [[lista]]) en caso de ejecutarse
    en versiones de Python anteriores a 3.11 sin tomllib.
    """
    datos = {}
    contexto_actual = datos

    for linea_cruda in contenido_toml.splitlines():
        en_comillas = False
        char_comilla = None
        linea_limpia = []
        for ch in linea_cruda:
            if ch in ('"', "'"):
                if not en_comillas:
                    en_comillas = True
                    char_comilla = ch
                elif char_comilla == ch:
                    en_comillas = False
            if ch == "#" and not en_comillas:
                break
            linea_limpia.append(ch)

        linea = "".join(linea_limpia).strip()
        if not linea:
            continue

        # Lista de tablas: [[jugadores]] o [[relaciones]]
        if linea.startswith("[[") and linea.endswith("]]"):
            ruta = [p.strip() for p in linea[2:-2].strip().split(".")]
            cursor = datos
            for parte in ruta[:-1]:
                cursor = cursor.setdefault(parte, {})
            clave_lista = ruta[-1]
            lista = cursor.setdefault(clave_lista, [])
            nuevo_item = {}
            lista.append(nuevo_item)
            contexto_actual = nuevo_item
            continue

        # Tabla estandar: [equipos.rojo]
        if linea.startswith("[") and linea.endswith("]"):
            ruta = [p.strip() for p in linea[1:-1].strip().split(".")]
            cursor = datos
            for parte in ruta:
                cursor = cursor.setdefault(parte, {})
            contexto_actual = cursor
            continue

        # Asignacion clave = valor
        if "=" in linea:
            clave, valor = linea.split("=", 1)
            contexto_actual[clave.strip()] = _parsear_valor_toml_simple(valor)

    return datos


def cargar_datos_partido(fuente, formato=None):
    """
    Funcion que recibe una ruta de archivo (.json o .toml), una cadena de texto
    en codigo JSON o TOML, o un diccionario de Python con las relaciones del
    partido (nombres de equipos, jugadores con su numero/dorsal y relaciones
    semanticas) y devuelve el diccionario normalizado.
    """
    if isinstance(fuente, dict):
        return dict(fuente)

    if not isinstance(fuente, str) or not fuente.strip():
        raise ValueError("Se requiere una ruta de archivo, cadena JSON/TOML o diccionario.")

    contenido = fuente
    formato_detectado = (formato or "").strip().lower()

    if os.path.isfile(fuente):
        if not formato_detectado:
            ext = os.path.splitext(fuente)[1].lower()
            if ext == ".json":
                formato_detectado = "json"
            elif ext == ".toml":
                formato_detectado = "toml"
        with open(fuente, "r", encoding="utf-8") as f:
            contenido = f.read()

    texto_strip = contenido.strip()

    if not formato_detectado:
        if texto_strip.startswith("{") or (texto_strip.startswith("[") and not texto_strip.startswith("[[")):
            primera_linea = texto_strip.splitlines()[0].strip()
            if primera_linea.startswith("{"):
                formato_detectado = "json"
            else:
                formato_detectado = "toml"
        else:
            formato_detectado = "toml"

    if formato_detectado == "json":
        return json.loads(texto_strip)

    if formato_detectado == "toml":
        if tomllib is not None:
            return tomllib.loads(texto_strip)
        return _parsear_toml_fallback(texto_strip)

    raise ValueError(f"Formato no soportado: {formato_detectado}. Usa 'json' o 'toml'.")


# =====================================================================
# 1. ESTRUCTURAS DE LA RED SEMANTICA (NODOS, ARISTAS E INFERENCIA)
# =====================================================================

class NodoSemantico:
    """Representa un concepto, categoria o instancia dentro de la red semantica."""

    def __init__(self, id_nodo, tipo="CONCEPTO", propiedades=None):
        self.id = id_nodo
        self.tipo = tipo  # "CONCEPTO" (clase) o "INSTANCIA" (entidad concreta)
        self.propiedades = dict(propiedades) if propiedades else {}

    def actualizar_propiedades(self, nuevas_propiedades):
        if nuevas_propiedades:
            self.propiedades.update(nuevas_propiedades)

    def __repr__(self):
        return f"NodoSemantico({self.id!r}, tipo={self.tipo!r})"


class RedSemanticaFutbol:
    """
    Grafo semantico dirigido con soporte para:
      - Tripletas (origen, relacion, destino)
      - Herencia multiple de propiedades mediante enlaces 'ES_UN' e 'INSTANCIA_DE'
      - Carga de conocimiento del partido desde JSON o TOML (equipos, jugadores,
        numeros/dorsales y relaciones personalizadas)
      - Inferencia contextual sobre el estado del partido y secuencias de jugadas
    """

    RELACIONES_HERENCIA = ("INSTANCIA_DE", "ES_UN")

    def __init__(self):
        self.nodos = {}          # id_nodo -> NodoSemantico
        self.aristas = []        # lista de diccionarios {origen, relacion, destino, meta}
        self._zona_anterior = None
        self._sector_anterior = None
        self._historial_pases = {}  # (equipo, j_ant, j_act) -> conteo

        # Conocimiento cargado del partido (equipos, plantilla y mapeo de tracker IDs)
        self.plantilla_equipos = {
            EQUIPO_A: {},  # id_o_numero -> dict con {nombre, numero, posicion, apodo, nodo_id}
            EQUIPO_B: {},
        }
        self._mapeo_tracker_a_jugador = {}  # (equipo_id, tracker_id) -> info_jugador
        self._cursores_plantilla = {EQUIPO_A: 0, EQUIPO_B: 0}
        self._datos_partido_cargados = None

        self._inicializar_ontologia_base()

    # -----------------------------------------------------------------
    # Operaciones basicas del grafo semantico
    # -----------------------------------------------------------------

    def agregar_nodo(self, id_nodo, tipo="CONCEPTO", propiedades=None):
        if id_nodo not in self.nodos:
            self.nodos[id_nodo] = NodoSemantico(id_nodo, tipo=tipo, propiedades=propiedades)
        else:
            if tipo:
                self.nodos[id_nodo].tipo = tipo
            self.nodos[id_nodo].actualizar_propiedades(propiedades)
        return self.nodos[id_nodo]

    def agregar_relacion(self, origen, relacion, destino, unica_por_origen=False, meta=None):
        """
        Inserta una tripleta (origen -[relacion]-> destino).
        Si unica_por_origen=True, reemplaza cualquier relacion previa del mismo
        tipo que salga del nodo origen (util para estados como UBICADO_EN o POSEE).
        """
        if origen not in self.nodos:
            self.agregar_nodo(origen, tipo="INSTANCIA")
        if destino not in self.nodos:
            self.agregar_nodo(destino, tipo="CONCEPTO")

        if unica_por_origen:
            self.aristas = [
                a for a in self.aristas
                if not (a["origen"] == origen and a["relacion"] == relacion)
            ]

        nueva_arista = {
            "origen": origen,
            "relacion": relacion,
            "destino": destino,
            "meta": dict(meta) if meta else {},
        }
        if nueva_arista not in self.aristas:
            self.aristas.append(nueva_arista)

    def eliminar_relaciones_de(self, origen, relacion=None):
        if relacion is None:
            self.aristas = [a for a in self.aristas if a["origen"] != origen]
        else:
            self.aristas = [
                a for a in self.aristas
                if not (a["origen"] == origen and a["relacion"] == relacion)
            ]

    def consultar_destinos(self, origen, relacion):
        return [
            a["destino"] for a in self.aristas
            if a["origen"] == origen and a["relacion"] == relacion
        ]

    def consultar_origenes(self, relacion, destino):
        return [
            a["origen"] for a in self.aristas
            if a["relacion"] == relacion and a["destino"] == destino
        ]

    def tiene_relacion(self, origen, relacion, destino=None):
        for a in self.aristas:
            if a["origen"] == origen and a["relacion"] == relacion:
                if destino is None or a["destino"] == destino:
                    return True
        return False

    def obtener_tripletas(self, origen=None, relacion=None):
        """Devuelve las tripletas (origen, relacion, destino) activas en la red."""
        resultado = []
        for a in self.aristas:
            if origen is not None and a["origen"] != origen:
                continue
            if relacion is not None and a["relacion"] != relacion:
                continue
            resultado.append((a["origen"], a["relacion"], a["destino"]))
        return resultado

    # -----------------------------------------------------------------
    # Herencia taxonomica por recorrido BFS (INSTANCIA_DE / ES_UN)
    # -----------------------------------------------------------------

    def heredar_propiedades(self, id_nodo):
        """
        Recorre la jerarquia taxonomica hacia arriba (INSTANCIA_DE -> ES_UN)
        mediante busqueda en anchura (BFS). Las propiedades mas especificas
        del nodo hijo tienen prioridad sobre las de sus ancestros.
        """
        if id_nodo not in self.nodos:
            return {}

        visitados = set()
        cola = deque([id_nodo])
        orden_jerarquico = []

        while cola:
            actual = cola.popleft()
            if actual in visitados or actual not in self.nodos:
                continue
            visitados.add(actual)
            orden_jerarquico.append(actual)

            for rel in self.RELACIONES_HERENCIA:
                for padre in self.consultar_destinos(actual, rel):
                    if padre not in visitados:
                        cola.append(padre)

        propiedades_heredadas = {}
        for nodo_id in reversed(orden_jerarquico):
            propiedades_heredadas.update(self.nodos[nodo_id].propiedades)

        return propiedades_heredadas

    def es_subtipo_de(self, id_nodo, concepto_ancestro):
        """Verifica si id_nodo es instancia o subclase de concepto_ancestro."""
        if id_nodo == concepto_ancestro:
            return True
        visitados = set()
        cola = deque([id_nodo])
        while cola:
            actual = cola.popleft()
            if actual == concepto_ancestro:
                return True
            if actual in visitados:
                continue
            visitados.add(actual)
            for rel in self.RELACIONES_HERENCIA:
                for padre in self.consultar_destinos(actual, rel):
                    cola.append(padre)
        return False

    # -----------------------------------------------------------------
    # Ontologia base del dominio futbolistico
    # -----------------------------------------------------------------

    def _inicializar_ontologia_base(self):
        # Jerarquia superior de entidades
        self.agregar_nodo("Entidad", "CONCEPTO", {"dominio": "futbol"})
        self.agregar_nodo("Agente", "CONCEPTO", {"animado": True})
        self.agregar_nodo("Equipo", "CONCEPTO", {"colectivo": True})
        self.agregar_nodo("Jugador", "CONCEPTO", {"rol_contextual": "el futbolista"})
        self.agregar_nodo("ObjetoJuego", "CONCEPTO", {"animado": False})
        self.agregar_nodo("ZonaCampo", "CONCEPTO", {"espacio": "terreno_de_juego"})
        self.agregar_nodo("EventoJuego", "CONCEPTO", {"temporal": True})

        self.agregar_relacion("Agente", "ES_UN", "Entidad")
        self.agregar_relacion("Equipo", "ES_UN", "Agente")
        self.agregar_relacion("Jugador", "ES_UN", "Agente")
        self.agregar_relacion("ObjetoJuego", "ES_UN", "Entidad")
        self.agregar_relacion("ZonaCampo", "ES_UN", "Entidad")
        self.agregar_relacion("EventoJuego", "ES_UN", "Entidad")

        # Subclases de sectores tacticos del campo (heredan de ZonaCampo)
        self.agregar_nodo("SectorDefensivo", "CONCEPTO", {
            "sector_tactico": "DEFENSA",
            "nivel_vertical": 1,
            "fase_tactica": "en fase de salida desde el fondo",
            "fase_recuperacion": "con un corte providencial en campo propio",
            "rol_habitual": "el defensor",
            "riesgo": "alto",
        })
        self.agregar_nodo("SectorMediocampo", "CONCEPTO", {
            "sector_tactico": "MEDIOCAMPO",
            "nivel_vertical": 2,
            "fase_tactica": "en fase de gestacion y elaboracion",
            "fase_recuperacion": "ganando la dividida en la zona de volantes",
            "rol_habitual": "el mediocampista",
            "riesgo": "medio",
        })
        self.agregar_nodo("SectorOfensivo", "CONCEPTO", {
            "sector_tactico": "ATAQUE",
            "nivel_vertical": 3,
            "fase_tactica": "en zona de peligro y definicion",
            "fase_recuperacion": "mediante presion alta en territorio rival",
            "rol_habitual": "el atacante",
            "riesgo": "ofensivo",
        })

        self.agregar_relacion("SectorDefensivo", "ES_UN", "ZonaCampo")
        self.agregar_relacion("SectorMediocampo", "ES_UN", "ZonaCampo")
        self.agregar_relacion("SectorOfensivo", "ES_UN", "ZonaCampo")

        # Subclases de carriles laterales/centrales
        self.agregar_nodo("CarrilIzquierdo", "CONCEPTO", {"carril": "IZQUIERDO", "banda": "por el costado izquierdo"})
        self.agregar_nodo("CarrilCentral", "CONCEPTO", {"carril": "CENTRAL", "banda": "por el eje central"})
        self.agregar_nodo("CarrilDerecho", "CONCEPTO", {"carril": "DERECHO", "banda": "por el costado derecho"})

        self.agregar_relacion("CarrilIzquierdo", "ES_UN", "ZonaCampo")
        self.agregar_relacion("CarrilCentral", "ES_UN", "ZonaCampo")
        self.agregar_relacion("CarrilDerecho", "ES_UN", "ZonaCampo")

        # Instancias de las 9 zonas 3x3 de tracker_engine._zona_3x3()
        zonas_config = [
            ("Def. Izq.", "SectorDefensivo", "CarrilIzquierdo", "el lateral izquierdo"),
            ("Defensa", "SectorDefensivo", "CarrilCentral", "el zaguero central"),
            ("Def. Der.", "SectorDefensivo", "CarrilDerecho", "el lateral derecho"),
            ("Med. Izq.", "SectorMediocampo", "CarrilIzquierdo", "el interior izquierdo"),
            ("Centro", "SectorMediocampo", "CarrilCentral", "el volante central"),
            ("Med. Der.", "SectorMediocampo", "CarrilDerecho", "el interior derecho"),
            ("Ata. Izq.", "SectorOfensivo", "CarrilIzquierdo", "el extremo izquierdo"),
            ("Ataque", "SectorOfensivo", "CarrilCentral", "el delantero centro"),
            ("Ata. Der.", "SectorOfensivo", "CarrilDerecho", "el extremo derecho"),
        ]
        for id_zona, sector_padre, carril_padre, rol_especifico in zonas_config:
            self.agregar_nodo(id_zona, "INSTANCIA", {
                "nombre_zona": id_zona,
                "rol_contextual": rol_especifico,
            })
            self.agregar_relacion(id_zona, "INSTANCIA_DE", sector_padre)
            self.agregar_relacion(id_zona, "INSTANCIA_DE", carril_padre)

        # Clases de Eventos
        self.agregar_nodo("POSESION_INICIAL", "CONCEPTO", {
            "categoria_evento": "CONTROL",
            "intencion_semantica": "iniciar la construccion de juego",
        })
        self.agregar_nodo("PASE", "CONCEPTO", {
            "categoria_evento": "ASOCIACION",
            "intencion_semantica": "mantener la circulacion colectiva",
        })
        self.agregar_nodo("CAMBIO_POSESION", "CONCEPTO", {
            "categoria_evento": "TRANSICION",
            "intencion_semantica": "romper el avance rival",
        })
        self.agregar_nodo("BALON_LIBRE", "CONCEPTO", {
            "categoria_evento": "DISPUTA",
            "intencion_semantica": "disputar la pelota dividida",
        })

        for ev_tipo in ("POSESION_INICIAL", "PASE", "CAMBIO_POSESION", "BALON_LIBRE"):
            self.agregar_relacion(ev_tipo, "ES_UN", "EventoJuego")

        # Instancias fijas de equipos y balon
        self.agregar_nodo("Equipo_Rojo", "INSTANCIA", {
            "id_numerico": EQUIPO_A,
            "color": "rojo",
            "color_f": "roja",
            "nombre_oficial": None,
            "apodos": [],
        })
        self.agregar_relacion("Equipo_Rojo", "INSTANCIA_DE", "Equipo")

        self.agregar_nodo("Equipo_Azul", "INSTANCIA", {
            "id_numerico": EQUIPO_B,
            "color": "azul",
            "color_f": "azul",
            "nombre_oficial": None,
            "apodos": [],
        })
        self.agregar_relacion("Equipo_Azul", "INSTANCIA_DE", "Equipo")

        self.agregar_relacion("Equipo_Rojo", "RIVAL_DE", "Equipo_Azul")
        self.agregar_relacion("Equipo_Azul", "RIVAL_DE", "Equipo_Rojo")

        self.agregar_nodo("Balon", "INSTANCIA", {"nombre": "esferico"})
        self.agregar_relacion("Balon", "INSTANCIA_DE", "ObjetoJuego")

    # -----------------------------------------------------------------
    # Carga de archivo JSON / TOML con relaciones del partido
    # -----------------------------------------------------------------

    def _resolver_equipo_id(self, ref_equipo):
        """Convierte referencias ('rojo', 'equipo_a', 1, 'azul', 'equipo_b', 2) en EQUIPO_A o EQUIPO_B."""
        if ref_equipo in (EQUIPO_A, 1, "1"):
            return EQUIPO_A
        if ref_equipo in (EQUIPO_B, 2, "2"):
            return EQUIPO_B
        if isinstance(ref_equipo, str):
            val = ref_equipo.strip().lower()
            if val in ("rojo", "equipo_a", "equipo_rojo", "local", "a"):
                return EQUIPO_A
            if val in ("azul", "equipo_b", "equipo_azul", "visitante", "b"):
                return EQUIPO_B
            nombre_a = (self.nodos["Equipo_Rojo"].propiedades.get("nombre_oficial") or "").lower()
            nombre_b = (self.nodos["Equipo_Azul"].propiedades.get("nombre_oficial") or "").lower()
            if nombre_a and val == nombre_a:
                return EQUIPO_A
            if nombre_b and val == nombre_b:
                return EQUIPO_B
        return None

    def cargar_relaciones_partido(self, fuente, formato=None):
        """
        Puebla la Red Semantica a partir de un archivo o cadena JSON / TOML
        con la informacion del partido:
          - Metadatos del partido (estadio, competicion, arbitro)
          - Equipos (nombres oficiales, apodos, entrenador, jugadores)
          - Jugadores (numero/dorsal, id de tracker opcional, nombre, posicion/rol, apodo)
          - Relaciones semanticas adicionales (CAPITAN_DE, SOCIO_DE, RIVAL_DIRECTO_DE, etc.)
        """
        datos = cargar_datos_partido(fuente, formato=formato)
        self._datos_partido_cargados = datos

        # 1. Metadatos generales del partido
        info_partido = datos.get("partido", {})
        if isinstance(info_partido, dict) and info_partido:
            self.agregar_nodo("Partido_Actual", "INSTANCIA", info_partido)
            self.agregar_relacion("Partido_Actual", "INSTANCIA_DE", "Entidad")

        # 2. Configuracion de equipos (soporta diccionario 'equipos' o claves directas)
        equipos_cfg = datos.get("equipos", {})
        if isinstance(equipos_cfg, dict):
            for clave_eq, info_eq in equipos_cfg.items():
                if not isinstance(info_eq, dict):
                    continue
                eq_id = self._resolver_equipo_id(info_eq.get("id", clave_eq))
                if eq_id is None:
                    continue
                nodo_eq = self._id_nodo_equipo(eq_id)
                nombre_eq = info_eq.get("nombre") or info_eq.get("nombre_oficial")
                apodos_raw = info_eq.get("apodos") or info_eq.get("apodo") or []
                if isinstance(apodos_raw, str):
                    apodos_lista = [apodos_raw]
                else:
                    apodos_lista = list(apodos_raw)

                props_eq = {
                    "nombre_oficial": nombre_eq,
                    "apodos": apodos_lista,
                    "entrenador": info_eq.get("entrenador") or info_eq.get("dt"),
                }
                self.agregar_nodo(nodo_eq, "INSTANCIA", props_eq)

                if props_eq["entrenador"]:
                    nodo_dt = f"DT_{props_eq['entrenador'].replace(' ', '_')}"
                    self.agregar_nodo(nodo_dt, "INSTANCIA", {
                        "nombre": props_eq["entrenador"],
                        "rol_contextual": "el director tecnico",
                    })
                    self.agregar_relacion(nodo_dt, "INSTANCIA_DE", "Agente")
                    self.agregar_relacion(nodo_dt, "DIRIGE_A", nodo_eq, unica_por_origen=True)

                jugadores_eq = info_eq.get("jugadores", [])
                if isinstance(jugadores_eq, list):
                    for jug in jugadores_eq:
                        if isinstance(jug, dict):
                            jug_copia = dict(jug)
                            jug_copia.setdefault("equipo", eq_id)
                            self._registrar_jugador_configurado(jug_copia)
                elif isinstance(jugadores_eq, dict):
                    for num_str, val_jug in jugadores_eq.items():
                        if isinstance(val_jug, str):
                            self._registrar_jugador_configurado({
                                "numero": int(num_str) if str(num_str).isdigit() else num_str,
                                "nombre": val_jug,
                                "equipo": eq_id,
                            })
                        elif isinstance(val_jug, dict):
                            jug_copia = dict(val_jug)
                            jug_copia.setdefault(
                                "numero",
                                int(num_str) if str(num_str).isdigit() else num_str,
                            )
                            jug_copia.setdefault("equipo", eq_id)
                            self._registrar_jugador_configurado(jug_copia)

        # 3. Lista de jugadores en el nivel raiz ([[jugadores]] en TOML o "jugadores": [...] en JSON)
        jugadores_raiz = datos.get("jugadores", [])
        if isinstance(jugadores_raiz, list):
            for jug in jugadores_raiz:
                if isinstance(jug, dict):
                    self._registrar_jugador_configurado(jug)

        # 4. Relaciones semanticas explicitas ([[relaciones]] en TOML o "relaciones": [...] en JSON)
        relaciones_cfg = datos.get("relaciones", [])
        if isinstance(relaciones_cfg, list):
            for rel_item in relaciones_cfg:
                if not isinstance(rel_item, dict):
                    continue
                origen_raw = rel_item.get("origen") or rel_item.get("sujeto")
                relacion = rel_item.get("relacion") or rel_item.get("predicado")
                destino_raw = rel_item.get("destino") or rel_item.get("objeto")
                if not (origen_raw and relacion and destino_raw):
                    continue

                nodo_origen = self._resolver_referencia_nodo(origen_raw, rel_item.get("equipo_origen"))
                nodo_destino = self._resolver_referencia_nodo(destino_raw, rel_item.get("equipo_destino"))
                rel_norm = str(relacion).strip().upper().replace(" ", "_")

                self.agregar_relacion(nodo_origen, rel_norm, nodo_destino)

                if rel_norm == "CAPITAN_DE" and nodo_origen in self.nodos:
                    self.nodos[nodo_origen].actualizar_propiedades({"es_capitan": True})

        return {
            "equipos_configurados": {
                EQUIPO_A: self.nodos["Equipo_Rojo"].propiedades.get("nombre_oficial"),
                EQUIPO_B: self.nodos["Equipo_Azul"].propiedades.get("nombre_oficial"),
            },
            "total_jugadores": sum(len(p) for p in self.plantilla_equipos.values()),
            "total_tripletas": len(self.aristas),
        }

    def _registrar_jugador_configurado(self, datos_jug):
        eq_id = self._resolver_equipo_id(datos_jug.get("equipo", EQUIPO_A))
        if eq_id is None:
            eq_id = EQUIPO_A

        numero = datos_jug.get("numero") if datos_jug.get("numero") is not None else datos_jug.get("dorsal")
        tracker_id = datos_jug.get("id") if datos_jug.get("id") is not None else numero
        if numero is None and tracker_id is not None:
            numero = tracker_id

        nombre = datos_jug.get("nombre") or f"Jugador #{numero}"
        posicion = datos_jug.get("posicion") or datos_jug.get("rol")
        apodo = datos_jug.get("apodo")
        es_capitan = bool(datos_jug.get("capitan", False))

        id_nodo = f"Jugador_Eq{eq_id}_{numero}"
        propiedades_nodo = {
            "dorsal": numero,
            "tracker_id": tracker_id,
            "nombre": nombre,
            "apodo": apodo,
            "equipo_id": eq_id,
            "es_capitan": es_capitan,
        }
        if posicion:
            propiedades_nodo["rol_contextual"] = (
                posicion if posicion.lower().startswith(("el ", "la ")) else f"el {posicion.lower()}"
            )

        self.agregar_nodo(id_nodo, "INSTANCIA", propiedades_nodo)
        self.agregar_relacion(id_nodo, "INSTANCIA_DE", "Jugador")

        nodo_eq = self._id_nodo_equipo(eq_id)
        if nodo_eq:
            self.agregar_relacion(id_nodo, "PERTENECE_A", nodo_eq, unica_por_origen=True)
            if es_capitan:
                self.agregar_relacion(id_nodo, "CAPITAN_DE", nodo_eq, unica_por_origen=True)

        info_registro = {
            "nodo_id": id_nodo,
            "nombre": nombre,
            "numero": numero,
            "tracker_id": tracker_id,
            "rol_contextual": propiedades_nodo.get("rol_contextual"),
            "apodo": apodo,
            "es_capitan": es_capitan,
            "equipo_id": eq_id,
        }
        if numero is not None:
            self.plantilla_equipos[eq_id][int(numero) if str(numero).isdigit() else numero] = info_registro
        if tracker_id is not None:
            clave_t = int(tracker_id) if str(tracker_id).isdigit() else tracker_id
            self.plantilla_equipos[eq_id][clave_t] = info_registro
            self._mapeo_tracker_a_jugador[(eq_id, clave_t)] = info_registro

        socio_num = datos_jug.get("socio")
        if socio_num is not None:
            nodo_socio = f"Jugador_Eq{eq_id}_{socio_num}"
            self.agregar_relacion(id_nodo, "SOCIO_DE", nodo_socio)

    def _resolver_referencia_nodo(self, ref, equipo_hint=None):
        """Resuelve un nombre o identificador de un archivo JSON/TOML al id_nodo de la red."""
        if ref in self.nodos:
            return ref

        eq_resolved = self._resolver_equipo_id(ref)
        if eq_resolved is not None:
            return self._id_nodo_equipo(eq_resolved)

        ref_str = str(ref).strip()
        eq_hint_id = self._resolver_equipo_id(equipo_hint) if equipo_hint is not None else None

        for id_nodo, nodo in self.nodos.items():
            if not id_nodo.startswith("Jugador_"):
                continue
            props = nodo.propiedades
            if eq_hint_id is not None and props.get("equipo_id") != eq_hint_id:
                continue
            if str(props.get("nombre", "")).lower() == ref_str.lower():
                return id_nodo
            if str(props.get("dorsal", "")) == ref_str.lstrip("#"):
                return id_nodo

        if ref_str.isdigit() and eq_hint_id is not None:
            return f"Jugador_Eq{eq_hint_id}_{ref_str}"
        return ref_str

    def obtener_info_jugador(self, jugador_id, equipo_id=None, zona=None, excluir_nodo_id=None):
        """
        Busca la identidad semantica de un jugador a partir del ID entregado por
        tracker_engine.py:
          1. Si coincide con un numero/ID configurado en el equipo, lo usa directamente.
          2. Si el tracker_engine genero un ID dinamico (ej. 18, 25) no listado
             literalmente pero el equipo tiene plantilla cargada, le asigna de forma
             estable un jugador distinto al emisor actual.
        """
        if jugador_id is None:
            return None

        clave_id = int(jugador_id) if str(jugador_id).isdigit() else jugador_id
        equipos_busqueda = (
            [equipo_id] if equipo_id in (EQUIPO_A, EQUIPO_B) else [EQUIPO_A, EQUIPO_B]
        )

        # 1. Busqueda exacta o previamente vinculada para este tracker_id
        for eq in equipos_busqueda:
            if (eq, clave_id) in self._mapeo_tracker_a_jugador:
                info = self._mapeo_tracker_a_jugador[(eq, clave_id)]
                if not excluir_nodo_id or info["nodo_id"] != excluir_nodo_id:
                    return info
            if clave_id in self.plantilla_equipos.get(eq, {}):
                info = self.plantilla_equipos[eq][clave_id]
                self._mapeo_tracker_a_jugador[(eq, clave_id)] = info
                return info

        # 2. Vinculacion automatica con la plantilla del equipo si el tracker_id es dinamico
        if equipo_id in (EQUIPO_A, EQUIPO_B) and self.plantilla_equipos[equipo_id]:
            candidatos_unicos = []
            vistos = set()
            for info in self.plantilla_equipos[equipo_id].values():
                if info["nodo_id"] not in vistos:
                    vistos.add(info["nodo_id"])
                    if not excluir_nodo_id or info["nodo_id"] != excluir_nodo_id:
                        candidatos_unicos.append(info)

            if candidatos_unicos:
                idx = self._cursores_plantilla[equipo_id] % len(candidatos_unicos)
                self._cursores_plantilla[equipo_id] += 1
                asignado = candidatos_unicos[idx]
                self._mapeo_tracker_a_jugador[(equipo_id, clave_id)] = asignado
                return asignado

        return None

    # -----------------------------------------------------------------
    # Actualizacion dinamica e inferencia a partir de un evento
    # -----------------------------------------------------------------

    def _id_nodo_equipo(self, equipo_id):
        if equipo_id == EQUIPO_A:
            return "Equipo_Rojo"
        if equipo_id == EQUIPO_B:
            return "Equipo_Azul"
        return None

    def _asegurar_nodo_jugador(self, jugador_id, equipo_id=None, zona=None, excluir_nodo_id=None):
        if jugador_id is None:
            return None

        info_cfg = self.obtener_info_jugador(
            jugador_id, equipo_id, zona, excluir_nodo_id=excluir_nodo_id
        )
        if info_cfg:
            id_nodo = info_cfg["nodo_id"]
        else:
            id_nodo = f"Jugador_{jugador_id}"
            self.agregar_nodo(id_nodo, "INSTANCIA", {"dorsal": jugador_id})
            self.agregar_relacion(id_nodo, "INSTANCIA_DE", "Jugador")

        nodo_eq = self._id_nodo_equipo(equipo_id)
        if nodo_eq:
            self.agregar_relacion(id_nodo, "PERTENECE_A", nodo_eq, unica_por_origen=True)

        if zona and zona in self.nodos:
            self.agregar_relacion(id_nodo, "UBICADO_EN", zona, unica_por_origen=True)

        return id_nodo

    def registrar_evento_e_inferir(self, evento):
        """
        Actualiza los enlaces de la red semantica con los datos del evento actual
        y ejecuta inferencia por herencia y recorrido de relaciones para devolver
        un diccionario de descriptores semanticos listos para el narrador.
        """
        tipo = evento.get("tipo")
        equipo_id = evento.get("equipo")
        equipo_ant_id = evento.get("equipo_anterior")
        jugador_id = evento.get("jugador")
        jugador_ant_id = evento.get("jugador_anterior")
        zona = evento.get("zona")

        nodo_eq = self._id_nodo_equipo(equipo_id)
        nodo_eq_ant = self._id_nodo_equipo(equipo_ant_id)
        zona_valida = zona if (zona and zona in self.nodos) else None

        eq_ant_efectivo = equipo_ant_id if equipo_ant_id is not None else equipo_id
        nodo_jug_ant = self._asegurar_nodo_jugador(
            jugador_ant_id, eq_ant_efectivo, self._zona_anterior
        )
        nodo_jug = self._asegurar_nodo_jugador(
            jugador_id,
            equipo_id,
            zona_valida,
            excluir_nodo_id=nodo_jug_ant if tipo == "PASE" else None,
        )

        # Limpiar evento activo previo y registrar el nuevo en la red
        self.eliminar_relaciones_de("Evento_Actual")
        self.agregar_nodo("Evento_Actual", "INSTANCIA", {"tipo": tipo})
        if tipo in self.nodos:
            self.agregar_relacion("Evento_Actual", "INSTANCIA_DE", tipo, unica_por_origen=True)

        if zona_valida:
            self.agregar_relacion("Balon", "UBICADO_EN", zona_valida, unica_por_origen=True)
            self.agregar_relacion("Evento_Actual", "OCURRE_EN", zona_valida, unica_por_origen=True)

        # Actualizar relaciones de posesion y accion segun el tipo de evento
        if tipo in ("POSESION_INICIAL", "PASE", "CAMBIO_POSESION"):
            if nodo_jug:
                self.agregar_relacion("Balon", "POSEIDO_POR", nodo_jug, unica_por_origen=True)
                self.agregar_relacion(nodo_jug, "POSEE", "Balon", unica_por_origen=True)
            if nodo_eq:
                self.agregar_relacion("Balon", "CONTROLADO_POR", nodo_eq, unica_por_origen=True)
        elif tipo == "BALON_LIBRE":
            self.eliminar_relaciones_de("Balon", "POSEIDO_POR")
            self.eliminar_relaciones_de("Balon", "CONTROLADO_POR")

        if tipo == "PASE" and nodo_jug_ant and nodo_jug:
            self.agregar_relacion("Evento_Actual", "AGENTE", nodo_jug_ant, unica_por_origen=True)
            self.agregar_relacion("Evento_Actual", "RECEPTOR", nodo_jug, unica_por_origen=True)
            self.agregar_relacion(nodo_jug_ant, "PASA_A", nodo_jug)
            par = (equipo_id, nodo_jug_ant, nodo_jug)
            self._historial_pases[par] = self._historial_pases.get(par, 0) + 1

        elif tipo == "CAMBIO_POSESION" and nodo_jug:
            self.agregar_relacion("Evento_Actual", "AGENTE", nodo_jug, unica_por_origen=True)
            if nodo_eq_ant:
                self.agregar_relacion(nodo_eq or nodo_jug, "RECUPERA_DE", nodo_eq_ant, unica_por_origen=True)

        # -------------------------------------------------------------
        # Inferencia semantica mediante herencia de propiedades (BFS)
        # -------------------------------------------------------------
        props_zona = self.heredar_propiedades(zona_valida) if zona_valida else {}
        props_zona_ant = self.heredar_propiedades(self._zona_anterior) if self._zona_anterior else {}
        props_evento = self.heredar_propiedades("Evento_Actual")
        props_jug = self.heredar_propiedades(nodo_jug) if nodo_jug else {}
        props_jug_ant = self.heredar_propiedades(nodo_jug_ant) if nodo_jug_ant else {}

        rol_jugador = (
            props_jug.get("rol_contextual")
            if props_jug.get("rol_contextual") and props_jug.get("rol_contextual") != "el futbolista"
            else (props_zona.get("rol_contextual") or props_zona.get("rol_habitual") or "el jugador")
        )
        rol_jugador_ant = (
            props_jug_ant.get("rol_contextual")
            if props_jug_ant.get("rol_contextual") and props_jug_ant.get("rol_contextual") != "el futbolista"
            else (props_zona_ant.get("rol_contextual") or props_zona_ant.get("rol_habitual") or "el jugador")
        )

        fase_tactica = props_zona.get("fase_tactica", "")
        fase_recuperacion = props_zona.get("fase_recuperacion", "con determinacion")
        nivel_actual = props_zona.get("nivel_vertical")
        nivel_anterior = props_zona_ant.get("nivel_vertical")

        # Inferir progresion, asociacion entre socios o duelos individuales
        matiz_semantico = props_evento.get("intencion_semantica", "")
        if tipo == "PASE":
            es_socio = (
                nodo_jug_ant and nodo_jug and (
                    self.tiene_relacion(nodo_jug_ant, "SOCIO_DE", nodo_jug)
                    or self.tiene_relacion(nodo_jug, "SOCIO_DE", nodo_jug_ant)
                )
            )
            par_inverso = (equipo_id, nodo_jug, nodo_jug_ant)
            if self._historial_pases.get(par_inverso, 0) > 0:
                matiz_semantico = "armando la pared asociativa"
            elif es_socio:
                matiz_semantico = "conectando con su socio habitual en la cancha"
            elif nivel_actual is not None and nivel_anterior is not None:
                if nivel_actual > nivel_anterior:
                    matiz_semantico = "rompiendo lineas en progresion ofensiva"
                elif nivel_actual < nivel_anterior:
                    matiz_semantico = "apoyandose atras para reorganizar la salida"
                else:
                    matiz_semantico = f"manteniendo la circulacion {fase_tactica}".strip()
            elif fase_tactica:
                matiz_semantico = fase_tactica

        elif tipo == "CAMBIO_POSESION":
            es_rival_directo = (
                nodo_jug and nodo_jug_ant and (
                    self.tiene_relacion(nodo_jug, "RIVAL_DIRECTO_DE", nodo_jug_ant)
                    or self.tiene_relacion(nodo_jug_ant, "RIVAL_DIRECTO_DE", nodo_jug)
                )
            )
            if es_rival_directo:
                matiz_semantico = f"ganando el duelo individual {fase_recuperacion}".strip()
            else:
                matiz_semantico = fase_recuperacion

        elif tipo == "POSESION_INICIAL":
            matiz_semantico = fase_tactica or "para iniciar la jugada"

        if zona_valida:
            if self._zona_anterior and self._zona_anterior != zona_valida:
                self.agregar_relacion(
                    self._zona_anterior,
                    "TRANSICION_HACIA",
                    zona_valida,
                    unica_por_origen=True,
                )
            self._zona_anterior = zona_valida
            self._sector_anterior = props_zona.get("sector_tactico")

        return {
            "nodo_jugador": nodo_jug,
            "nodo_jugador_ant": nodo_jug_ant,
            "props_jugador": props_jug,
            "props_jugador_ant": props_jug_ant,
            "rol_jugador": rol_jugador,
            "rol_jugador_ant": rol_jugador_ant,
            "fase_tactica": fase_tactica,
            "fase_recuperacion": fase_recuperacion,
            "matiz_semantico": matiz_semantico,
            "sector_tactico": props_zona.get("sector_tactico", "CAMPO"),
            "carril_tactico": props_zona.get("carril", "CENTRAL"),
            "tripletas_evento": self.obtener_tripletas(origen="Evento_Actual")
            + self.obtener_tripletas(origen="Balon"),
        }

    def reset(self):
        """
        Reinicia las relaciones dinamicas del partido en curso conservando
        la ontologia base y los datos de equipos/jugadores cargados desde JSON/TOML.
        """
        datos_previos = self._datos_partido_cargados
        self.nodos.clear()
        self.aristas.clear()
        self._zona_anterior = None
        self._sector_anterior = None
        self._historial_pases.clear()
        self.plantilla_equipos = {EQUIPO_A: {}, EQUIPO_B: {}}
        self._mapeo_tracker_a_jugador.clear()
        self._cursores_plantilla = {EQUIPO_A: 0, EQUIPO_B: 0}
        self._inicializar_ontologia_base()
        if datos_previos is not None:
            self.cargar_relaciones_partido(datos_previos)


# =====================================================================
# 2. MOTOR DE REESCRITURA DIRIGIDO POR TABLAS
# =====================================================================

class MotorReescritura:
    """
    Motor de reescritura dirigido por tablas para Generacion de Lenguaje Natural (NLG).
    Expande simbolos no-terminales (<SIMBOLO>) alimentados tanto por las tablas
    lexicas como por las inferencias de la Red Semantica.
    """

    PATRON_NO_TERMINAL = re.compile(r"<([A-Z0-9_]+)>")

    def __init__(self, tabla_estructural, tabla_lexica, tabla_morfologica):
        self.tabla_estructural = tabla_estructural
        self.tabla_lexica = tabla_lexica
        self.tabla_morfologica = tabla_morfologica
        self._cursores = {}

    def _seleccionar_variante(self, clave, opciones):
        if not opciones:
            return ""
        idx = self._cursores.get(clave, 0) % len(opciones)
        self._cursores[clave] = idx + 1
        return opciones[idx]

    def reescribir(self, simbolo_inicial, contexto_terminales, max_pasadas=10):
        plantillas = self.tabla_estructural.get(simbolo_inicial)
        if not plantillas:
            return ""

        texto = self._seleccionar_variante(f"EST_{simbolo_inicial}", plantillas)

        for _ in range(max_pasadas):
            coincidencias = list(self.PATRON_NO_TERMINAL.finditer(texto))
            if not coincidencias:
                break

            nuevo_texto = []
            ultimo_idx = 0

            for match in coincidencias:
                nuevo_texto.append(texto[ultimo_idx:match.start()])
                simbolo = match.group(1)

                if simbolo in contexto_terminales:
                    reemplazo = str(contexto_terminales[simbolo])
                elif simbolo in self.tabla_lexica:
                    opciones = self.tabla_lexica[simbolo]
                    reemplazo = self._seleccionar_variante(f"LEX_{simbolo}", opciones)
                else:
                    reemplazo = ""

                nuevo_texto.append(reemplazo)
                ultimo_idx = match.end()

            nuevo_texto.append(texto[ultimo_idx:])
            texto = "".join(nuevo_texto)

        return self.aplicar_reglas_morfologicas(texto)

    def aplicar_reglas_morfologicas(self, texto):
        if not texto:
            return ""

        resultado = str(texto)
        for patron, reemplazo in self.tabla_morfologica:
            resultado = re.sub(patron, reemplazo, resultado)

        resultado = " ".join(resultado.split())
        if resultado:
            resultado = resultado[0].upper() + resultado[1:]
        return resultado

    def reset(self):
        self._cursores.clear()


# =====================================================================
# 3. NARRADOR DE FUTBOL (RED SEMANTICA + REESCRITURA + JSON/TOML)
# =====================================================================

class NarradorFutbol:
    """
    Generador de narraciones futbolisticas que integra:
      - RedSemanticaFutbol (ontologia, herencia de propiedades e inferencia tactica)
      - Carga de relaciones del partido en formato JSON o TOML
      - MotorReescritura (tablas estructurales, lexicas y morfologicas)
    Mantiene compatibilidad 100% directa con event_detector.py y main.py.
    """

    ARCHIVOS_PARTIDO_POR_DEFECTO = (
        "partido.json",
        "partido.toml",
        "relaciones_partido.json",
        "relaciones_partido.toml",
    )

    # ===================================================
    # 1. TABLA DE REESCRITURA ESTRUCTURAL
    # ===================================================
    TABLA_ESTRUCTURAL = {
        "POSESION_INICIAL": [
            "<SN_EQUIPO_CAP> <SV_POSESION_INICIAL> <SN_BALON> <COMP_ZONA> <MATIZ_SEMANTICO>.",
            "Con <SN_JUGADOR_ROL>, <SN_EQUIPO> <SV_POSESION_INICIAL> <SN_BALON> <COMP_ZONA>.",
            "<SN_BALON_CAP> queda bajo el mando de <SN_EQUIPO> mediante <SN_JUGADOR_ROL> <COMP_ZONA>.",
            "Arranca la jugada <COMP_ZONA>: <SN_EQUIPO> <SV_POSESION_INICIAL> <SN_BALON> <MATIZ_SEMANTICO>.",
        ],
        "PASE": [
            "<SN_PASE_CAP> de <SN_JUGADOR_ANT_ROL> <PREP_DESTINO> <SN_JUGADOR_ROL> <COMP_ZONA> <MATIZ_SEMANTICO>.",
            "<SN_JUGADOR_ANT_CAP> <SV_PASE> con <SN_JUGADOR_ROL> para <SN_EQUIPO> <COMP_ZONA>.",
            "<SN_EQUIPO_CAP> circula <SN_BALON> <MATIZ_SEMANTICO>: <SN_JUGADOR_ANT> habilita a <SN_JUGADOR_ROL> <COMP_ZONA>.",
            "Buena asociacion en <SN_EQUIPO>, <SN_JUGADOR_ANT_ROL> entrega <SN_BALON> a <SN_JUGADOR> <COMP_ZONA>.",
        ],
        "CAMBIO_POSESION": [
            "<SN_EQUIPO_CAP> <SV_RECUPERACION> <SN_BALON> por medio de <SN_JUGADOR_ROL> <MATIZ_SEMANTICO>.",
            "Cambio de posesion <COMP_ZONA>: <SN_EQUIPO> corta el avance <DE_EQUIPO_ANT> con <SN_JUGADOR_ROL>.",
            "<SN_JUGADOR_ROL_CAP> <SV_INTERCEPCION> y <SN_EQUIPO> <SV_RECUPERACION> <SN_BALON> <COMP_ZONA>.",
            "Transicion rapida <COMP_ZONA>, <SN_EQUIPO> recupera la posesion ante <SN_EQUIPO_ANT> <MATIZ_SEMANTICO>.",
        ],
        "BALON_LIBRE": [
            "<SN_BALON_CAP> <SV_DISPUTA> <COMP_ZONA> <FASE_TACTICA>.",
            "Pelota dividida <COMP_ZONA>, ningun equipo consolida el control de <SN_BALON>.",
            "Disputa abierta por <SN_BALON> <COMP_ZONA> tras la intervencion de <SN_JUGADOR_ANT_ROL>.",
            "Sin posesion definida <COMP_ZONA>: <SN_BALON> queda suelto en el terreno.",
        ],
        "ESTADO_BALON_POSESION": [
            "<SN_BALON_CAP> sigue con <SN_EQUIPO>, <MOV_BALON> <COMP_ZONA> <FASE_TACTICA>.",
            "<SN_EQUIPO_CAP> conduce <SN_BALON>, que <MOV_BALON> <COMP_ZONA>.",
        ],
        "ESTADO_BALON_LIBRE": [
            "<SN_BALON_CAP> continua en juego abierto y <MOV_BALON> <COMP_ZONA>.",
            "Sin dueno fijo, <SN_BALON> <MOV_BALON> <COMP_ZONA>.",
        ],
    }

    # ===================================================
    # 2. TABLA DE REESCRITURA LEXICA
    # ===================================================
    TABLA_LEXICA = {
        "SN_BALON": [
            "el balon",
            "el esferico",
            "la pelota",
            "el cuero",
        ],
        "SN_BALON_CAP": [
            "El balon",
            "El esferico",
            "La pelota",
        ],
        "PREF_EQUIPO": [
            "el equipo",
            "el conjunto",
            "el cuadro",
            "la escuadra",
        ],
        "PREF_JUGADOR": [
            "el jugador",
            "el dorsal",
            "el elemento",
        ],
        "SV_POSESION_INICIAL": [
            "toma la posesion de",
            "comienza con el control de",
            "asegura la tenencia de",
            "se aduena de",
        ],
        "SN_PASE_CAP": [
            "Pase",
            "Envio",
            "Servicio",
            "Conexion",
        ],
        "PREP_DESTINO": [
            "hacia",
            "para",
            "en direccion a",
            "buscando a",
        ],
        "SV_PASE": [
            "conecta",
            "combina",
            "se asocia",
            "descarga el juego",
        ],
        "SV_RECUPERACION": [
            "recupera",
            "retoma el control de",
            "reconquista",
            "captura",
        ],
        "SV_INTERCEPCION": [
            "anticipa la jugada",
            "corta la trayectoria",
            "interviene con firmeza",
        ],
        "SV_DISPUTA": [
            "queda sin una posesion clara",
            "esta en disputa",
            "permanece dividido entre ambos cuadros",
        ],
    }

    # ===================================================
    # 3. TABLA DE REESCRITURA DE DOMINIO (ZONAS 3x3 Y EQUIPOS)
    # ===================================================
    TABLA_ZONAS = {
        "Def. Izq.": [
            "en el sector defensivo izquierdo",
            "por el costado izquierdo de la defensa",
        ],
        "Defensa": [
            "en la zona defensiva central",
            "en el fondo de la zaga",
        ],
        "Def. Der.": [
            "en el sector defensivo derecho",
            "por el lateral derecho defensivo",
        ],
        "Med. Izq.": [
            "en el mediocampo izquierdo",
            "por el carril interior izquierdo",
        ],
        "Centro": [
            "en el circulo central",
            "en el centro del campo",
            "en la medular",
        ],
        "Med. Der.": [
            "en el mediocampo derecho",
            "por el carril interior derecho",
        ],
        "Ata. Izq.": [
            "en el extremo izquierdo de ataque",
            "por la banda izquierda ofensiva",
        ],
        "Ataque": [
            "en la zona frontal de ataque",
            "en el area de definicion",
        ],
        "Ata. Der.": [
            "en el extremo derecho de ataque",
            "por la banda derecha ofensiva",
        ],
        "SIN ACCION": [
            "",
        ],
    }

    TABLA_EQUIPOS = {
        EQUIPO_A: "rojo",
        EQUIPO_B: "azul",
    }

    TABLA_GENERO_EQUIPO = {
        "el equipo": "m",
        "el conjunto": "m",
        "el cuadro": "m",
        "la escuadra": "f",
    }

    TABLA_ADJETIVO_EQUIPO = {
        (EQUIPO_A, "m"): "rojo",
        (EQUIPO_A, "f"): "roja",
        (EQUIPO_B, "m"): "azul",
        (EQUIPO_B, "f"): "azul",
    }

    # ===================================================
    # 4. TABLA DE REESCRITURA MORFOLOGICA Y SANEAMIENTO
    # ===================================================
    TABLA_MORFOLOGICA = [
        (r"\bde el\b", "del"),
        (r"\ba el\b", "al"),
        (r"\bDe el\b", "Del"),
        (r"\bA el\b", "Al"),
        (r"\s+\.", "."),
        (r"\s+,", ","),
        (r"\s+:", ":"),
        (r"\.\.+", "."),
        (r"\s{2,}", " "),
    ]

    TABLA_SANEAMIENTO_ASCII = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ñ": "n", "Ñ": "N", "ü": "u", "Ü": "U",
        "¿": "", "?": "", "¡": "", "!": "",
    }

    def __init__(self, archivo_partido=None):
        self.red_semantica = RedSemanticaFutbol()
        self.motor = MotorReescritura(
            tabla_estructural=self.TABLA_ESTRUCTURAL,
            tabla_lexica=self.TABLA_LEXICA,
            tabla_morfologica=self.TABLA_MORFOLOGICA,
        )
        self.frases = self.TABLA_ESTRUCTURAL
        self.ultima_inferencia = {}

        if archivo_partido is not None:
            self.cargar_archivo_partido(archivo_partido)
        else:
            self._intentar_carga_automatica()

    # ===================================================
    # FUNCION PARA CARGAR ARCHIVO JSON O TOML DEL PARTIDO
    # ===================================================

    def cargar_archivo_partido(self, fuente, formato=None):
        """
        Recibe una ruta de archivo (.json o .toml), una cadena en formato
        JSON o TOML, o un diccionario con las relaciones del partido
        (nombres de equipos, jugadores con su numero, posiciones y relaciones)
        y lo incorpora a la Red Semantica para enriquecer la narracion.
        """
        return self.red_semantica.cargar_relaciones_partido(fuente, formato=formato)

    def _intentar_carga_automatica(self):
        """
        Si existe un archivo partido.json o partido.toml en el directorio
        actual o junto a narrator.py, lo carga automaticamente sin requerir
        cambios en main.py.
        """
        directorios = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
        for directorio in directorios:
            for nombre_archivo in self.ARCHIVOS_PARTIDO_POR_DEFECTO:
                ruta = os.path.join(directorio, nombre_archivo)
                if os.path.isfile(ruta):
                    try:
                        self.cargar_archivo_partido(ruta)
                        return
                    except Exception:
                        pass

    # ===================================================
    # NOMBRE DEL EQUIPO Y CONSTRUCCION DE SINTAGMAS
    # ===================================================

    def nombre_equipo(self, equipo):
        nodo_eq = self.red_semantica._id_nodo_equipo(equipo)
        if nodo_eq and nodo_eq in self.red_semantica.nodos:
            nombre_oficial = self.red_semantica.nodos[nodo_eq].propiedades.get("nombre_oficial")
            if nombre_oficial:
                return nombre_oficial
        return self.TABLA_EQUIPOS.get(equipo, "desconocido")

    def _construir_sn_equipo(self, equipo_id, prefijo_clave="SN_EQUIPO"):
        if equipo_id not in (EQUIPO_A, EQUIPO_B):
            return "el equipo en turno"

        prefijo = self.motor._seleccionar_variante(
            f"LEX_PREF_EQUIPO_{prefijo_clave}",
            self.TABLA_LEXICA["PREF_EQUIPO"],
        )
        genero = self.TABLA_GENERO_EQUIPO.get(prefijo, "m")
        adj = self.TABLA_ADJETIVO_EQUIPO.get((equipo_id, genero), self.TABLA_EQUIPOS.get(equipo_id, ""))
        variante_color = f"{prefijo} {adj}".strip()

        nodo_eq = self.red_semantica._id_nodo_equipo(equipo_id)
        if nodo_eq and nodo_eq in self.red_semantica.nodos:
            props = self.red_semantica.nodos[nodo_eq].propiedades
            nombre_oficial = props.get("nombre_oficial")
            apodos = props.get("apodos") or []

            variantes_equipo = []
            if nombre_oficial:
                variantes_equipo.append(nombre_oficial)
                variantes_equipo.append(f"{prefijo} de {nombre_oficial}")
            for apodo in apodos:
                if apodo:
                    variantes_equipo.append(apodo)
            variantes_equipo.append(variante_color)

            if len(variantes_equipo) > 1:
                return self.motor._seleccionar_variante(
                    f"EQ_CUSTOM_{equipo_id}_{prefijo_clave}",
                    variantes_equipo,
                )

        return variante_color

    def _construir_sn_jugador(
        self,
        jugador_id,
        prefijo_clave="SN_JUGADOR",
        rol_semantico=None,
        props_jugador=None,
        incluir_rol=False,
    ):
        props = props_jugador or {}
        nombre = props.get("nombre")
        dorsal = props.get("dorsal") if props.get("dorsal") is not None else jugador_id
        apodo = props.get("apodo")
        es_capitan = props.get("es_capitan", False)
        identificador = f"#{dorsal}" if dorsal is not None else "--"

        if nombre:
            variantes_jug = [f"{nombre} ({identificador})"]
            if incluir_rol and rol_semantico:
                variantes_jug.append(f"{rol_semantico} {nombre} ({identificador})")
            if es_capitan:
                variantes_jug.append(f"el capitan {nombre} ({identificador})")
            if apodo:
                variantes_jug.append(f"{nombre} '{apodo}' ({identificador})")
            return self.motor._seleccionar_variante(
                f"JUG_CUSTOM_{dorsal}_{prefijo_clave}",
                variantes_jug,
            )

        if incluir_rol and rol_semantico:
            return f"{rol_semantico} {identificador}"

        prefijo = self.motor._seleccionar_variante(
            f"LEX_PREF_JUGADOR_{prefijo_clave}",
            self.TABLA_LEXICA["PREF_JUGADOR"],
        )
        return f"{prefijo} {identificador}"

    def _construir_comp_zona(self, zona):
        if not zona or zona == "SIN ACCION":
            return ""
        opciones = self.TABLA_ZONAS.get(zona)
        if opciones:
            return self.motor._seleccionar_variante(f"ZONA_{zona}", opciones)
        return f"en zona {zona.lower()}"

    @staticmethod
    def _capitalizar_sintagma(texto):
        if not texto:
            return ""
        return texto[0].upper() + texto[1:]

    # ===================================================
    # GENERAR FRASE A PARTIR DE EVENTO (RED SEMANTICA + TABLAS)
    # ===================================================

    def narrar(self, evento):
        """
        1. Actualiza la Red Semantica con el evento y ejecuta inferencia por herencia.
        2. Incorpora los nombres de equipos, jugadores, dorsales y relaciones del JSON/TOML.
        3. Alimenta los descriptores inferidos en las Tablas de Reescritura.
        4. Devuelve la narracion final saneada para OpenCV y Tkinter.
        """
        if not evento:
            return ""

        tipo = evento.get("tipo")
        if not tipo or tipo not in self.TABLA_ESTRUCTURAL:
            return ""

        inferencia = self.red_semantica.registrar_evento_e_inferir(evento)
        self.ultima_inferencia = inferencia

        equipo_id = evento.get("equipo")
        equipo_ant_id = evento.get("equipo_anterior")
        jugador_id = evento.get("jugador")
        jugador_ant_id = evento.get("jugador_anterior")
        zona = evento.get("zona")

        props_jug = inferencia.get("props_jugador", {})
        props_jug_ant = inferencia.get("props_jugador_ant", {})

        sn_equipo = self._construir_sn_equipo(equipo_id, "ACTUAL")
        sn_equipo_ant = (
            self._construir_sn_equipo(equipo_ant_id, "ANTERIOR")
            if equipo_ant_id in (EQUIPO_A, EQUIPO_B)
            else "el rival"
        )
        sn_jugador = self._construir_sn_jugador(
            jugador_id,
            "ACTUAL",
            rol_semantico=inferencia.get("rol_jugador"),
            props_jugador=props_jug,
            incluir_rol=False,
        )
        sn_jugador_ant = self._construir_sn_jugador(
            jugador_ant_id,
            "ANTERIOR",
            rol_semantico=inferencia.get("rol_jugador_ant"),
            props_jugador=props_jug_ant,
            incluir_rol=False,
        )

        sn_jugador_rol = self._construir_sn_jugador(
            jugador_id,
            "ACTUAL_ROL",
            rol_semantico=inferencia.get("rol_jugador"),
            props_jugador=props_jug,
            incluir_rol=True,
        )
        sn_jugador_ant_rol = self._construir_sn_jugador(
            jugador_ant_id,
            "ANTERIOR_ROL",
            rol_semantico=inferencia.get("rol_jugador_ant"),
            props_jugador=props_jug_ant,
            incluir_rol=True,
        )
        comp_zona = self._construir_comp_zona(zona)

        contexto = {
            "SN_EQUIPO": sn_equipo,
            "SN_EQUIPO_CAP": self._capitalizar_sintagma(sn_equipo),
            "SN_EQUIPO_ANT": sn_equipo_ant,
            "DE_EQUIPO_ANT": f"de {sn_equipo_ant}",
            "SN_JUGADOR": sn_jugador,
            "SN_JUGADOR_CAP": self._capitalizar_sintagma(sn_jugador),
            "SN_JUGADOR_ANT": sn_jugador_ant,
            "SN_JUGADOR_ANT_CAP": self._capitalizar_sintagma(sn_jugador_ant),
            "SN_JUGADOR_ROL": sn_jugador_rol,
            "SN_JUGADOR_ROL_CAP": self._capitalizar_sintagma(sn_jugador_rol),
            "SN_JUGADOR_ANT_ROL": sn_jugador_ant_rol,
            "COMP_ZONA": comp_zona,
            "FASE_TACTICA": inferencia.get("fase_tactica", ""),
            "MATIZ_SEMANTICO": inferencia.get("matiz_semantico", ""),
        }

        texto = self.motor.reescribir(tipo, contexto)
        return self._sanear(texto)

    # ===================================================
    # NARRACION CONTINUA DEL BALON Y CONSULTA SEMANTICA
    # ===================================================

    def narrar_estado_balon(self, eq_posesion, movimiento, zona):
        """
        Genera la narracion continua del panel del balon consultando las
        propiedades heredadas de la zona y los nombres de equipos en la red semantica.
        """
        tipo = (
            "ESTADO_BALON_POSESION"
            if eq_posesion in (EQUIPO_A, EQUIPO_B)
            else "ESTADO_BALON_LIBRE"
        )
        props_zona = (
            self.red_semantica.heredar_propiedades(zona)
            if zona in self.red_semantica.nodos
            else {}
        )
        sn_equipo = self._construir_sn_equipo(eq_posesion, "BALON")
        comp_zona = self._construir_comp_zona(zona)

        contexto = {
            "SN_EQUIPO": sn_equipo,
            "SN_EQUIPO_CAP": self._capitalizar_sintagma(sn_equipo),
            "MOV_BALON": movimiento or "se mantiene en juego",
            "COMP_ZONA": comp_zona,
            "FASE_TACTICA": props_zona.get("fase_tactica", ""),
        }

        texto = self.motor.reescribir(tipo, contexto)
        return self._sanear(texto)

    def obtener_estado_semantico(self):
        """
        Devuelve las tripletas (origen, relacion, destino) activas y la ultima
        inferencia realizada por la red semantica para depuracion o interfaz.
        """
        return {
            "inferencia": dict(self.ultima_inferencia),
            "tripletas": self.red_semantica.obtener_tripletas(),
        }

    # ===================================================
    # LIMPIEZA Y SANEAMIENTO (TABLA ASCII / OPENCV)
    # ===================================================

    def _sanear(self, texto):
        if texto is None:
            return ""

        resultado = str(texto)
        for origen, destino in self.TABLA_SANEAMIENTO_ASCII.items():
            resultado = resultado.replace(origen, destino)

        return " ".join(resultado.split())

    def reset(self):
        """Reinicia la red semantica y los cursores del motor de reescritura."""
        self.red_semantica.reset()
        self.motor.reset()
        self.ultima_inferencia.clear()
