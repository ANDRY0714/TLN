# Cambios v7: panel de eventos, colores de equipo y voz del narrador

Este documento describe los cambios hechos sobre la versión anterior del
proyecto (commit `3f4bf12`, *"Filtro de campo, panel de eventos con pestañas y
detección de gol/celebración"*). Para cada cambio se explica cómo funcionaba
antes, qué se cambió, por qué y cómo se usa en el programa.

Los cambios responden a tres problemas:

1. **Panel de eventos poco legible**: el profesor observó que los eventos no se
   apreciaban; la letra del panel lateral era muy pequeña.
2. **Equipos mal coloreados**: jugadores del mismo equipo salían con colores
   distintos, y España aparecía en azul y Argentina en rojo (al revés de lo
   lógico). Además, la calibración de colores era muy sensible a la luz.
3. **Sin voz**: el sistema es un narrador, pero los eventos solo se mostraban
   como texto.

---

## Resumen de archivos

| Archivo | Estado | Qué cambió |
|---|---|---|
| `main.py` | Modificado (v6 → v7) | Panel lateral más ancho, con scroll y letra grande. Lee los colores de camiseta del toml. Integra la voz y la casilla "Voz". Muestra nombres de equipo en el panel táctico. |
| `tracker_engine.py` | Modificado (v5 → v6) | Nuevo clasificador de equipos `ClasificadorEquipo`, con corrección de luz, histogramas, prototipos y recalibración continua. Votación por ventana en `Track`. Detección de jugadores superpuestos. |
| `partido.toml` | Modificado | Nueva clave `camiseta` en cada equipo. |
| `voz.py` | **Nuevo** | Texto a voz con SAPI de Windows y cola con prioridades. |
| `CAMBIOS.md` | **Nuevo** | Este documento. |

No se agregaron dependencias: la voz usa el sintetizador que ya trae Windows.

---

## 1. Panel de eventos (`main.py`)

### Antes
- Panel lateral fijo de **400 px**, sin scroll. Si el contenido no cabía, se
  cortaba.
- Tamaños de letra entre **7 y 11 pt**: títulos de sección a 8 pt, señales a
  7 pt, texto del evento a 10 pt e historial a 9 pt.
- Orden: marcador → estado del plano → medidor de celebración → balón → último
  evento → historial. El evento más importante quedaba a media pantalla.

### Ahora
- Panel de **580 px** (`PANEL_W`) dentro de un `Canvas` con barra de
  desplazamiento vertical. La rueda del ratón desplaza el panel cuando el
  puntero está encima. Sobre el historial, la rueda mueve solo el historial,
  que conserva su propio scroll.
- Toda la tipografía está en constantes al inicio de `main.py`, para ajustarla
  en un solo lugar:

| Elemento | Antes | Ahora | Constante |
|---|---|---|---|
| Título "EVENTOS DEL PARTIDO" | 14 pt | 20 pt | `F_TITULO` |
| Marcador | 15 pt | 24 pt | `F_MARCADOR` |
| Títulos de sección | 8 pt | 13 pt | `F_SECCION` |
| Plano / estado | 9 pt | 13 pt | `F_ESTADO` |
| Señales de celebración | 7 pt | 11 pt | `F_SENALES` |
| Narración del balón | 10 pt | 16 pt | `F_TEXTO` |
| Título del último evento | 11 pt | 20 pt | `F_EVENTO_TIT` |
| Texto del último evento | 10 pt | 17 pt | `F_EVENTO_TXT` |
| Historial | 9 pt | 14 pt | `F_HIST` (+ `F_HIST_TIPO`, `F_HIST_T`, `F_HIST_MOTIVO`) |

- **Nuevo orden**: marcador → **último evento** → balón → estado (plano, tiempo,
  medidor de celebración, señales, estado de la voz) → historial.
- La tarjeta del último evento tiene un **borde del color del tipo de evento**
  (verde pase, azul cambio de posesión, naranja gol, rosa celebración), para
  que se note cuando llega un evento nuevo. GOL y CELEBRACIÓN siguen cambiando
  el fondo de la tarjeta durante 4 s, como antes.
- El medidor de celebración pasó de 10 px a 18 px de alto.
- Nueva línea `lbl_voz` con el estado de la voz, por ejemplo
  `voz: Microsoft Sabina Desktop - hablando`.
- Al cambiar de pestaña (Eventos / Análisis táctico), el scroll vuelve arriba.

### Panel de análisis táctico
- Ahora muestra los nombres de los equipos (`España`, `Argentina`) en lugar de
  "Equipo A / Equipo B" y "Eq. ROJO / Eq. AZUL". El nombre se toma del
  narrador (`NarradorFutbol.nombre_equipo`), que a su vez lo lee de
  `partido.toml`. Se usa el helper `DetectorFutbol._nombre(equipo)`.

---

## 2. Clasificación de equipos por color (`tracker_engine.py`)

Es el cambio más grande. Afecta a `ClasificadorEquipo`, a `Track` y a
`MotorTracking.procesar_frame`.

### Cómo funcionaba antes
1. Durante los primeros **30 frames** (`WARMUP_FRAMES`) se recortaba el torso
   de cada jugador (15 %–50 % de la altura de la caja), se quitaban los
   píxeles verdes y se calculaba el **color Lab promedio** del recorte.
2. Con esos promedios se hacía **k-means con k = 2**. El cluster 0 pasaba a
   ser `EQUIPO_A` y el 1, `EQUIPO_B`.
3. La calibración se hacía **una sola vez** y quedaba congelada.
4. Cada frame, cada jugador votaba por el centroide más cercano y los votos se
   **acumulaban para siempre** (`votos_a`, `votos_b`).

### Problemas que causaba

| Problema | Causa | Efecto visible |
|---|---|---|
| España en azul y Argentina en rojo | El orden de los clusters del k-means es aleatorio | Los colores de los equipos salían invertidos según la ejecución |
| Camiseta a rayas mal representada | El promedio de las rayas blancas y celestes es un color "intermedio" que no existe en la camiseta | Argentina se confundía con otros colores claros |
| Sensible a la luz | La L (brillo) de Lab pesaba igual que el color | Un jugador al pasar a la sombra "cambiaba de equipo" |
| Calibración pobre | Solo 30 frames: en `final.mp4` casi solo aparecían 4 argentinos y el portero | El segundo cluster se lo quedaba el portero verde o un recorte con dos jugadores encimados |
| Errores pegados | Votos acumulados para siempre; si ByteTrack intercambia el id de dos jugadores, el error no se corrige | Jugadores con el color del rival durante todo el video |

### Cómo funciona ahora

Proceso completo por frame:

```
frame ─► FiltroCampo (máscara de césped)
      ─► YOLO + ByteTrack (cajas de personas)
      ─► plano (general / medio / primer plano / fuera)
      ─► clf.nuevo_frame(frame, mask_cesped, calibrar = plano general o medio)
      ─► cajas_solapadas(cajas)            # qué jugadores están tapados
      ─► para cada jugador:
             eq = clf.observar(frame, caja, solapado)   # A / B / NA
             track.votar(eq)                            # ventana con histéresis
```

#### 2.1 Corrección local de iluminación
- Mientras no hay calibración, `nuevo_frame` aprende el **brillo de
  referencia del césped** (mediana del canal V de HSV dentro de la máscara de
  césped, suavizada).
- Para cada jugador, `_ganancia_local` mide el brillo del césped **alrededor de
  sus pies** (una franja de 1.5 veces el ancho de la caja, en la parte baja).
- Los píxeles de la camiseta se multiplican por
  `ganancia = brillo_referencia / brillo_local`, limitada a [0.6, 1.8].

Así, un jugador en sombra se "ilumina" hasta el nivel de referencia antes de
medir su color. Es un modelo de iluminación diagonal sencillo, pero local:
compensa sombras dentro del mismo frame, no solo cambios de exposición de la
cámara.

#### 2.2 Descriptor de color: histograma en lugar de promedio
`_sample` devuelve, para cada recorte:
- Un **histograma 2D de cromaticidad** (canales a, b de Lab): 8×8 bins en el
  rango [80, 176], suavizado con un filtro gaussiano 3×3 (con pocos píxeles el
  histograma sale "ruidoso").
- Un **histograma de brillo L** de 4 bins con peso `PESO_L = 0.5` (el brillo
  importa menos que el color).
- Se toma la **raíz cuadrada** de los histogramas normalizados. Con eso, la
  distancia euclidiana entre descriptores equivale a la **distancia de
  Hellinger**, una medida estándar para comparar distribuciones.

Una camiseta a rayas queda como un histograma con **dos picos** (blanco y
celeste), en lugar de un promedio que no se parece a nada.

Recorte usado: 15 %–50 % de la altura y 20 %–80 % del ancho de la caja (para
evitar fondo y brazos). Se descartan píxeles de césped y sombras muy oscuras
(V ≤ 25).

#### 2.3 Muestras limpias
- **Jugadores superpuestos** (`cajas_solapadas`): si otra caja tapa más del
  15 % del área de un jugador, ese jugador no aporta muestra ni vota en ese
  frame. El recorte mezclaría dos camisetas.
- **Submuestreo**: solo se guardan muestras 1 de cada 3 frames
  (`SUBMUESTREO`), para no llenar la memoria con el mismo jugador repetido.
- Solo se guardan muestras en planos **general y medio**, no en primeros planos
  ni tomas de la grada.
- Se guardan como máximo las últimas 800 muestras (`MAX_MUESTRAS`).

#### 2.4 Colores de referencia desde `partido.toml`
Cada equipo puede declarar su camiseta:

```toml
[equipos.local]
id = 1
nombre = "España"
camiseta = ["rojo"]

[equipos.visitante]
id = 2
nombre = "Argentina"
camiseta = ["blanco", "celeste"]
```

- Se aceptan **nombres** (`rojo`, `granate`, `blanco`, `negro`, `gris`, `azul`,
  `azul marino`, `celeste`, `verde`, `amarillo`, `naranja`, `morado`, `rosa`;
  tabla `COLORES_NOMBRE`) o **hexadecimal** (`"#C8102E"`). La conversión la
  hace `color_a_bgr`.
- Si hay varios colores, la referencia es su promedio en Lab (el blanco y el
  celeste dan el tono "medio" de las rayas).
- `main.py` lee esta clave con `DetectorFutbol._colores_camiseta()`. Usa los
  datos que el narrador ya cargó del toml y resuelve `local`/`visitante`/`id`
  con `RedSemanticaFutbol._resolver_equipo_id`. El resultado se pasa a
  `MotorTracking(..., colores_equipo=...)`.
- Con esto **`EQUIPO_A` es siempre el equipo con `id = 1` (local, España)** y
  se dibuja en rojo, y `EQUIPO_B` (Argentina) en azul.

#### 2.5 Prototipos por equipo y recalibración continua (`_fit`)
Con colores de referencia:
1. Se hace **k-means con k entre 2 y 5** (según la cantidad de muestras) sobre
   los descriptores.
2. De cada cluster se descarta el 10 % de muestras más lejanas (atípicas) y se
   calcula:
   - su centro;
   - su color Lab medio;
   - un umbral de distancia máxima (percentil 95 × 1.6).
3. Cada cluster se compara (en Lab, con el brillo a la mitad de peso) con los
   colores de referencia. Se asigna a un equipo solo si:
   - está a menos de `DIST_REF_MAX = 55` del color de ese equipo, **y**
   - está claramente más cerca de ese equipo que del otro (menos del 70 % de
     la distancia).

   Los clusters que no cumplen (porteros, árbitro, recortes malos) **se
   ignoran**.
4. Un equipo puede quedarse con **varios clusters** ("prototipos"): por
   ejemplo, Argentina con rayas más blancas, rayas más celestes y en sombra.
5. Esto se repite **cada 60 frames** (cada 30 mientras falte un equipo) con las
   últimas 800 muestras. Así, los prototipos siguen los cambios de luz y de
   cámara a lo largo del video. Si un equipo no aparece en esa ventana,
   conserva sus prototipos anteriores.
6. Si al principio del video solo aparece un equipo, se clasifica solo ese
   (con un umbral más estricto) y el otro se agrega cuando aparece.

Sin colores en el toml, se mantiene el comportamiento clásico (k = 2), pero en
cada recalibración los clusters se emparejan con los prototipos anteriores
para que A y B no se intercambien.

#### 2.6 Clasificación con margen (`observar`)
- Se busca el prototipo más cercano y el prototipo más cercano del **otro**
  equipo.
- `margen = (d_otro − d_cerca) / (d_otro + d_cerca)`.
- Si `margen < 0.08` (`MARGEN_MIN`), es decir, el jugador está a medio camino
  entre los dos equipos, o si está más lejos que el umbral de su prototipo, el
  resultado es `EQUIPO_NA` y **no vota**.

#### 2.7 Votación por ventana con histéresis (`Track.votar`)
- Antes: contadores `votos_a` / `votos_b` acumulados para siempre.
- Ahora: `deque` con los **últimos 25 votos** (`VOTOS_VENTANA`). Los votos
  `EQUIPO_NA` no entran.
- El primer equipo se asigna con 3 votos (`VOTOS_MIN`).
- Para **cambiar** de equipo, el otro debe tener más del **65 %** de la ventana
  (`VOTOS_CAMBIO`). Esto evita parpadeos, pero permite corregir un error de
  ByteTrack en pocos segundos.

#### 2.8 Cambios de API (para quien toque el código)

| Antes | Ahora |
|---|---|
| `ClasificadorEquipo()` | `ClasificadorEquipo(colores_ref=None)` |
| `clf.warm_up(frame, dets)` (solo antes de calibrar) | `clf.nuevo_frame(frame, mask_cesped, calibrar)` una vez por frame |
| `clf.clasificar(frame, x1, y1, x2, y2)` | `clf.observar(frame, x1, y1, x2, y2, solapado)`: guarda la muestra y clasifica |
| `clf.centroids_lab`, `clf.centroids_bgr` | `clf.protos`, `protos_eq`, `protos_umbral`, `protos_lab`, `protos_n` |
| — | `clf.completo()`: ya hay prototipos de los dos equipos |
| `Track.votos_a`, `Track.votos_b`, `equipo_estable()` | `Track.votos` (deque) y `Track.votar(eq)` |
| `MotorTracking(w, h, modelo)` | `MotorTracking(w, h, modelo, colores_equipo=None)` |

`clf.listo()`, `clf.centroid_bgr(equipo)` y las claves de `estado`
(`warmup`, `color_a_bgr`, `color_b_bgr`, …) se mantienen, así que
`main.py`, `event_detector.py`, `jersey_recognizer.py` y
`momentos_clave.py` no necesitaron cambios por esto.

#### 2.9 Parámetros ajustables

| Constante | Valor | Dónde | Para qué |
|---|---|---|---|
| `BINS_AB`, `RANGO_AB` | 8, (80, 176) | `ClasificadorEquipo` | Resolución del histograma de color |
| `PESO_L` | 0.5 | `ClasificadorEquipo` | Peso del brillo frente al color |
| `MARGEN_MIN` | 0.08 | `ClasificadorEquipo` | Duda mínima para abstenerse |
| `MIN_MUESTRAS` | 40 | `ClasificadorEquipo` | Muestras antes del primer ajuste |
| `MAX_MUESTRAS` | 800 | `ClasificadorEquipo` | Memoria de la recalibración |
| `SUBMUESTREO` | 3 | `ClasificadorEquipo` | Guardar muestras 1 de cada N frames |
| `MIN_CLUSTER` | 6 | `ClasificadorEquipo` | Tamaño mínimo de un prototipo |
| `DIST_REF_MAX` | 55 | `ClasificadorEquipo` | Parecido mínimo con el color del toml |
| `REFIT_CADA` | 60 | `ClasificadorEquipo` | Frames entre recalibraciones |
| `VOTOS_VENTANA` | 25 | módulo | Votos recientes por jugador |
| `VOTOS_MIN` | 3 | módulo | Votos para la primera asignación |
| `VOTOS_CAMBIO` | 0.65 | módulo | Mayoría necesaria para cambiar de equipo |

### Resultados medidos en `final.mp4`

| Medida | Antes | Ahora |
|---|---|---|
| Color asignado a España | Aleatorio (rojo o azul) | Siempre rojo (`EQUIPO_A`) |
| Cambios de equipo de un mismo jugador (primeros 400 frames) | 17 | 1 |
| Margen típico entre equipos (mediana) | 0.08 | 0.47 |
| Comportamiento en sombra / primer plano | Confusión frecuente | Correcto en los frames revisados |

En los frames revisados a lo largo de todo el video, España aparece en rojo y
Argentina en azul, también en la zona de sombra y en la celebración del gol.

---

## 3. Voz del narrador (`voz.py`, nuevo)

### Antes
No había voz: los eventos solo se mostraban en el panel.

### Ahora
Cada evento que se escribe en el panel también se **dice en voz alta**.

#### 3.1 Motor de voz
- Usa **SAPI de Windows** (`System.Speech.Synthesis.SpeechSynthesizer`) a través
  de un proceso de **PowerShell** que queda abierto y recibe las frases por su
  entrada estándar. No hay que instalar nada.
- Elige automáticamente la primera voz en español instalada. En las PC de
  prueba es **Microsoft Sabina Desktop (es-MX)**.
- Protocolo: cada frase se envía como una línea `velocidad|volumen|texto_en_base64`.
  El base64 evita problemas de codificación con ñ y tildes. PowerShell responde
  `OK` al terminar de hablar.
- El script de PowerShell se escribe en un archivo temporal
  (`%TEMP%\tln_voz_narrador.ps1`) y se ejecuta con `-File`.
  **Detalle importante**: con `-Command` o `-EncodedCommand`, el propio
  PowerShell lee el stdin redirigido y "se come" las frases, y la voz falla de
  forma intermitente. Con `-File` esto no pasa.
- Todo corre en un **hilo aparte**, así que la interfaz y el video no se
  bloquean mientras se habla.

#### 3.2 Prioridades y cola
Los eventos llegan más rápido de lo que se pueden pronunciar. Por eso no hay
cola larga: se guarda **una sola frase pendiente**, la más reciente.

| Prioridad | Constante | Qué se dice | Velocidad SAPI | Volumen |
|---|---|---|---|---|
| 0 | `RELLENO` | Estado del balón y `BALON_LIBRE` | 1 | 80 |
| 1 | `EVENTO` | Pases, cambios de posesión, posesión inicial | 2 | 100 |
| 2 | `URGENTE` | `GOL`, `CELEBRACION` | 4 | 100 |

- Una frase pendiente solo se reemplaza por otra de prioridad igual o mayor.
- Un evento **URGENTE corta lo que se esté diciendo**. Como SAPI no se puede
  interrumpir por stdin, se termina el proceso y se vuelve a crear para la
  siguiente frase (en las pruebas, el gol se escuchó 1.7 s después).
- El **relleno** (narración del balón) solo se dice si la voz lleva más de 4 s
  callada (`SILENCIO_ANTES_DE_RELLENO_S` en `main.py`). Así hay narración
  continua sin tapar los eventos.
- La relación tipo de evento → prioridad está en `PRIORIDAD_VOZ` (`main.py`).

#### 3.3 Limpieza del texto para la voz (`limpiar_para_voz`)
El texto del panel se adapta antes de enviarlo a la voz:

| Texto del panel | Texto hablado |
|---|---|
| `Pau Cubarsi (#22)` | `Pau Cubarsí, dorsal 22` |
| `#7` | `número 7` |
| `(marcador 1-0)` | `, marcador 1 a 0` |
| `10.5km/h` | `10.5 kilómetros por hora` |
| `Espana`, `balon`, `posesion`… | `España`, `balón`, `posesión`… |

El narrador escribe sin tildes. La tabla `_TILDES` corrige las palabras y
apellidos más frecuentes para que la voz los pronuncie bien. Si aparecen otras
palabras mal pronunciadas, basta con agregarlas ahí.

#### 3.4 Integración en `main.py`
- `self.voz = VozNarrador()` se crea al iniciar la app.
- `registrar_evento` llama a `self.voz.decir(texto, prioridad)`.
- `actualizar_narracion_balon` llama a `self.voz.decir(texto, RELLENO)` si la
  voz está libre.
- Casilla **"Voz"** en la barra superior: activa o silencia la voz
  (`VozNarrador.set_activa`).
- La voz **se calla** al pausar, al reiniciar y al abrir otro video.
- Al cerrar la ventana (`WM_DELETE_WINDOW` → `DetectorFutbol.cerrar`) se
  termina el proceso de PowerShell.
- Si no es Windows o el sintetizador falla, la app sigue funcionando sin voz y
  el panel lo indica (`voz: no disponible` / `voz: error (...)`).

#### 3.5 Prueba rápida de la voz
```bash
python voz.py
```
Dice una frase de ejemplo, la interrumpe con un "gol" e imprime el estado.

---

## 4. Cómo usar el programa

1. `python main.py`
2. **Abrir Video**: durante los primeros segundos la barra superior muestra
   "Calibrando colores…" (los jugadores salen en gris hasta tener la primera
   calibración).
3. Los eventos aparecen en el panel y se narran en voz alta. Usa la barra de
   scroll o la rueda del ratón para ver el panel completo.
4. Casillas de la barra superior:
   - **Ver campo**: dibuja el contorno del césped detectado y la red (depuración).
   - **Voz**: activa o silencia la narración hablada.
5. **Para otro partido**: cambiar en `partido.toml` los nombres, jugadores y la
   clave `camiseta` de cada equipo. Si se quita `camiseta`, la clasificación
   vuelve a separar en dos grupos de color sin saber cuál es cada equipo.

---

## 5. Limitaciones conocidas

- En `final.mp4`, los jugadores de España salen en **gris durante los primeros
  ~5.6 s**, porque casi no aparecen al inicio y todavía no hay suficientes
  muestras de su color.
- Puede haber errores aislados con colores parecidos al del rival, por ejemplo
  el portero de verde marcado en rojo en un frame.
- La voz SAPI suena **robótica**. Es suficiente para empezar a escuchar la
  narración, pero no es una voz natural.
- La voz depende de Windows. En otro sistema operativo la app funciona sin voz.

---

## 6. Trabajo futuro: voz más natural

La interfaz de `voz.py` (cola + prioridades) es independiente del motor, así
que se puede cambiar SAPI por un motor neuronal sin tocar `main.py`:

| Opción | Tipo | Ventajas | Inconvenientes |
|---|---|---|---|
| **Piper** | Local, CPU | Rápido, gratis, voces en es-MX / es-ES | Expresividad limitada |
| **XTTS-v2 (Coqui)** | Local, GPU | Muy expresivo, clona una voz con unos segundos de audio | Más lento, licencia no comercial |
| **edge-tts** | En línea | Voces neuronales de Microsoft (p. ej. `es-MX-JorgeNeural`) | No es un servicio oficial |
| **Azure / Google / ElevenLabs / OpenAI TTS** | Nube, de pago | Máxima calidad, SSML / estilos emocionales | Requiere cuenta e internet |

Pasos sugeridos:
1. Definir una interfaz de motor `sintetizar(texto, estilo) → audio`.
2. Reproducir el audio con `sounddevice` o `pygame`. Así interrumpir sería
   detener la reproducción, sin terminar procesos.
3. Usar **SSML** o estilos por tipo de evento: euforia en el gol, tono calmado
   en los pases.
4. Generar frases cortas, pensadas para escucharse y no para leerse, y usar
   caché para las frases frecuentes, para reducir la latencia.
5. Opcional: sonido ambiente de estadio que sube de volumen en los goles.
