# App de Segmentación de Imágenes Agrícolas

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SAM](https://img.shields.io/badge/SAM-ViT--L-0064e0?style=flat-square&logo=meta&logoColor=white)](https://github.com/facebookresearch/segment-anything)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![CPU Ready](https://img.shields.io/badge/Hardware-CPU%20%7C%20CUDA-f59e0b?style=flat-square&logo=nvidia&logoColor=white)](https://pytorch.org/)
[![GeoTIFF](https://img.shields.io/badge/Format-GeoTIFF%20%7C%20PNG-6366f1?style=flat-square&logo=qgis&logoColor=white)](https://rasterio.readthedocs.io/)
[![Rasterio](https://img.shields.io/badge/Rasterio-1.5-16a34a?style=flat-square)](https://rasterio.readthedocs.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000?style=flat-square)](https://github.com/psf/black)

Aplicación web interactiva para segmentación de imágenes agrícolas y satelitales con 8 métodos, incluyendo PyTorch + SAM (Segment Anything Model).

> ⏬ **Primera carga en Streamlit Cloud:** la app descarga automáticamente el checkpoint SAM ViT-B (~375 MB) desde los servidores de Meta la primera vez que se inicia (1–2 minutos). Los **métodos clásicos 1–7 funcionan inmediatamente** sin necesidad del modelo. Una vez descargado, el modelo queda en caché y las cargas siguientes son instantáneas.

> ⚠️ **Memoria en Streamlit Community Cloud:** el límite por app es de ~2.7 GB de RAM. Se usa ViT-B por defecto porque ViT-L (~1.2 GB de pesos) suele provocar el reinicio del contenedor (la app queda en “Connecting…”). Para cambiar de variante, definir `sam_model_type = "vit_l"` en los secrets de la app o la variable de entorno `SAM_MODEL_TYPE` en local.

## Características

- **7 métodos clásicos**: Otsu, Canny, Region Growing, Watershed, K-Means, Mean-Shift, GrabCut
- **SAM (ViT-B por defecto; ViT-L / ViT-H configurables)**: Segment Anything Model de Meta con:
  - Modo automático: segmenta toda la imagen
  - Modo por clic: el usuario indica qué segmentar
- Soporte para imágenes JPG, PNG, TIF/GeoTIFF
- Exportación PNG y GeoTIFF (conservando georreferenciación)
- Comparación visual de múltiples métodos

## Requisitos del sistema

- Python 3.10 o superior
- Windows / Linux / macOS
- CPU (sin GPU requerida); CUDA se usa automáticamente si está disponible

## Instalación

### 1. Instalar dependencias base

```bash
pip install -r requirements.txt
```

### 2. Instalar SAM (Segment Anything)

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 3. Checkpoint SAM

Se descarga automáticamente al iniciar la app (tanto en local como en Streamlit Cloud) desde
`https://dl.fbaipublicfiles.com/segment_anything/`. No se requiere ninguna acción manual.

Si se prefiere colocarlo a mano, guardarlo en la carpeta `models/`:
```
models/
└── sam_vit_b_01ec64.pth
```

Variante alternativa (más precisa, mucho más pesada):
```bash
SAM_MODEL_TYPE=vit_l streamlit run app.py
```

> **Nota:** Los métodos clásicos (1-7) funcionan sin necesidad del checkpoint SAM.

## Ejecutar la app

Desde la carpeta `app/`:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en el navegador en `http://localhost:8501`

## Estructura del proyecto

```
app/
├── app.py                  # Punto de entrada Streamlit
├── core/
│   ├── segmentation.py     # 7 métodos clásicos de segmentación
│   ├── checkpoint.py       # Descarga/cache del checkpoint SAM
│   └── sam_handler.py      # SAM (auto + clic)
├── utils/
│   ├── image_io.py         # Carga/guardado PNG y GeoTIFF
│   └── visualization.py    # Overlay de máscaras y comparaciones
├── models/                 # Checkpoint SAM (.pth) - no incluido en git
├── requirements.txt
└── README.md
```

## Uso

1. **Cargar imagen**: usa el panel lateral para subir una imagen (JPG/PNG/TIF)
2. **Seleccionar método**: elige entre los 8 métodos disponibles
3. **Ajustar parámetros**: cada método tiene parámetros configurables
4. **Ver resultado**: se muestra la imagen original y el resultado segmentado
5. **Exportar**: descarga el resultado como PNG o GeoTIFF

### SAM modo automático
- Selecciona "SAM - Automático" en el selector de métodos
- Haz clic en "Segmentar todo"
- SAM detectará y coloreará todas las regiones automáticamente

### SAM modo por clic
- Selecciona "SAM - Por clic" en el selector de métodos
- Haz clic sobre la imagen para marcar puntos de interés
- Los puntos verdes = objeto a segmentar
- Los puntos rojos = fondo a excluir
- Haz clic en "Segmentar selección"

## Imágenes de ejemplo compatibles

- Imágenes agrícolas multi-banda (GeoTIFF)
- Imágenes satelitales RGB
- Fotografías de cultivos, parcelas, campos
- Formatos: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

## Créditos

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) - Meta AI Research
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)
- [Rasterio](https://rasterio.readthedocs.io/)

---
ABC Geomática Agrícola SRL - 2026
