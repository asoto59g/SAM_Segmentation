# App de Segmentación de Imágenes Agrícolas

Aplicación web interactiva para segmentación de imágenes agrícolas y satelitales con 8 métodos, incluyendo PyTorch + SAM (Segment Anything Model).

## Características

- **7 métodos clásicos**: Otsu, Canny, Region Growing, Watershed, K-Means, Mean-Shift, GrabCut
- **SAM ViT-L**: Segment Anything Model de Meta con:
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

### 3. Descargar checkpoint SAM ViT-L

Descargar el archivo `sam_vit_l_0b3195.pth` (~1.2 GB) desde:
https://github.com/facebookresearch/segment-anything#model-checkpoints

Colocar el archivo en la carpeta `models/`:
```
app/
└── models/
    └── sam_vit_l_0b3195.pth
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
│   └── sam_handler.py      # SAM ViT-L (auto + clic)
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
