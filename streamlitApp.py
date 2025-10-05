# streamlit_app.py
# App única con 11 mini-demos basados en "OpenCV 3.x with Python By Example"
# Autor: adaptado para Imanol - Demo para clase
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =============================
# ESTILO Y DISEÑO GENERAL
# =============================
st.set_page_config(
    page_title="OpenCV by Example — 11 Demos",
    page_icon="📘",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
/* Fondo degradado general */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f8f9fa, #e3f2fd);
}

/* Texto del cuerpo principal */
html, body, [data-testid="stMarkdownContainer"], .stMarkdown {
    color: #0a0a0a !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Títulos principales */
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    color: #0d47a1;
}

/* Barra lateral */
[data-testid="stSidebar"] {
    background-color: #1565c0;
}

/* Texto blanco solo dentro de la barra lateral */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] div {
    color: white !important;
}

/* Botones */
div.stButton > button {
    background-color: #0d47a1;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1em;
    font-weight: 600;
    border: none;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #1976d2;
    transform: scale(1.03);
}

/* Sliders, selectbox y file uploader */
.stSlider label, .stSelectbox label, .stFileUploader label {
    font-weight: bold;
    color: #0d47a1 !important;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.9em;
    color: #444;
    margin-top: 3em;
    padding: 1em;
}
</style>
""", unsafe_allow_html=True)

st.title("📘 OpenCV 3.x with Python — 11 Programas Interactivos")
st.markdown("---")


# -----------------------
# Helpers
# -----------------------
def read_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def cv2_to_pil(img_cv):
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def pil_to_cv2(img_pil):
    img = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def find_contours_from_gray(gray):
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# -----------------------
# Sidebar - Selección de capítulo
# -----------------------
st.sidebar.title("OpenCV 3.x - Selector de capítulos (11)")
chapter = st.sidebar.selectbox("Selecciona capítulo", [
    "1 - Geometric Transformations",
    "2 - Edges & Filters",
    "3 - Cartoonizing (webcam/photo)",
    "4 - Body Parts (Haar Cascades)",
    "5 - Feature Extraction",
    "6 - Seam Carving (demo simplificado)",
    "7 - Detecting Shapes & Approximating Contours",
    "8 - Object Tracking (color-based)",
    "9 - Object Recognition (feature matching)",
    "10 - Augmented Reality (homography overlay)",
    "11 - Machine Learning (ANN) - simple demo"
])

st.markdown("Selecciona un capítulo en la barra lateral. Cada demo muestra una implementación corta y explicativa de los temas del libro.")
chapter_number = int(chapter.split(" - ")[0])
# -----------------------
# CAPÍTULO 1 - Geometric Transformations
# -----------------------
if chapter_number == 1:
    # ==== ESTILO VISUAL ====
    st.markdown("""
    <style>
    .title-geom {
        font-size: 2em;
        font-weight: 800;
        color: #1565c0;
        text-align: center;
        margin-bottom: 0.3em;
    }
    .subtitle-geom {
        text-align: center;
        color: #444;
        font-size: 1.05em;
        margin-bottom: 1.5em;
    }
    /* Bordes suaves en imágenes */
    img {
        border-radius: 14px !important;
        box-shadow: 0 4px 20px rgba(21, 101, 192, 0.2);
        transition: transform 0.2s ease-in-out;
    }
    img:hover {
        transform: scale(1.01);
    }
    /* Sliders y number inputs */
    .stSlider label, .stNumberInput label {
        color: #1565c0 !important;
        font-weight: 600 !important;
    }
    /* Botones */
    div.stButton > button {
        background-color: #1565c0 !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: 0.3s;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #1976d2 !important;
        transform: scale(1.05);
    }
    </style>

    <div class="title-geom">📐 Capítulo 1 — Transformaciones Geométricas</div>
    <div class="subtitle-geom">
        Sube una imagen y aplica transformaciones geométricas como <b>rotación</b>, <b>escala</b>, <b>traslación</b> y <b>perspectiva</b>.<br>
        Observa en tiempo real cómo cambia la forma de la imagen.
    </div>
    """, unsafe_allow_html=True)

    # ==== CONTENIDO ====
    uploaded = st.file_uploader("📤 Sube una imagen (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = read_image(uploaded)
        st.image(cv2_to_pil(img), caption="🖼 Imagen original", use_container_width=True)

        st.subheader("🔄 Rotación / Escala / Traslación")
        angle = st.slider("Rotar (°)", -180, 180, 0)
        scale = st.slider("Escala", 0.1, 3.0, 1.0, step=0.1)
        tx = st.slider("Trasladar X (px)", -200, 200, 0)
        ty = st.slider("Trasladar Y (px)", -200, 200, 0)

        h, w = img.shape[:2]
        Mrot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
        rotated = cv2.warpAffine(img, Mrot, (w, h))
        Mtrans = np.float32([[1, 0, tx], [0, 1, ty]])
        trans = cv2.warpAffine(rotated, Mtrans, (w, h))
        st.image(
            cv2_to_pil(trans),
            caption=f"Rotado {angle}°, escala {scale}, trasladado ({tx},{ty})",
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("📈 Transformación Afín")
        st.markdown("""
        Este tipo de transformación mantiene las líneas paralelas y permite distorsionar la imagen
        según tres puntos de referencia.  
        *(En este ejemplo, modificas manualmente los puntos de destino para visualizar el efecto)*.
        """)

        # puntos fijos para demo
        src = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
        col1, col2, col3 = st.columns(3)
        with col1:
            dst_x1 = st.number_input("x₁", value=0)
            dst_y1 = st.number_input("y₁", value=0)
        with col2:
            dst_x2 = st.number_input("x₂", value=w - 1)
            dst_y2 = st.number_input("y₂", value=10)
        with col3:
            dst_x3 = st.number_input("x₃", value=10)
            dst_y3 = st.number_input("y₃", value=h - 1)

        dst = np.float32([[dst_x1, dst_y1], [dst_x2, dst_y2], [dst_x3, dst_y3]])
        M_affine = cv2.getAffineTransform(src, dst)
        warped_affine = cv2.warpAffine(img, M_affine, (w, h))
        st.image(cv2_to_pil(warped_affine), caption="Resultado: Transformación Afín", use_container_width=True)

        st.markdown("---")
        st.subheader("🧭 Transformación de Perspectiva (Homografía)")
        st.markdown("""
        Se aplica `warpPerspective()` para modificar la perspectiva simulando un cambio de ángulo de visión.  
        Ideal para efectos tipo “vista 3D” o corrección de proyecciones.
        """)

        pts1 = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        pts2 = np.float32([[0, 0], [w - 1, 0], [int(0.6 * w), h - 1], [int(0.4 * w), h - 1]])
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        warped_persp = cv2.warpPerspective(img, M_persp, (w, h))
        st.image(cv2_to_pil(warped_persp), caption="Transformación de Perspectiva", use_container_width=True)

# -----------------------
# CAPÍTULO 2 - Edges & Filters
# -----------------------
# CAPÍTULO 2 - Edges & Filters
# -----------------------
elif chapter_number == 2:
    # ==== ESTILO VISUAL ====
    st.markdown("""
    <style>
    .title-edges {
        font-size: 2em;
        font-weight: 800;
        color: #1565c0;
        text-align: center;
        margin-bottom: 0.3em;
    }
    .subtitle-edges {
        text-align: center;
        color: #444;
        font-size: 1.05em;
        margin-bottom: 1.5em;
    }
    img {
        border-radius: 14px !important;
        box-shadow: 0 4px 18px rgba(21, 101, 192, 0.2);
        transition: transform 0.2s ease-in-out;
    }
    img:hover {
        transform: scale(1.01);
    }
    .stSlider label {
        color: #1565c0 !important;
        font-weight: 600 !important;
    }
    div.stButton > button {
        background-color: #1565c0 !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: 0.3s;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #1976d2 !important;
        transform: scale(1.05);
    }
    </style>

    <div class="title-edges">✨ Capítulo 2 — Filtros y Detección de Bordes</div>
    <div class="subtitle-edges">
        Experimenta con diferentes <b>filtros de suavizado</b> y el clásico <b>detector de bordes Canny</b>.<br>
        Observa cómo cambia la nitidez, el ruido y las líneas en una imagen.
    </div>
    """, unsafe_allow_html=True)

    # ==== CONTENIDO ====
    uploaded = st.file_uploader("📤 Sube una imagen para aplicar filtros", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = read_image(uploaded)
        st.image(cv2_to_pil(img), caption="🖼 Imagen original", use_container_width=True)

        st.markdown("---")
        st.subheader("🎛 Filtros de Suavizado")
        st.markdown("Ajusta los kernels y observa cómo se reducen los detalles o el ruido de la imagen.")

        # Gaussian Blur
        k = st.slider("Kernel para Gaussian Blur (valor impar)", 1, 31, 5, step=2)
        blurred = cv2.GaussianBlur(img, (k, k), 0)
        st.image(cv2_to_pil(blurred), caption=f"Gaussian Blur (k={k})", use_container_width=True)

        # Median Blur
        median_k = st.slider("Kernel para Median Blur", 1, 15, 3, step=2)
        median = cv2.medianBlur(img, median_k)
        st.image(cv2_to_pil(median), caption=f"Median Blur (k={median_k})", use_container_width=True)

        st.markdown("---")
        st.subheader("🪞 Filtro de Enfoque (Sharpen)")
        st.markdown("Aumenta el contraste de los bordes para resaltar detalles finos mediante la técnica de *unsharp masking*.")
        amount = st.slider("Nivel de nitidez", 0.0, 3.0, 1.0, step=0.1)
        blurred_small = cv2.GaussianBlur(img, (0, 0), 3)
        sharpen = cv2.addWeighted(img, 1 + amount, blurred_small, -amount, 0)
        st.image(cv2_to_pil(sharpen), caption=f"Imagen con mayor nitidez (amount={amount})", use_container_width=True)

        st.markdown("---")
        st.subheader("⚡ Detección de Bordes — Canny Edge Detector")
        st.markdown("""
        El detector de Canny busca zonas de cambio brusco de intensidad (bordes).  
        Ajusta los umbrales para controlar qué tan sensibles son los bordes detectados.
        """)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col1, col2 = st.columns(2)
        with col1:
            low = st.slider("Canny — Límite inferior", 10, 200, 100)
        with col2:
            high = st.slider("Canny — Límite superior", 50, 300, 200)

        edges = cv2.Canny(gray, low, high)
        st.image(Image.fromarray(edges), caption=f"Bordes detectados (Canny {low}-{high})", use_container_width=True)

# -----------------------
# CAPÍTULO 3 - Cartoonizing (webcam/photo)
# -----------------------
elif chapter_number == 3:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

    # Encabezado visual atractivo
    st.markdown("""
    <style>
    .title-cartoon {
        font-size: 2em;
        font-weight: 800;
        color: #1565c0;
        text-align: center;
        margin-bottom: 0.3em;
    }
    .subtitle-cartoon {
        text-align: center;
        color: #444;
        font-size: 1.1em;
        margin-bottom: 1.5em;
    }
    /* Mejora visual del marco de cámara */
    video {
        border: 4px solid #1565c0 !important;
        border-radius: 16px !important;
        box-shadow: 0 0 20px rgba(21,101,192,0.4);
    }
    /* Estilo del slider */
    .stSlider label {
        color: #1565c0 !important;
        font-weight: 600;
    }
    /* Botón de cámara y controles */
    button[title="Start"] {
        background-color: #1565c0 !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: 0.3s;
    }
    button[title="Start"]:hover {
        background-color: #1e88e5 !important;
        transform: scale(1.03);
    }
    button[title="Stop"] {
        background-color: #c62828 !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    </style>

    <div class="title-cartoon">🎨 Capítulo 3 — Cartoonizing en Tiempo Real 📸</div>
    <div class="subtitle-cartoon">
        Activa tu cámara y observa cómo el filtro <b>cartoon</b> convierte tu video en una caricatura en vivo.<br>
        Ajusta la intensidad del efecto desde la barra lateral.
    </div>
    """, unsafe_allow_html=True)

    # --- Procesamiento de video ---
    class Cartoonizer(VideoProcessorBase):
        def __init__(self):
            self.num_bilateral = 5

        def cartoonize(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 9, 2
            )
            color = img.copy()
            for _ in range(self.num_bilateral):
                color = cv2.bilateralFilter(color, 9, 75, 75)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            return av.VideoFrame.from_ndarray(cartoon, format="bgr24")

        def recv(self, frame):
            return self.cartoonize(frame)

    # --- Barra lateral ---
    st.sidebar.subheader("🎛 Ajustes del filtro cartoon")
    st.sidebar.markdown("Usa el control deslizante para cambiar la **suavidad del efecto**.")
    bilateral = st.sidebar.slider("Nivel de suavizado bilateral", 1, 10, 5)

    # --- Cámara ---
    webrtc_ctx = webrtc_streamer(
        key="cartoon",
        video_processor_factory=Cartoonizer,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.num_bilateral = bilateral

# -----------------------
# CAPÍTULO 4 - Body Parts (Haar Cascades)
# -----------------------
# CAPÍTULO 4 - Body Parts (Haar Cascades)
# -----------------------
elif chapter_number == 4:
    # ==== ESTILO VISUAL ====
    st.markdown("""
    <style>
    .title-haar {
        font-size: 2em;
        font-weight: 800;
        color: #c2185b;
        text-align: center;
        margin-bottom: 0.3em;
    }
    .subtitle-haar {
        text-align: center;
        color: #444;
        font-size: 1.05em;
        margin-bottom: 1.5em;
    }
    img {
        border-radius: 14px !important;
        box-shadow: 0 4px 18px rgba(194, 24, 91, 0.25);
        transition: transform 0.2s ease-in-out;
    }
    img:hover {
        transform: scale(1.01);
    }
    div.stButton > button {
        background-color: #c2185b !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: 0.3s;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #d81b60 !important;
        transform: scale(1.05);
    }
    .stSlider label {
        color: #c2185b !important;
        font-weight: 600 !important;
    }
    </style>

    <div class="title-haar">💀 Capítulo 4 — Detección Facial con Haar Cascades</div>
    <div class="subtitle-haar">
        Utiliza clasificadores <b>Haar Cascade</b> preentrenados para detectar rostros y ojos en imágenes.  
        ¡Sube una foto y observa cómo la IA encuentra las caras y sus miradas! 👁👁
    </div>
    """, unsafe_allow_html=True)

    # ==== LÓGICA ORIGINAL (sin cambios) ====
    uploaded = st.file_uploader("📤 Sube una foto con rostros", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = read_image(uploaded)
        img_draw = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Clasificadores preentrenados
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        # Detección
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        st.success(f"✅ Caras detectadas: {len(faces)}")

        # Dibujar detecciones
        for (x, y, w, h) in faces:
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img_draw[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

        # Mostrar resultado
        st.image(cv2_to_pil(img_draw), caption="🧍‍♂️ Detección de rostros y ojos", use_container_width=True)

        st.markdown("""
        ---
        <div style="text-align:center; color:#777; font-size:0.9em;">
        <b>Nota:</b> los clasificadores Haar funcionan mejor con buena iluminación y rostros frontales.  
        Si no detecta correctamente, intenta con otra imagen o ajusta el ángulo.
        </div>
        """, unsafe_allow_html=True)

# -----------------------
# CAPÍTULO 5 - Feature Extraction
# -----------------------
elif chapter_number == 5:
    # ==== ESTILO PERSONALIZADO ====
    st.markdown("""
    <style>
    .title-orb {
        font-size: 2em;
        font-weight: 800;
        color: #00897b;
        text-align: center;
        margin-bottom: 0.3em;
    }
    .subtitle-orb {
        text-align: center;
        color: #333;
        font-size: 1.05em;
        margin-bottom: 1.5em;
    }
    img {
        border-radius: 14px !important;
        box-shadow: 0 4px 18px rgba(0, 137, 123, 0.25);
        transition: transform 0.2s ease-in-out;
    }
    img:hover {
        transform: scale(1.01);
    }
    div.stButton > button {
        background-color: #00897b !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: 0.3s;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #009688 !important;
        transform: scale(1.05);
    }
    .stFileUploader label {
        color: #00897b !important;
        font-weight: 600 !important;
    }
    </style>

    <div class="title-orb">🔍 Capítulo 5 — Extracción de Características con ORB</div>
    <div class="subtitle-orb">
        El detector <b>ORB (Oriented FAST and Rotated BRIEF)</b> identifica puntos clave en una imagen  
        y los compara con otra para encontrar coincidencias.  
        ¡Perfecto para reconocimiento de objetos o logos en escenas complejas! 💫
    </div>
    """, unsafe_allow_html=True)

    # ==== LÓGICA ORIGINAL (sin cambios) ====
    uploaded = st.file_uploader("📤 Sube imagen para detectar keypoints", type=["png","jpg","jpeg"])
    uploaded2 = st.file_uploader("📸 Imagen para matching (opcional)", type=["png","jpg","jpeg"])

    if uploaded:
        img1 = read_image(uploaded)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(500)
        kp1 = orb.detect(gray1, None)
        img_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0),
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        st.image(cv2_to_pil(img_kp),
                 caption=f"✨ Keypoints detectados con ORB: {len(kp1)}",
                 use_container_width=True)

        if uploaded2:
            img2 = read_image(uploaded2)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            kp1, des1 = orb.detectAndCompute(gray1, None)

            if des1 is None or des2 is None:
                st.warning("⚠️ No se encontraron descriptores en alguna imagen.")
            else:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                match_img = cv2.drawMatches(img1, kp1, img2, kp2,
                                            matches[:20], None,
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                st.image(cv2_to_pil(match_img),
                         caption="🔗 Top 20 coincidencias (ORB + BFMatcher)",
                         use_container_width=True)

                st.markdown("""
                <div style="text-align:center; color:#555; font-size:0.9em;">
                    <b>Nota:</b> Las líneas verdes conectan los puntos coincidentes entre ambas imágenes.<br>
                    A menor distancia, mayor similitud visual.
                </div>
                """, unsafe_allow_html=True)

# -----------------------
# -----------------------
# CAPÍTULO 6 - Seam Carving (completo)
# -----------------------
# CAPÍTULO 6 — Seam Carving (Reducción inteligente de contenido) 🧩
# ------------------------------------------------------------
elif chapter_number == 6:
    st.markdown("""
    <style>
    .main-title {
        font-size: 2em;
        font-weight: 800;
        color: #005f99;
        text-align: center;
        margin-bottom: 0.2em;
    }
    .subtitle {
        text-align: center;
        font-size: 1.05em;
        color: #444;
        margin-bottom: 1.5em;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #0072ff, #00c6ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
    }
    img {
        border-radius: 14px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .stProgress > div > div > div {
        background-color: #00c6ff !important;
    }
    </style>

    <div class="main-title">🧩 Capítulo 6 — Seam Carving (Reducción inteligente de contenido)</div>
    <div class="subtitle">
        Este método elimina píxeles de <b>menor energía visual</b> (baja importancia),
        conservando áreas relevantes como personas, edificios o líneas del horizonte.<br>
        🔹 Calcula un <b>mapa de energía</b> con gradientes Sobel.<br>
        🔹 Encuentra la <b>ruta mínima</b> y elimina un píxel vertical por iteración.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("📷 Sube una imagen (de preferencia un paisaje)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = read_image(uploaded)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        st.image(img, caption=f"📸 Imagen Original ({w}x{h})", use_container_width=True)

        target_w = st.slider("🎯 Ancho objetivo (px)", 50, w, int(w * 0.8), step=10)
        carve_btn = st.button("✨ Aplicar Seam Carving completo")

        # ---- FUNCIONES ----
        @st.cache_data(show_spinner=False)
        def compute_energy(img):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            return np.abs(sobelx) + np.abs(sobely)

        def find_seam(energy):
            h, w = energy.shape
            M = energy.copy()
            backtrack = np.zeros_like(M, dtype=np.int32)
            for i in range(1, h):
                for j in range(w):
                    if j == 0:
                        idx = np.argmin(M[i-1, j:j+2])
                        backtrack[i, j] = idx + j
                        min_energy = M[i-1, idx + j]
                    else:
                        idx_range = M[i-1, max(j-1, 0):min(j+2, w)]
                        idx = np.argmin(idx_range)
                        backtrack[i, j] = idx + j - 1
                        min_energy = idx_range[idx]
                    M[i, j] += min_energy
            seam = []
            j = np.argmin(M[-1])
            for i in reversed(range(h)):
                seam.append((i, j))
                j = backtrack[i, j]
            seam.reverse()
            return seam

        def remove_seam(img, seam):
            h, w, _ = img.shape
            output = np.zeros((h, w-1, 3), dtype=np.uint8)
            for i, (r, c) in enumerate(seam):
                output[i, :, 0] = np.delete(img[i, :, 0], c)
                output[i, :, 1] = np.delete(img[i, :, 1], c)
                output[i, :, 2] = np.delete(img[i, :, 2], c)
            return output

        def seam_carve(img, new_width):
            carved = img.copy()
            total = img.shape[1] - new_width
            progress = st.progress(0, text="Eliminando seams de baja energía...")
            for i in range(total):
                energy = compute_energy(carved)
                seam = find_seam(energy)
                carved = remove_seam(carved, seam)
                progress.progress((i + 1) / total, text=f"Progreso: {i+1}/{total} seams")
            return carved

        # ---- EJECUCIÓN ----
        if carve_btn:
            if target_w >= w:
                st.info("⚠️ El ancho objetivo debe ser menor al original.")
            else:
                with st.spinner("🧠 Calculando y eliminando rutas de baja energía..."):
                    carved_img = seam_carve(img, target_w)
                st.success("✅ Seam Carving completado con éxito.")
                st.image(carved_img, caption=f"Resultado final ({target_w}x{h})", use_container_width=True)

                st.markdown("""
                ---
                ### 🧠 Explicación visual
                - Cada iteración genera un mapa de energía.
                - Se elimina el **camino más “barato”** (energía mínima).
                - Este proceso se repite hasta alcanzar el nuevo ancho.

                **Resultado:** el contenido importante se conserva sin distorsión 👁️‍🗨️
                """)

                resized = cv2.resize(img, (target_w, h))
                st.subheader("📊 Comparativa: Resize clásico vs Seam Carving")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(resized, caption="🪄 Resize clásico", use_container_width=True)
                with col2:
                    st.image(carved_img, caption="🤖 Seam Carving inteligente", use_container_width=True)

# -----------------------
# CAPÍTULO 7 - Detecting Shapes & Approximating Contours
# -----------------------
# CAPÍTULO 7 — Aproximación de contornos (implementado)
# ------------------------------------------------------------
elif chapter_number == 7:
    st.markdown("""
    <style>
    .title-7 {
        font-size: 1.8em;
        font-weight: 800;
        color: #005f99;
        text-align: center;
        margin-bottom: 0.2em;
    }
    .subtitle-7 {
        text-align: center;
        font-size: 1.05em;
        color: #333;
        margin-bottom: 1.5em;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #0072ff, #00c6ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
    }
    img {
        border-radius: 14px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .stCheckbox > label {
        color: #222 !important;
        font-weight: 500 !important;
    }
    </style>

    <div class="title-7">🔺 Capítulo 7 — Detección y Aproximación de Contornos</div>
    <div class="subtitle-7">
        Sube una imagen o usa la <b>estrella ruidosa de ejemplo</b> 🌟.<br>
        Ajusta el <b>factor de aproximación</b> para observar cómo se simplifica la forma con <code>approxPolyDP</code>.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader("📂 Sube imagen (PNG/JPG)", type=["png", "jpg", "jpeg"])
        use_example = False
        if uploaded is None:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✨ Usar estrella ruidosa de ejemplo"):
                use_example = True

        image = None
        if uploaded:
            image = read_image(uploaded)
        elif use_example:
            img = np.ones((400, 400, 3), dtype=np.uint8) * 255
            points = np.array([
                [200, 50], [225, 160], [310, 140],
                [240, 210], [270, 320], [200, 250],
                [130, 320], [160, 200], [90, 150],
                [185, 140]
            ], np.int32)
            pts = points.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], (0, 0, 0))
            noise = np.random.randint(0, 60, (400, 400, 3), dtype=np.uint8)
            noisy_img = cv2.add(img, noise)
            image = noisy_img

        if image is not None:
            st.image(cv2_to_pil(image), caption="🖼️ Imagen original", use_container_width=True)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        factor = st.slider("🎯 Factor de aproximación (ε = factor × perímetro)", 0.001, 0.2, 0.05, step=0.001)
        show_orig = st.checkbox("Mostrar contornos originales", value=False)
        show_bbox = st.checkbox("Mostrar boundingRect / minAreaRect", value=False)
        run = st.button("▶️ Aplicar aproximación")

        if run and image is not None:
            img_proc = image.copy()
            gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
            contours = find_contours_from_gray(gray)
            result_img = img_proc.copy()
            approx_img = img_proc.copy()
            bbox_img = img_proc.copy()

            for i, c in enumerate(contours):
                perim = cv2.arcLength(c, True)
                epsilon = factor * perim
                approx = cv2.approxPolyDP(c, epsilon, True)

                if show_orig:
                    cv2.drawContours(result_img, contours, -1, (0, 255, 0), 1)
                cv2.drawContours(approx_img, [approx], -1, (0, 0, 255), 2)

                if show_bbox:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(bbox_img, [box], 0, (0, 128, 255), 2)

                st.markdown(
                    f"<div style='color:#005f99; font-weight:500;'>"
                    f"🔹 Contorno {i+1}: puntos originales = {len(c)}, perímetro = {perim:.1f}, "
                    f"aproximados = {len(approx)}, ε = {epsilon:.3f}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.markdown("<hr style='margin-top:1em; margin-bottom:1em;'>", unsafe_allow_html=True)
            st.subheader("📊 Resultados visuales")

            if show_orig:
                st.image(cv2_to_pil(result_img), caption="🟢 Contornos originales", use_container_width=True)

            st.image(cv2_to_pil(approx_img),
                     caption=f"🔴 Aproximación (factor = {factor})",
                     use_container_width=True)

            if show_bbox:
                st.image(cv2_to_pil(bbox_img),
                         caption="🟦 Bounding boxes",
                         use_container_width=True)

# -----------------------
# CAPÍTULO 8 - Object Tracking (color-based)
# -----------------------
# CAPÍTULO 8 — Object Tracking (basado en color)
# ------------------------------------------------------------
elif chapter_number == 8:
    st.markdown("""
    <style>
    .title-8 {
        font-size: 1.8em;
        font-weight: 800;
        color: #0072ff;
        text-align: center;
        margin-bottom: 0.2em;
    }
    .subtitle-8 {
        text-align: center;
        font-size: 1.05em;
        color: #333;
        margin-bottom: 1.5em;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #0072ff, #00c6ff) !important;
    }
    img {
        border-radius: 14px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .hsv-box {
        background-color: #f5f9ff;
        border-radius: 10px;
        padding: 10px 15px;
        margin-top: 1em;
        border: 1px solid #cce4ff;
    }
    .hsv-title {
        color: #005f99;
        font-weight: 600;
        margin-bottom: 0.5em;
    }
    </style>

    <div class="title-8">🎯 Capítulo 8 — Seguimiento de Objetos por Color</div>
    <div class="subtitle-8">
        Este demo permite rastrear un objeto en función de su <b>color HSV</b>.  
        Puedes usar la <b>cámara</b> o subir una imagen, luego ajustar los valores para detectar el color deseado.  
        Ideal para reconocer pelotas, marcadores o superficies específicas.
    </div>
    """, unsafe_allow_html=True)

    # Entrada: cámara o archivo
    camera_img = st.camera_input("📸 Toma una foto (opcional)")
    uploaded = st.file_uploader("📂 O sube una imagen desde tu galería", type=["png", "jpg", "jpeg"])
    img = None

    if camera_img is not None:
        img = pil_to_cv2(Image.open(camera_img))
    elif uploaded:
        img = read_image(uploaded)

    if img is not None:
        st.image(cv2_to_pil(img), caption="🖼️ Imagen original", use_container_width=True)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        st.markdown("""
        <div class="hsv-box">
            <div class="hsv-title">🎨 Ajusta el rango HSV para detectar un color (por ejemplo: rojo, verde, azul)</div>
        </div>
        """, unsafe_allow_html=True)

        # Sliders HSV
        h1 = st.slider("Hue mínimo (H)", 0, 179, 35)
        s1 = st.slider("Saturación mínima (S)", 0, 255, 50)
        v1 = st.slider("Valor mínimo (V)", 0, 255, 50)
        h2 = st.slider("Hue máximo (H)", 0, 179, 85)
        s2 = st.slider("Saturación máxima (S)", 0, 255, 255)
        v2 = st.slider("Valor máximo (V)", 0, 255, 255)

        lower = np.array([h1, s1, v1])
        upper = np.array([h2, s2, v2])
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(img, img, mask=mask)

        st.subheader("🧩 Resultado de detección")
        st.image(Image.fromarray(mask), caption="🕳️ Máscara (escala de grises)", use_container_width=True)
        st.image(cv2_to_pil(res), caption="🎯 Detección de color", use_container_width=True)

        # Cálculo de centroide
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(res, (cx, cy), 8, (0, 0, 255), -1)

            st.success(f"✅ Objeto detectado — Centroid: ({cx}, {cy})")
            st.image(cv2_to_pil(res), caption="📍 Resultado con centroide", use_container_width=True)
        else:
            st.warning("⚠️ No se detectó ningún objeto dentro del rango HSV seleccionado.")


# -----------------------
# CAPÍTULO 9 - Object Recognition (feature matching)
# -----------------------
# CAPÍTULO 9 — Object Recognition (ORB + BFMatcher demo)
# ------------------------------------------------------------
elif chapter_number == 9:
    st.markdown("""
    <style>
    .title-9 {
        font-size: 1.8em;
        font-weight: 800;
        color: #009688;
        text-align: center;
        margin-bottom: 0.2em;
    }
    .subtitle-9 {
        text-align: center;
        font-size: 1.05em;
        color: #333;
        margin-bottom: 1.5em;
    }
    div.stFileUploader {
        background-color: #f0f9f7;
        border-radius: 10px;
        padding: 10px 15px;
        border: 1px solid #b2dfdb;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #26a69a, #00796b) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #00796b, #26a69a) !important;
    }
    img {
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .match-box {
        background-color: #e0f2f1;
        border-left: 5px solid #009688;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 1em;
    }
    </style>

    <div class="title-9">🧩 Capítulo 9 — Reconocimiento de Objetos (ORB + BFMatcher)</div>
    <div class="subtitle-9">
        Esta demo permite reconocer un objeto en una escena a partir de sus <b>características locales</b> usando <b>ORB (Oriented FAST and BRIEF)</b>  
        y el emparejamiento de descriptores con <b>BFMatcher</b> y prueba de relación de Lowe.  
        Ideal para reconocer logos o patrones en imágenes.
    </div>
    """, unsafe_allow_html=True)

    uploaded_q = st.file_uploader("📷 Imagen del objeto (query)", type=["png", "jpg", "jpeg"], key="q")
    uploaded_t = st.file_uploader("🌆 Imagen de la escena (train)", type=["png", "jpg", "jpeg"], key="t")

    if uploaded_q and uploaded_t:
        img_q = read_image(uploaded_q)
        img_t = read_image(uploaded_t)

        st.markdown("### 🔍 Imágenes cargadas")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2_to_pil(img_q), caption="🔸 Query (objeto a buscar)", use_container_width=True)
        with col2:
            st.image(cv2_to_pil(img_t), caption="🔹 Scene (donde buscar)", use_container_width=True)

        gray_q = cv2.cvtColor(img_q, cv2.COLOR_BGR2GRAY)
        gray_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(1000)

        kp1, des1 = orb.detectAndCompute(gray_q, None)
        kp2, des2 = orb.detectAndCompute(gray_t, None)

        if des1 is None or des2 is None:
            st.warning("⚠️ No se encontraron descriptores suficientes en alguna imagen.")
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(des1, des2, k=2)

            # Lowe's ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            st.markdown(f"<div class='match-box'>✨ Matches buenos encontrados: <b>{len(good)}</b></div>", unsafe_allow_html=True)

            if len(good) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if H is not None:
                    hq, wq = gray_q.shape
                    pts = np.float32([[0, 0], [0, hq - 1], [wq - 1, hq - 1], [wq - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, H)

                    scene_with_box = img_t.copy()
                    dst_int = np.int32(dst)
                    cv2.polylines(scene_with_box, [dst_int], True, (0, 255, 0), 3)
                    st.success("✅ Objeto detectado correctamente en la escena.")
                    st.image(cv2_to_pil(scene_with_box), caption="🟩 Objeto detectado (bounding polygon)", use_container_width=True)
                else:
                    st.warning("⚠️ No se pudo calcular la homografía. Intenta con imágenes más claras o menos ruido.")
            else:
                st.warning("⚠️ No hay suficientes matches buenos para localizar el objeto. Intenta mejorar el enfoque o usar otra imagen.")

# -----------------------
# CAPÍTULO 10 - Realidad Aumentada en Tiempo Real (Detección por color)
# ------------------------------------------------------
elif chapter_number == 10:
    st.header("Capítulo 10 — Realidad Aumentada con Cámara (Detección por color) ❤️📓")
    st.markdown("""
    Esta versión usa la cámara en vivo para detectar una **superficie roja** (como una funda o cuaderno rojo)
    y proyectar una imagen sobre ella, incluso si no hay bordes visibles.
    """)

    import av
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

    uploaded_overlay = st.file_uploader(
        "Sube la imagen que quieras proyectar (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
    )

    # selector de color objetivo
    color_choice = st.selectbox("Selecciona color a detectar", ["Rojo", "Verde", "Azul"])

    if uploaded_overlay is not None:
        overlay_img = read_image(uploaded_overlay)
        h_ol, w_ol = overlay_img.shape[:2]

        class ColorAR(VideoTransformerBase):
            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                if color_choice == "Rojo":
                    # Rango rojo (dos zonas en HSV)
                    lower1 = np.array([0, 120, 70])
                    upper1 = np.array([10, 255, 255])
                    lower2 = np.array([170, 120, 70])
                    upper2 = np.array([180, 255, 255])
                    mask1 = cv2.inRange(hsv, lower1, upper1)
                    mask2 = cv2.inRange(hsv, lower2, upper2)
                    mask = cv2.bitwise_or(mask1, mask2)
                elif color_choice == "Verde":
                    lower = np.array([40, 40, 40])
                    upper = np.array([90, 255, 255])
                    mask = cv2.inRange(hsv, lower, upper)
                else:  # Azul
                    lower = np.array([100, 150, 0])
                    upper = np.array([140, 255, 255])
                    mask = cv2.inRange(hsv, lower, upper)

                # Suavizar y encontrar contornos
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                output = img.copy()

                if contours:
                    c = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(c)
                    if area > 30000:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        pts_dst = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
                        pts_src = np.float32([[0,0], [w_ol,0], [w_ol,h_ol], [0,h_ol]])
                        H, _ = cv2.findHomography(pts_src, pts_dst)

                        if H is not None:
                            warped = cv2.warpPerspective(overlay_img, H, (img.shape[1], img.shape[0]))
                            gray_warp = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                            _, mask2 = cv2.threshold(gray_warp, 10, 255, cv2.THRESH_BINARY)
                            mask_inv = cv2.bitwise_not(mask2)
                            bg = cv2.bitwise_and(output, output, mask=mask_inv)
                            fg = cv2.bitwise_and(warped, warped, mask=mask2)
                            output = cv2.add(bg, fg)
                        else:
                            cv2.putText(output, "No homografía", (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv2.putText(output, "Superficie diminuta", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                else:
                    cv2.putText(output, "Buscando color...", (20,40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

                return output

        webrtc_streamer(
            key="color-ar",
            video_transformer_factory=ColorAR,
            media_stream_constraints={"video": True, "audio": False},
        )

        st.info("Apunta la cámara a una superficie **roja**, **verde** o **azul** grande. Tu imagen se proyectará sobre ella.")
    else:
        st.warning("Sube primero una imagen para usarla como overlay.")

# -----------------------
# CAPÍTULO 11 - Machine Learning (ANN) - simple demo
# -----------------------
elif chapter_number == 11:
    st.header("Capítulo 11 — Machine Learning (ANN) - demo con digits (sklearn)")
    st.markdown("Usamos el dataset `sklearn.datasets.load_digits` (pequeño) para entrenar un MLP rápido y predecir imágenes subidas por el usuario (resized a 8x8). Ideal para mostrar pipeline: features -> train -> predict.")
    st.markdown("Nota: esta es una demo educativa, no un sistema de producción.")
    # Cargar dataset y entrenar (rápido)
    digits = load_digits()
    X = digits.data
    y = digits.target
    st.write(f"Dataset digits: {len(X)} muestras, {len(np.unique(y))} clases")
    test_size = st.slider("Tamaño test (%)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)
    with st.spinner("Entrenando MLP..."):
        clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.success(f"Entrenamiento listo — accuracy en test: {acc*100:.2f}%")

    st.subheader("Sube una imagen (dibujada) o usa cámara para predecir")
    camera_img = st.camera_input("Toma una foto de un dígito escrito (ideal en blanco y negro)")
    uploaded = st.file_uploader("O sube una imagen (mejor recortada al dígito)", type=["png","jpg","jpeg"])
    sample = None
    if camera_img:
        sample = Image.open(camera_img).convert("L")
    elif uploaded:
        sample = Image.open(uploaded).convert("L")

    if sample:
        st.image(sample, caption="Input (gris)", use_container_width=True)
        # resize to 8x8 and invert/scale similar to sklearn digits
        # Convertir a escala de grises y binarizar
        img_np = np.array(sample)
        _, binary = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY_INV)

        # Asegurar bordes y centrar el dígito
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        digit = binary[y:y+h, x:x+w]

        # Redimensionar a 8x8
        digit_resized = cv2.resize(digit, (8, 8), interpolation=cv2.INTER_AREA)

        # Normalizar
        arr = digit_resized.astype(np.float32)
        arr = (arr / 255.0) * 16.0
        arr = arr.flatten().reshape(1, -1)

        arr = arr.flatten()
        # normalize to same range as digits dataset (0-16)
        arr = (arr / 255.0) * 16.0
        arr = arr.reshape(1, -1)
        pred = clf.predict(arr)[0]
        st.write(f"Predicción: {pred}")

# -----------------------
# Footer elegante
# -----------------------
st.markdown("""
<div class="footer">
    <hr>
    <p>📚 Proyecto académico de visión por computadora — basado en "OpenCV 3.x with Python by Example".</p>
    <p>Desarrollado con ❤️ por <b>Imanol Polonio</b></p>
    <p><b>Univseridad Nacional de Trujillo</b></p>
    
</div>
""", unsafe_allow_html=True)

