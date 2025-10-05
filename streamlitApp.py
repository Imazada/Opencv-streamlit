# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="OpenCV by Example - Mini Apps", layout="wide")

# -----------------------
# Helpers
# -----------------------
def read_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def pil_to_cv2(img_pil):
    img = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv):
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def find_contours_from_gray(gray):
    # Binary threshold then find contours
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# -----------------------
# Sidebar - Selección de capítulo
# -----------------------
st.sidebar.title("OpenCV 3.x - Selector de capítulos")
chapter = st.sidebar.selectbox("Selecciona capítulo", [
    "1 - Geometric Transformations",
    "2 - Edges & Filters",
    "3 - Cartoonizing (webcam)",
    "4 - Body Parts (Haar Cascades)",
    "5 - Feature Extraction",
    "6 - Seam Carving",
    "7 - Detecting Shapes & Approximating Contours",
    "8 - Object Tracking (webcam)",
    "9 - Object Recognition",
    "10 - Augmented Reality",
    "11 - Machine Learning (ANN)"
])

st.title("OpenCV 3.x with Python — Mini Apps (Streamlit)")
st.markdown("Selecciona un capítulo en la barra lateral. Este demo implementa completamente el **Capítulo 7: Approximating a contour**. "
            "Los demás capítulos contienen descripciones y scaffolds para que los completes o los usemos como plantilla.")

# -----------------------
# CAPÍTULO 7: IMPLEMENTACIÓN COMPLETA
# -----------------------
if chapter.startswith("7"):
    st.header("Capítulo 7 — Aproximación de contornos (demo)")
    st.markdown("""
    Este demo muestra cómo:
    - Cargar una imagen (puedes usar la estrella ruidosa).
    - Detectar contornos.
    - Aproximarlos con `cv2.approxPolyDP` variando el *factor* (epsilon = factor * perímetro).
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Sube o usa imagen de ejemplo")
        uploaded = st.file_uploader("Sube una imagen (PNG/JPG). Si no subes, se usará una estrella ruidosa de ejemplo.", type=["png","jpg","jpeg"])
        use_example = False
        if uploaded is None:
            if st.button("Usar estrella ruidosa de ejemplo"):
                use_example = True
        image = None
        if uploaded:
            image = read_image(uploaded)
        elif use_example:
            # Crear estrella ruidosa programáticamente
            img = np.ones((400, 400, 3), dtype=np.uint8) * 255
            points = np.array([
                [200, 50], [225, 160], [310, 140],
                [240, 210], [270, 320], [200, 250],
                [130, 320], [160, 200], [90, 150],
                [185, 140]
            ], np.int32)
            pts = points.reshape((-1,1,2))
            cv2.fillPoly(img, [pts], (0,0,0))
            noise = np.random.randint(0, 60, (400,400,3), dtype=np.uint8)
            noisy_img = cv2.add(img, noise)
            image = noisy_img
        else:
            st.info("Sube una imagen o pulsa 'Usar estrella ruidosa de ejemplo'.")

        if image is not None:
            st.image(cv2_to_pil(image), caption="Imagen original (BGR -> RGB)", use_column_width=True)

    with col2:
        st.subheader("Ajustes y resultados")
        factor = st.slider("Factor de aproximación (epsilon = factor * perímetro)", min_value=0.001, max_value=0.2, value=0.05, step=0.001)
        show_orig_contours = st.checkbox("Mostrar contornos originales", value=False)
        show_bbox = st.checkbox("Mostrar boundingRect y minAreaRect", value=False)
        run_button = st.button("Aplicar aproximación")

        if run_button and image is not None:
            img_proc = image.copy()
            gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
            contours = find_contours_from_gray(gray)

            # Imágenes para cada resultado
            result_img = img_proc.copy()
            approx_img = img_proc.copy()
            bbox_img = img_proc.copy()

            # Dibujar contornos originales (opcional)
            if show_orig_contours:
                cv2.drawContours(result_img, contours, -1, (0,255,0), 1)

            # Aproximar contornos y dibujar
            for i, c in enumerate(contours):
                perim = cv2.arcLength(c, True)
                epsilon = factor * perim
                approx = cv2.approxPolyDP(c, epsilon, True)
                cv2.drawContours(approx_img, [approx], -1, (0,0,255), 2)
                # Para explicación: imprimir tamaños
                st.write(f"Contorno {i}: puntos orig={len(c)}, perim={perim:.1f} px -> approx={len(approx)} (epsilon={epsilon:.2f})")

                if show_bbox:
                    # boundingRect (axis-aligned)
                    x,y,w,h = cv2.boundingRect(c)
                    cv2.rectangle(bbox_img, (x,y), (x+w, y+h), (255,0,0), 2)
                    # minAreaRect (rotated)
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(bbox_img, [box], 0, (0,128,255), 2)

            st.subheader("Resultados")
            st.image(cv2_to_pil(result_img), caption="Contornos detectados (opcional)", use_column_width=True)
            st.image(cv2_to_pil(approx_img), caption=f"Aproximación (factor={factor})", use_column_width=True)
            if show_bbox:
                st.image(cv2_to_pil(bbox_img), caption="Bounding boxes", use_column_width=True)

            st.success("Listo — ajusta `factor` y vuelve a aplicar para ver cambios.")
            st.markdown("**Nota:** `approxPolyDP` suaviza contornos; si quieres un rectángulo perfecto usa `cv2.boundingRect()` o `cv2.minAreaRect()`.")

# -----------------------
# SCAFFOLD para otros capítulos (placeholders - explicaciones y entradas simples)
# -----------------------
else:
    st.header(f"{chapter} — Demo / Scaffold")
    st.markdown("Aquí van una demo ligera o explicación interactiva del capítulo. Abajo verás una breve descripción y un ejemplo minimalista.")

    if chapter.startswith("1"):
        st.subheader("Geometric Transformations")
        st.markdown("- Subir imagen y aplicar: translate, rotate, scale, affine, perspective (transformaciones).")
        img_file = st.file_uploader("Sube imagen para transformaciones", type=["png","jpg"])
        if img_file:
            img = read_image(img_file)
            st.image(cv2_to_pil(img), caption="Original")
            # Ejemplo: rotar con slider
            angle = st.slider("Rotar (grados)", -180, 180, 0)
            h,w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w,h))
            st.image(cv2_to_pil(rotated), caption=f"Rotada {angle}°")

    elif chapter.startswith("2"):
        st.subheader("Detecting Edges & Filters")
        st.markdown("- Blur, Gaussian, Sharpen, Canny.")
        uploaded = st.file_uploader("Sube imagen para filtros", type=["png","jpg"])
        if uploaded:
            img = read_image(uploaded)
            k = st.slider("Kernel (odd)", 1, 31, 5, step=2)
            blurred = cv2.GaussianBlur(img, (k,k), 0)
            edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
            st.image(cv2_to_pil(blurred), caption="Blur")
            st.image(Image.fromarray(edges), caption="Canny (grayscale)")

    elif chapter.startswith("3"):
        st.subheader("Cartoonizing (Webcam) — Placeholder")
        st.markdown("Este capítulo usa webcam. En Streamlit Cloud no siempre funciona la webcam del cliente. "
                    "Localmente puedes usar `cv2.VideoCapture(0)` y mostrar en `st.image` en un loop.")

    elif chapter.startswith("4"):
        st.subheader("Detecting and Tracking Body Parts (Haar Cascades)")
        st.markdown("Sube una foto y aplica Haar cascades para detectar rostros/ojos.")
        uploaded = st.file_uploader("Sube imagen para Haar", type=["png","jpg"])
        if uploaded:
            img = read_image(uploaded)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            st.image(cv2_to_pil(img), caption=f"Detección de {len(faces)} caras")

    elif chapter.startswith("5"):
        st.subheader("Feature Extraction (ORB/SIFT/FAST) — Mini ORB")
        uploaded = st.file_uploader("Imagen para keypoints", type=["png","jpg"])
        if uploaded:
            img = read_image(uploaded)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(300)
            kp = orb.detect(gray, None)
            img_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            st.image(cv2_to_pil(img_kp), caption=f"Keypoints ORB: {len(kp)}")

    elif chapter.startswith("6"):
        st.subheader("Seam Carving — Explicación")
        st.markdown("Seam carving es content-aware resizing. Aquí ponemos explicación y link a código externo si se requiere.")

    elif chapter.startswith("8"):
        st.subheader("Object Tracking (webcam) — Placeholder")
        st.markdown("Ejemplos: trackeo por color (HSV), frame differencing. Para demos en vivo usa `cv2.VideoCapture` localmente.")

    elif chapter.startswith("9"):
        st.subheader("Object Recognition — Explicación y scaffold")
        st.markdown("Build a visual dictionary, train classifier. Por simplicidad, se puede mostrar ejemplo de ORB+FLANN matching.")

    elif chapter.startswith("10"):
        st.subheader("Augmented Reality — Placeholder")
        st.markdown("Pose estimation, homography. Puedes subir una imagen con un marcador y demostrar homografía simple.")

    elif chapter.startswith("11"):
        st.subheader("Machine Learning (ANN)")
        st.markdown("Mini-ML demo: usar sklearn MLP para clasificar imágenes simples (ej: MNIST pequeño) o features extraídos con ORB.")

    st.info("Estos scaffolds son plantillas rápidas. Si quieres que implemente uno de estos capítulos en detalle (con webcam o con imagen), lo hago a continuación en el orden que prefieras.")
