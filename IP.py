import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage import exposure, color, img_as_float, img_as_ubyte

st.set_page_config(page_title="Image Processing Demo", layout="wide")

# ---------- Helper functions ----------
def show_hist(image, is_color):
    fig, ax = plt.subplots()
    if is_color:
        colors = ('r', 'g', 'b')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([image],[i],None,[256],[0,256])
            ax.plot(hist, color=col)
            ax.set_xlim([0,256])
    else:
        ax.hist(image.ravel(), bins=256, range=(0,256), color='black')
    ax.set_title("Histogram")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def to_gray_if_needed(img, as_gray):
    if as_gray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def linear_negative(img):
    return cv2.bitwise_not(img)

def contrast_stretch(img, low_perc, high_perc):
    p2, p98 = np.percentile(img, (low_perc, high_perc))
    return exposure.rescale_intensity(img, in_range=(p2, p98))

def piecewise_linear(img, r1,s1,r2,s2):
    img = img.astype(np.float32)
    output = np.piecewise(img,
                          [img < r1,
                           (img >= r1) & (img <= r2),
                           img > r2],
                          [lambda x: (s1/r1)*x,
                           lambda x: ((s2 - s1)/(r2 - r1))*(x - r1) + s1,
                           lambda x: ((255 - s2)/(255 - r2))*(x - r2) + s2])
    return np.clip(output,0,255).astype(np.uint8)

def log_transform(img, c):
    img_float = img_as_float(img)
    log_img = c * np.log1p(img_float)
    return img_as_ubyte(log_img / log_img.max())

def gamma_transform(img, gamma):
    img_float = img_as_float(img)
    gamma_img = np.power(img_float, gamma)
    return img_as_ubyte(gamma_img)

def hist_equal(img):
    if len(img.shape)==2:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def adaptive_hist_equal(img, clip):
    if len(img.shape)==2:
        return exposure.equalize_adapthist(img, clip_limit=clip)
    img_lab = color.rgb2lab(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_lab[:,:,0] = exposure.equalize_adapthist(img_lab[:,:,0]/100, clip_limit=clip)*100
    return cv2.cvtColor((color.lab2rgb(img_lab)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def clahe(img, clip, tile):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile))
    if len(img.shape)==2:
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ---------- UI ----------
st.title("🖼️ Interactive Image Processing")

uploaded = st.file_uploader("Upload an image", type=['png','jpg','jpeg'])
img_type = st.radio("Image type", ["Grayscale", "Color"])

method = st.selectbox(
    "Select processing method",
    ["Linear Negative", "Contrast Stretching", "Piecewise Linear Transformation",
     "Log Transformation", "Gamma Transformation",
     "Histogram Equalization", "Adaptive Histogram Equalization", "CLAHE"]
)

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray_requested = (img_type == "Grayscale")
    if gray_requested:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    processed = None

    # Parameter controls
    if method == "Linear Negative":
        processed = linear_negative(img)

    elif method == "Contrast Stretching":
        low = st.slider("Low Percentile", 0.0, 10.0, 2.0, step=0.5)
        high = st.slider("High Percentile", 90.0, 100.0, 98.0, step=0.5)
        processed = contrast_stretch(img, low, high)

    elif method == "Piecewise Linear Transformation":
        r1 = st.slider("r1", 0, 255, 70)
        s1 = st.slider("s1", 0, 255, 0)
        r2 = st.slider("r2", 0, 255, 140)
        s2 = st.slider("s2", 0, 255, 255)
        processed = piecewise_linear(img, r1,s1,r2,s2)

    elif method == "Log Transformation":
        c = st.slider("Constant c", 0.1, 5.0, 1.0, step=0.1)
        processed = log_transform(img, c)

    elif method == "Gamma Transformation":
        g = st.slider("Gamma", 0.1, 5.0, 1.0, step=0.1)
        processed = gamma_transform(img, g)

    elif method == "Histogram Equalization":
        processed = hist_equal(img)

    elif method == "Adaptive Histogram Equalization":
        clip = st.slider("Clip Limit", 0.01, 0.1, 0.03, step=0.01)
        processed = adaptive_hist_equal(img, clip)

    elif method == "CLAHE":
        clip = st.slider("Clip Limit", 1.0, 10.0, 2.0, step=0.5)
        tile = st.slider("Tile Grid Size", 2, 16, 8)
        processed = clahe(img, clip, tile)

    # ---------- Display ----------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if not gray_requested else img, channels="RGB" if not gray_requested else "GRAY")
        show_hist(img, not gray_requested)
    with col2:
        st.subheader("Processed Image")
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB) if (processed is not None and len(processed.shape)==3) else processed,
                 channels="RGB" if (processed is not None and len(processed.shape)==3) else "GRAY")
        show_hist(processed, not gray_requested)
