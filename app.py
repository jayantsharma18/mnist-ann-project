import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

def center_image(img_gray):
    img_gray = np.array(img_gray)
    img_gray[img_gray < 50] = 0
    
    rows = np.any(img_gray > 0, axis=1)
    cols = np.any(img_gray > 0, axis=0)
    
    if not np.any(rows):
        return np.zeros((28, 28))
        
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    cropped = img_gray[rmin:rmax+1, cmin:cmax+1]
    
    im = Image.fromarray(cropped)
    im.thumbnail((20, 20), Image.Resampling.LANCZOS)
    cropped_resized = np.array(im)
    
    final_img = np.zeros((28, 28))
    r, c = cropped_resized.shape
    
    start_r = (28 - r) // 2
    start_c = (28 - c) // 2
    
    final_img[start_r:start_r+r, start_c:start_c+c] = cropped_resized
    return final_img

st.set_page_config(page_title="MNIST ANN Model", layout="centered")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #000000 0%, #050E24 40%, #0D2861 100%);
    background-attachment: fixed;
}
</style>
""", unsafe_allow_html=True)

st.title("MNIST ANN Model")

st.markdown("### Upload an image or draw a digit below 👇")

# Upload Section
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('L')
    img_array = np.array(img)

    if np.mean(img_array) > 127:
        img_array = 255 - img_array
        
    final_img = center_image(img_array)

    st.image(final_img, caption="Processed Image (28x28)", width=150)

# Drawing Section
st.markdown("---")
st.subheader("Draw a Digit")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img_array = np.array(canvas_result.image_data)
    img_gray = img_array[:, :, 0]

    if np.sum(img_gray) > 0:
        final_img = center_image(img_gray)
        st.image(final_img, caption="Processed Drawing (28x28)", width=150)
