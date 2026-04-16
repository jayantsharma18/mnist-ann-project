import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from streamlit_drawable_canvas import st_canvas
from model import train_model

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

def predict(model, img_array):
    flat = img_array.reshape(1, -1)
    probs = model.predict_proba(flat)[0]
    pred = np.argmax(probs)
    confidence = np.max(probs) * 100
    return pred, confidence

st.set_page_config(page_title="MNIST ANN Model", layout="centered")
st.title("MNIST ANN Model")

@st.cache_resource
def load_model():
    return train_model()

model, history, x_test, y_test = load_model()

st.subheader("Training Loss vs Epoch")
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='Loss')
ax.plot(history.history['val_loss'], label='Validation Score')
ax.legend()
st.pyplot(fig)

y_pred = model.predict(x_test.reshape(len(x_test), -1))
st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.markdown("---")
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('L')
    img_array = np.array(img)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    final_img = center_image(img_array)
    final_img = final_img / 255.0
    pred, confidence = predict(model, final_img)
    st.subheader(f"🧠 Prediction: **{pred}**")
    st.write(f"Confidence: {confidence:.2f}%")
    st.progress(int(confidence))

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
        final_img = final_img / 255.0
        pred, confidence = predict(model, final_img)
        st.subheader(f"🎨 Drawing Prediction: **{pred}**")
        st.write(f"Confidence: {confidence:.2f}%")
        st.progress(int(confidence))
