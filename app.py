import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import datetime
from streamlit_drawable_canvas import st_canvas
from openai import OpenAI

st.set_page_config(page_title="ShadeGPT", page_icon="🦷", layout="centered")

st.markdown("## 🦷 ShadeGPT – AI Tooth Shade Detection")
st.markdown("Detect VITA 3D-Master shades from smile images using AI and GPT explanations.")

language = st.selectbox("🌐 Select Language", ["English", "Arabic"])
mode = st.radio("🧑‍⚕️ Who is using this app?", ["Doctor", "Patient"])

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"]) if "OPENAI_API_KEY" in st.secrets else None

# Model selection
available_models = ["gpt-3.5-turbo", "gpt-4"]
selected_model = st.selectbox("🤖 Choose GPT model", available_models)

def estimate_vita_shade(lab):
    L, a, b = lab
    if L > 75:
        return "1M1" if b < 15 else "1M2" if b < 25 else "2M2"
    elif L > 65:
        return "2L1.5" if b < 15 else "2M2" if b < 25 else "3M2"
    elif L > 55:
        return "3L1.5" if b < 15 else "3M3"
    else:
        return "4M2"

def explain_shade_with_gpt(L, a, b, shade):
    if not client:
        return "⚠️ No API key found."
    role = "dentist" if mode == "Doctor" else "patient"
    prompt = f"""You are an AI {role}. A tooth has these CIELAB values:
- L*: {L:.2f}
- a*: {a:.2f}
- b*: {b:.2f}
The estimated VITA 3D-Master shade is {shade}. Explain what these values mean and why this shade was chosen in simple terms."""
    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception:
        return "⚠️ GPT could not generate an explanation. Please check your quota or try again later."

def analyze_image(image_np):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    return np.mean(lab.reshape(-1, 3), axis=0)

uploaded_file = st.file_uploader("📸 Upload a smile image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    method = st.radio("Select analysis mode:", ["Automatic (center crop)", "Manual (draw box)"])
    if method == "Automatic (center crop)":
        h, w, _ = img_np.shape
        cropped = img_np[h//3:2*h//3, w//4:3*w//4]
    else:
        st.markdown("Draw a rectangle around the teeth.")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=3,
            background_image=Image.fromarray(img_np),
            update_streamlit=True,
            height=image.height,
            width=image.width,
            drawing_mode="rect",
            key="canvas",
        )
        if canvas_result.json_data and canvas_result.json_data["objects"]:
            obj = canvas_result.json_data["objects"][0]
            left = int(obj["left"])
            top = int(obj["top"])
            width = int(obj["width"])
            height = int(obj["height"])
            cropped = img_np[top:top+height, left:left+width]
        else:
            st.warning("Draw a rectangle to analyze.")
            st.stop()

    avg_lab = analyze_image(cropped)
    L, a, b = avg_lab
    shade = estimate_vita_shade(avg_lab)

    st.subheader("📊 CIELAB Values")
    st.write(f"L*: {L:.2f}, a*: {a:.2f}, b*: {b:.2f}")
    st.subheader("🎨 Estimated VITA 3D-Master Shade")
    st.success(shade)

    if client:
        st.subheader(f"💬 GPT Explanation ({selected_model})")
        st.info(explain_shade_with_gpt(L, a, b, shade))

    report = pd.DataFrame([{
        "Timestamp": datetime.datetime.now().isoformat(),
        "L*": L,
        "a*": a,
        "b*": b,
        "Estimated Shade": shade,
    }])
    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV Report", csv, "shadegpt_report.csv", "text/csv")
