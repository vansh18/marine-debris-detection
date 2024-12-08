import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import time

st.set_page_config(layout="wide")
st.title("Marine Debris Detection")
#preprocessing func
def preprocess_image(image):
    image_array = np.array(image)
    image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    gray_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    
    overlay_image = inpainted_image.copy()
    overlay_image[edges > 0] = [0, 255, 0]
    
    return cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)

@st.cache_resource
def load_model():
    model = YOLO('../src/models/runs/segment/train23/weights/best.pt')
    return model

model = load_model()
if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = False
if 'preprocessed_img' not in st.session_state:
    st.session_state.preprocessed_img = None
if 'preprocess_time' not in st.session_state:
    st.session_state.preprocess_time = None

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.preprocessed = False
        st.session_state.preprocessed_img = None
        st.session_state.current_file = uploaded_file.name
    
    col1, col2, col3 = st.columns(3)

    image = Image.open(uploaded_file)
    
    with col1:
        st.write("Original Image")
        st.image(image, use_column_width=True)

    if not st.session_state.preprocessed:
        if st.button("1. Preprocess Image"):
            preprocess_start = time.time()
            with st.spinner("Preprocessing..."):
                preprocessed_img = preprocess_image(image)
                st.session_state.preprocessed_img = preprocessed_img
                st.session_state.preprocess_time = time.time() - preprocess_start
                st.session_state.preprocessed = True
                st.experimental_rerun()
    
    with col2:
        st.write("Preprocessed Image")
        if st.session_state.preprocessed_img is not None:
            st.image(st.session_state.preprocessed_img, use_column_width=True)
            st.write(f"Preprocessing Time: {st.session_state.preprocess_time:.3f} seconds")
    
    if st.session_state.preprocessed:
        if st.button("2. Detect Debris"):
            try:
                detect_start = time.time()
                with st.spinner("Detecting..."):
                    results = model(np.array(image))
                detect_time = time.time() - detect_start

                with col3:
                    st.write("Detection Results")
                    for r in results:
                        boxes = r.boxes
                        plotted_img = np.array(image).copy()
                        
                        detections = []
                        
                        if len(boxes) > 0:
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                class_name = model.names[cls]
                                
                                detections.append({
                                    'class': class_name,
                                    'confidence': conf
                                })
                                
                                cv2.rectangle(plotted_img, 
                                            (int(x1), int(y1)), 
                                            (int(x2), int(y2)), 
                                            (0, 255, 0), 3)
                                
                                label = f"{class_name} {conf:.2f}"
                                cv2.putText(plotted_img, label, 
                                          (int(x1), int(y1) - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 
                                          1.0, 
                                          (0, 255, 0), 
                                          3)
                            
                            st.image(plotted_img, use_column_width=True)
                            
                            st.write("### Detection Summary")
                            st.write(f"Processing Time: {detect_time:.3f} seconds")
                            st.write(f"Number of instances = {len(boxes)}")
                            
                            classes = [f"class{i+1} = {det['class']} ({det['confidence']:.2%})" 
                                     for i, det in enumerate(detections)]
                            st.write(f"Classes = {', '.join(classes)}")
                        else:
                            st.image(plotted_img, use_column_width=True)
                            st.write("No instances detected")
                            st.write(f"Processing Time: {detect_time:.3f} seconds")
                            
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.error(f"Error type: {type(e).__name__}")
    else:
        st.write("Please preprocess the image first before detection.")