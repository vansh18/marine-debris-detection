import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Marine Debris Detection",
    page_icon="🌊",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    st.session_state.model = None

def load_model():
    """Load the YOLOv8 model"""
    try:
        model_path = '../src/models/runs/segment/train23/weights/best.pt'
        model = YOLO(model_path)  # Load the model
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Custom preprocessing function placeholder
def preprocess_image(image):
    """
    Preprocess the image by removing text, detecting edges, and overlaying them
    
    Args:
        image: PIL Image object
    Returns:
        processed_image: numpy array of processed image
    """
    # Convert PIL Image to numpy array (BGR format for OpenCV)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create a mask for the text using thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Inpaint the image to remove text
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    
    # Overlay the edges onto the inpainted image
    overlay_image = inpainted_image.copy()
    overlay_image[edges > 0] = [0, 255, 0]  # Highlight edges in green
    
    # Convert back to RGB for display
    overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
    
    return overlay_image

def process_image(image, model):
    """Process the image and return the results"""
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Run inference for segmentation using predict with mode='segment'
        results = model.predict(source=processed_image, conf=0.25, mode='segment', save=False)
        return results
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.error(f"Full error details: {str(type(e).__name__)}")
        return None

def draw_predictions(image, results):
    """Draw segmentation masks on the image"""
    try:
        # For segmentation, we use the masks plotting
        annotated_frame = results[0].plot(boxes=True, masks=True, probs=True)
        return Image.fromarray(annotated_frame)
    except Exception as e:
        st.error(f"Error drawing predictions: {str(e)}")
        return image

def main():
    st.title("Marine Debris Detection System 🌊")
    st.write("Upload an image to detect marine debris")

    # Load model on first run
    if st.session_state.model is None:
        with st.spinner("Loading model..."):
            st.session_state.model = load_model()
        if st.session_state.model is not None:
            st.success("Model loaded successfully!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add preprocessing view
            st.subheader("Preprocessed Image")
            preprocessed_image = preprocess_image(image)
            st.image(preprocessed_image, caption="Preprocessed Image", use_column_width=True)

        # Process image button
        if st.button("Detect Debris"):
            if st.session_state.model is not None:
                with st.spinner("Processing image..."):
                    # Process the image
                    results = process_image(image, st.session_state.model)
                    
                    if results is not None:
                        # Display results
                        with col2:
                            st.subheader("Detected Debris")
                            result_image = draw_predictions(image, results)
                            st.image(result_image, caption="Detected Debris", use_column_width=True)
                            
                            # Display detection details
                            st.subheader("Detection Details")
                            for r in results:
                                for box in r.boxes:
                                    confidence = float(box.conf[0])
                                    class_id = int(box.cls[0])
                                    class_name = st.session_state.model.names[class_id]
                                    st.write(f"- {class_name}: {confidence:.2%} confidence")
            else:
                st.error("Please ensure the model is loaded correctly")

if __name__ == "__main__":
    main()