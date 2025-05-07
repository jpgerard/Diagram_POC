import streamlit as st
import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
import os
import time
import re

# Load the OCR model once (outside the function to avoid reloading)
@st.cache_resource
def load_ocr_model():
    # Initialize OCR predictor with correct parameters
    return ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

def extract_text(image):
    """
    Extract text from the uploaded diagram using OCR.
    
    Args:
        image (PIL.Image): The uploaded diagram image.
        
    Returns:
        dict: A dictionary containing extracted text and metadata.
    """
    st.write("### OCR Processing")
    
    # Display the original image
    st.image(image, caption="Original Diagram", use_column_width=True)
    
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Image preprocessing options
    st.write("#### Image Preprocessing Options")
    
    preprocessing_options = st.multiselect(
        "Select preprocessing techniques",
        ["Grayscale", "Thresholding", "Noise Reduction", "Edge Enhancement", "Technical Diagram Enhancement", "Minimal Processing"],
        default=["Minimal Processing"]
    )
    
    # Apply selected preprocessing techniques
    processed_img = img_cv.copy()
    
    if "Minimal Processing" in preprocessing_options:
        # Check if the image is already grayscale
        if len(processed_img.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        else:
            # Already grayscale
            gray = processed_img.copy()
        
        # Apply very light sharpening to enhance text edges without distortion
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        
        # Apply very mild contrast enhancement
        alpha = 1.1  # Contrast control (1.0 means no change)
        beta = 10    # Brightness control (0 means no change)
        enhanced = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
        
        processed_img = enhanced
    elif "Grayscale" in preprocessing_options:
        # Check if the image is already grayscale
        if len(processed_img.shape) == 3:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    
    if "Thresholding" in preprocessing_options:
        # Ensure we're working with a grayscale image
        if len(processed_img.shape) == 3:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding instead of global thresholding
        processed_img = cv2.adaptiveThreshold(
            processed_img, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11,  # Block size
            2    # Constant subtracted from mean
        )
    
    if "Noise Reduction" in preprocessing_options:
        processed_img = cv2.medianBlur(processed_img, 3)
    
    if "Edge Enhancement" in preprocessing_options:
        # Ensure we're working with a grayscale image for Canny edge detection
        if len(processed_img.shape) == 3:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_img.copy()
            
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Convert edges to 3-channel for visualization
        processed_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Convert back to grayscale for OCR
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    
    if "Technical Diagram Enhancement" in preprocessing_options:
        # This is a specialized preprocessing for technical diagrams with boxes and text
        # First ensure we're working with a grayscale image
        if len(processed_img.shape) == 3:
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_img.copy()
            
        # Apply morphological operations to enhance text
        # Create a kernel for morphological operations
        kernel = np.ones((2, 2), np.uint8)
        
        # Apply dilation to thicken text
        dilated = cv2.dilate(gray, kernel, iterations=1)
        
        # Apply erosion to remove noise
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # Apply sharpening to enhance text edges
        blur = cv2.GaussianBlur(eroded, (0, 0), 3)
        sharpened = cv2.addWeighted(eroded, 1.5, blur, -0.5, 0)
        
        # Apply contrast enhancement
        # Convert to float and normalize
        normalized = sharpened.astype(float) / 255.0
        # Apply power-law transformation (gamma correction)
        gamma = 0.7  # Values < 1 will enhance darker regions
        enhanced = np.power(normalized, gamma)
        # Convert back to uint8
        enhanced = (enhanced * 255).astype(np.uint8)
        
        # Special processing for hardware diagrams with boxes
        # Try to detect and enhance text in boxes
        # First detect rectangles/boxes
        contours, _ = cv2.findContours(enhanced.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask for the detected boxes
        box_mask = np.zeros_like(enhanced)
        
        # Draw the detected boxes on the mask
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If the polygon has 4 vertices (rectangle/square)
            if len(approx) == 4:
                # Get the bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)
                
                # Only consider boxes of reasonable size (not too small, not too large)
                if w > 30 and h > 20 and w < enhanced.shape[1] * 0.8 and h < enhanced.shape[0] * 0.8:
                    # Draw the box on the mask
                    cv2.rectangle(box_mask, (x, y), (x + w, y + h), 255, -1)
        
        # Invert the mask to get the text regions
        text_mask = cv2.bitwise_not(box_mask)
        
        # Apply the mask to the enhanced image
        box_enhanced = cv2.bitwise_and(enhanced, enhanced, mask=box_mask)
        
        # Further enhance the text in boxes
        # Apply a stronger contrast enhancement to the text in boxes
        box_normalized = box_enhanced.astype(float) / 255.0
        box_gamma = 0.5  # Even stronger enhancement for text in boxes
        box_enhanced = np.power(box_normalized, box_gamma)
        box_enhanced = (box_enhanced * 255).astype(np.uint8)
        
        # Combine the enhanced box regions with the original enhanced image
        processed_img = cv2.bitwise_or(box_enhanced, enhanced)
    
    # Display the processed image
    st.image(processed_img, caption="Processed Image for OCR", use_column_width=True)
    
    # Perform OCR processing with DocTR
    with st.spinner("Performing OCR with DocTR (this may take a moment)..."):
        try:
            # DocTR model options
            model_type = st.selectbox(
                "Select DocTR model type",
                [
                    "Default (crnn_vgg16_bn)",
                    "Fast (crnn_mobilenet_v3_small)",
                    "Accurate (crnn_resnet31)"
                ],
                index=0,
                help="Select the OCR model type. Default is a good balance, Fast is quicker but less accurate, Accurate is slower but more precise."
            )
            
            # Load the OCR model
            predictor = load_ocr_model()
            
            # Convert processed image to format DocTR expects
            if len(processed_img.shape) == 3:
                # If color image, convert from BGR to RGB
                processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            else:
                # If grayscale, convert to RGB
                processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(processed_img_rgb)
            
            # Create a temporary directory if it doesn't exist
            temp_dir = os.path.join("data", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save the PIL image to a temporary file with a unique name
            temp_path = os.path.join(temp_dir, f"temp_image_{int(time.time())}.png")
            pil_image.save(temp_path)
            
            # Load the image using DocTR's DocumentFile
            doc = DocumentFile.from_images([temp_path])
            
            # Run OCR prediction
            result = predictor(doc)
            
            # Extract text from the result
            extracted_text = ""
            confidence_scores = []
            
            # Process the DocTR result
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            extracted_text += word.value + " "
                            confidence_scores.append(word.confidence)
                        extracted_text += "\n"
            
            # Clean up the extracted text
            extracted_text = extracted_text.strip()
            
            # Store the raw OCR data
            ocr_data = result.export()
            
            # Check if any text was extracted
            if not extracted_text.strip():
                st.error("No text was extracted from the image. Try different preprocessing options or a clearer image.")
                return None
            
            # Display the extracted text
            st.write("#### Extracted Text")
            st.text_area("OCR Results", extracted_text, height=200)
            
            # Extract components from the text
            components = extract_components(extracted_text)
            
            # Create a dictionary with the extracted text and metadata
            ocr_results = {
                "text": extracted_text,
                "components": components,
                "confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
                "processing_options": preprocessing_options,
                "raw_detections": ocr_data
            }
                
        except Exception as e:
            st.error(f"Error during OCR processing: {str(e)}")
            st.error("Please check that all required dependencies are installed.")
            return None
        
        # Display the identified components
        st.write("#### Identified Components")
        for i, component in enumerate(ocr_results["components"]):
            st.write(f"Component {i+1}: {component['name']}")
            st.write(f"Type: {component['type']}")
            st.write(f"Description: {component['description']}")
            st.write("---")
        
        return ocr_results

def extract_components(text):
    """
    Extract components from the OCR text.
    
    Args:
        text (str): The OCR extracted text.
        
    Returns:
        list: A list of dictionaries containing component information.
    """
    components = []
    
    # Split the text into lines
    lines = text.strip().split('\n')
    
    # Common component types to look for
    component_types = {
        # Software components
        "ui": "Frontend",
        "interface": "Frontend",
        "frontend": "Frontend",
        "app": "Frontend",
        "application": "Frontend",
        "web": "Frontend",
        "mobile": "Frontend",
        
        "api": "Service",
        "service": "Service",
        "gateway": "Service",
        "server": "Service",
        "microservice": "Service",
        "controller": "Service",
        "handler": "Service",
        "processor": "Service",
        "manager": "Service",
        "auth": "Service",
        "authentication": "Service",
        
        "db": "Database",
        "database": "Database",
        "storage": "Database",
        "repository": "Database",
        "store": "Database",
        "cache": "Database",
        
        # Hardware components
        "board": "Hardware",
        "module": "Hardware",
        "chip": "Hardware",
        "processor": "Hardware",
        "mcu": "Hardware",
        "cpu": "Hardware",
        "gpu": "Hardware",
        "memory": "Hardware",
        "ram": "Hardware",
        "rom": "Hardware",
        "flash": "Hardware",
        "nand": "Hardware",
        "ddr": "Hardware",
        "connector": "Hardware",
        "camera": "Hardware",
        "sensor": "Hardware",
        "battery": "Hardware",
        "power": "Hardware",
        "audio": "Hardware",
        "codec": "Hardware",
        "wifi": "Hardware",
        "bluetooth": "Hardware",
        "gsm": "Hardware",
        "gps": "Hardware",
        "uart": "Hardware",
        "usb": "Hardware",
        "can": "Hardware",
        "spi": "Hardware",
        "i2c": "Hardware",
        "sd": "Hardware",
        "card": "Hardware",
        "keypad": "Hardware",
        "mic": "Hardware",
        "speaker": "Hardware",
        "sim": "Hardware",
        "ir": "Hardware",
        "rgb": "Hardware",
        "projector": "Hardware",
        "driver": "Hardware"
    }
    
    # Process each line to identify potential components
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip lines that are likely not component names
        if line.startswith('+') or line.startswith('|') or line.startswith('-') or line.startswith('='):
            continue
            
        # Try to determine component type
        component_type = "Unknown"
        component_name = line
        
        # Check if any known component type keywords are in the line
        for keyword, type_name in component_types.items():
            if keyword.lower() in line.lower():
                component_type = type_name
                break
                
        # Generate a description based on the component type
        description = ""
        if component_type == "Frontend":
            description = f"User interface component for {component_name}"
        elif component_type == "Service":
            description = f"Service component that processes {component_name} operations"
        elif component_type == "Database":
            description = f"Data storage for {component_name}"
        elif component_type == "Hardware":
            # More specific descriptions for common hardware components
            if "processor" in component_name.lower() or "mcu" in component_name.lower() or "cpu" in component_name.lower() or "imx" in component_name.lower():
                description = f"Processing unit: {component_name}"
            elif "memory" in component_name.lower() or "ram" in component_name.lower() or "rom" in component_name.lower() or "flash" in component_name.lower() or "ddr" in component_name.lower() or "nand" in component_name.lower():
                description = f"Memory component: {component_name}"
            elif "connector" in component_name.lower() or "usb" in component_name.lower() or "uart" in component_name.lower() or "can" in component_name.lower() or "spi" in component_name.lower() or "i2c" in component_name.lower():
                description = f"Interface connector: {component_name}"
            elif "camera" in component_name.lower() or "sensor" in component_name.lower():
                description = f"Input sensor: {component_name}"
            elif "battery" in component_name.lower() or "power" in component_name.lower():
                description = f"Power component: {component_name}"
            elif "audio" in component_name.lower() or "codec" in component_name.lower() or "speaker" in component_name.lower() or "mic" in component_name.lower():
                description = f"Audio component: {component_name}"
            elif "wifi" in component_name.lower() or "bluetooth" in component_name.lower() or "gsm" in component_name.lower() or "gps" in component_name.lower():
                description = f"Wireless communication component: {component_name}"
            else:
                description = f"Hardware component: {component_name}"
        else:
            description = f"Component identified in the diagram: {component_name}"
            
        # Add the component to the list
        components.append({
            "name": component_name,
            "type": component_type,
            "description": description
        })
    
    # If no components were found, return an empty list with a message
    if not components:
        st.warning("No components were automatically identified. The OCR text may need manual processing.")
        
    return components
