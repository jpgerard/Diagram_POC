import streamlit as st
import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
import os
import time
import re
import pandas as pd

# Load the OCR model once (outside the function to avoid reloading)
@st.cache_resource
def load_ocr_model():
    # Initialize OCR predictor with correct parameters
    return ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

def apply_technical_diagram_preprocessing(img_cv):
    """
    Apply specialized preprocessing for technical diagrams to improve OCR accuracy.
    
    Args:
        img_cv (numpy.ndarray): The input image in OpenCV format.
        
    Returns:
        numpy.ndarray: The preprocessed image.
    """
    # Convert to grayscale if the image is in color
    if len(img_cv.shape) == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(clahe_img, 9, 75, 75)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        bilateral, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11,  # Block size
        2    # Constant subtracted from mean
    )
    
    # Apply morphological operations to enhance text
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    # Apply sharpening to enhance text edges
    kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(eroded, -1, kernel_sharp)
    
    return sharpened

def apply_alternative_preprocessing(img_cv):
    """
    Apply alternative preprocessing that worked well in tests.
    
    Args:
        img_cv (numpy.ndarray): The input image in OpenCV format.
        
    Returns:
        numpy.ndarray: The preprocessed image.
    """
    # Convert to grayscale if the image is in color
    if len(img_cv.shape) == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    
    # Apply Otsu's thresholding
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opening

def extract_text_enhanced(image):
    """
    Extract text from the uploaded diagram using enhanced OCR with DocTR.
    
    Args:
        image (PIL.Image): The uploaded diagram image.
        
    Returns:
        dict: A dictionary containing extracted text and metadata.
    """
    st.write("### Enhanced OCR Processing with DocTR")
    
    # Display the original image
    st.image(image, caption="Original Diagram", use_column_width=True)
    
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Image preprocessing options
    st.write("#### Image Preprocessing Options")
    
    preprocessing_options = st.selectbox(
        "Select preprocessing technique",
        ["Standard", "Technical Diagram Enhancement", "Alternative Enhancement", "No Preprocessing"],
        index=1,
        help="Select the preprocessing technique to apply before OCR. Technical Diagram Enhancement works best for most diagrams."
    )
    
    # Apply selected preprocessing technique
    if preprocessing_options == "Standard":
        # Convert to grayscale
        if len(img_cv.shape) == 3:
            processed_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            processed_img = img_cv.copy()
        
        # Apply adaptive thresholding
        processed_img = cv2.adaptiveThreshold(
            processed_img, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11,  # Block size
            2    # Constant subtracted from mean
        )
    elif preprocessing_options == "Technical Diagram Enhancement":
        processed_img = apply_technical_diagram_preprocessing(img_cv)
    elif preprocessing_options == "Alternative Enhancement":
        processed_img = apply_alternative_preprocessing(img_cv)
    else:  # No Preprocessing
        processed_img = img_cv.copy()
    
    # Display the processed image
    st.image(processed_img, caption="Processed Image for OCR", use_column_width=True)
    
    # Perform OCR processing with DocTR
    with st.spinner("Performing OCR with DocTR (this may take a moment)..."):
        try:
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
            word_details = []
            
            # Process the DocTR result
            for page_idx, page in enumerate(result.pages):
                for block_idx, block in enumerate(page.blocks):
                    for line_idx, line in enumerate(block.lines):
                        line_text = ""
                        for word_idx, word in enumerate(line.words):
                            line_text += word.value + " "
                            confidence_scores.append(word.confidence)
                            word_details.append({
                                "value": word.value,
                                "confidence": word.confidence,
                                "page_idx": page_idx,
                                "block_idx": block_idx,
                                "line_idx": line_idx,
                                "word_idx": word_idx
                            })
                        extracted_text += line_text.strip() + "\n"
            
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
            
            # Display confidence information
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            st.write(f"Average confidence score: {avg_confidence:.2f}")
            
            # Display detailed word information
            st.write("#### Word Details")
            word_df = pd.DataFrame(word_details)
            st.dataframe(word_df)
            
            # Extract components from the text
            components = extract_components(extracted_text)
            
            # Create a dictionary with the extracted text and metadata
            ocr_results = {
                "text": extracted_text,
                "components": components,
                "confidence": avg_confidence,
                "processing_option": preprocessing_options,
                "word_details": word_details,
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
