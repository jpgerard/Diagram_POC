import streamlit as st
from PIL import Image
import os
import io
import time

def process_direct_upload():
    """
    Process the uploaded diagram image and save it to disk.
    This function is designed to work with the direct OCR approach
    by saving the image to disk and returning the path.
    
    Returns:
        tuple: (PIL.Image, str) The uploaded image and its path if successful, (None, None) otherwise.
    """
    # Create a file uploader widget
    uploaded_file = st.file_uploader(
        "Choose a diagram image file", 
        type=["png", "jpg", "jpeg", "pdf"],
        help="Upload a technical diagram image (PNG, JPG, or PDF)"
    )
    
    if uploaded_file is not None:
        # Display file details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        st.write("File Details:")
        for key, value in file_details.items():
            st.write(f"- {key}: {value}")
        
        try:
            # Open the image
            image = Image.open(uploaded_file)
            
            # Display the image
            st.image(image, caption="Uploaded Diagram", width=None)
            
            # Create a unique filename to avoid overwriting
            timestamp = int(time.time())
            filename = f"{timestamp}_{uploaded_file.name}"
            
            # Save the image to a temporary file for processing
            temp_dir = os.path.join("data", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, filename)
            
            # Save the image to disk
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            st.success(f"Image uploaded successfully and saved to {temp_path}")
            
            return image, temp_path
            
        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")
            return None, None
    
    # For demonstration purposes, add a sample image option
    if st.checkbox("Use a sample diagram instead"):
        st.info("Using a sample diagram for demonstration purposes.")
        
        # Use the sample diagram from the project
        sample_path = os.path.join("data", "sample_diagrams", "car-hud-decoding-hardware_preprocessed.png")
        
        # Check if the sample exists
        if os.path.exists(sample_path):
            # Load the sample image
            sample_image = Image.open(sample_path)
            st.image(sample_image, caption="Sample Diagram", width=None)
            st.success(f"Using sample image from {sample_path}")
            return sample_image, sample_path
        else:
            # Create a simple placeholder image
            sample_image = Image.new('RGB', (800, 600), color='white')
            st.image(sample_image, caption="Sample Diagram", width=None)
            
            # Save the sample image
            temp_dir = os.path.join("data", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"sample_diagram_{int(time.time())}.png")
            sample_image.save(temp_path)
            
            st.success(f"Sample image created and saved to {temp_path}")
            
            return sample_image, temp_path
    
    return None, None
