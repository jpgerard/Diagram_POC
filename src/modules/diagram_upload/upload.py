import streamlit as st
from PIL import Image
import os
import io

def process_upload():
    """
    Process the uploaded diagram image.
    
    Returns:
        PIL.Image or None: The uploaded image if successful, None otherwise.
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
            st.image(image, caption="Uploaded Diagram", use_column_width=True)
            
            # Save the image to a temporary file for processing
            temp_dir = os.path.join("data", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            st.success(f"Image uploaded successfully and saved to {temp_path}")
            
            return image
            
        except Exception as e:
            st.error(f"Error processing the image: {str(e)}")
            return None
    
    # For demonstration purposes, add a sample image option
    if st.checkbox("Use a sample diagram instead"):
        st.info("Using a sample diagram for demonstration purposes.")
        
        # In a real application, you would load a sample image from the data directory
        # For this prototype, we'll create a simple placeholder image
        sample_image = Image.new('RGB', (800, 600), color='white')
        st.image(sample_image, caption="Sample Diagram", use_column_width=True)
        
        # Save the sample image
        temp_dir = os.path.join("data", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "sample_diagram.png")
        sample_image.save(temp_path)
        
        st.success(f"Sample image created and saved to {temp_path}")
        
        return sample_image
    
    return None
