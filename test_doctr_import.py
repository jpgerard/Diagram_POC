import streamlit as st

st.write("Testing DocTR import...")

try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    
    st.success("DocTR imported successfully!")
    
    # Try to load the model
    st.write("Loading OCR model...")
    predictor = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    st.success("OCR model loaded successfully!")
    
except ImportError as e:
    st.error(f"Error importing DocTR: {str(e)}")
except Exception as e:
    st.error(f"Error loading OCR model: {str(e)}")
