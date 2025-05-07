import streamlit as st
import os
import time
from dotenv import load_dotenv
import json

# Import modules
from src.modules.diagram_upload.upload import process_upload
from src.modules.diagram_upload.direct_upload import process_direct_upload
from src.modules.ocr_processing.ocr import extract_text
from src.modules.ocr_processing.enhanced_ocr import extract_text_enhanced
from src.modules.ocr_processing.streamlined_ocr import streamlined_ocr_process
from src.modules.ocr_processing.direct_ocr import process_direct_ocr
from src.modules.graph_construction.graph_builder import build_graph
from src.modules.requirement_generation.generator import generate_requirements
from src.modules.review_edit.editor import display_requirements_editor
from src.modules.export.exporter import export_to_json

# Load environment variables
load_dotenv()

# For Streamlit Cloud: Use secrets if available
try:
    # Check if we're running on Streamlit Cloud
    if 'openai' in st.secrets:
        os.environ['OPENAI_API_KEY'] = st.secrets['openai']['api_key']
        st.sidebar.success("API key loaded from Streamlit secrets")
except Exception as e:
    st.sidebar.warning(f"Note: {str(e)}")

# Set page configuration
st.set_page_config(
    page_title="Diagram to Requirements",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the application state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'uploaded_image_path' not in st.session_state:
    st.session_state.uploaded_image_path = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'requirements' not in st.session_state:
    st.session_state.requirements = None
if 'edited_requirements' not in st.session_state:
    st.session_state.edited_requirements = None

# Function to navigate between steps
def set_step(step):
    st.session_state.current_step = step

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("## Process Steps")

steps = [
    "1. Upload Diagram",
    "2. OCR Processing",
    "3. Graph Construction",
    "4. Requirement Generation",
    "5. Review & Edit",
    "6. Export"
]

for i, step in enumerate(steps, 1):
    if st.sidebar.button(step, key=f"nav_{i}"):
        # Only allow navigation to completed steps or the next step
        if i <= st.session_state.current_step:
            set_step(i)

# Display progress
st.sidebar.progress(st.session_state.current_step / len(steps))

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("## About")
st.sidebar.info(
    "This application converts technical diagrams into system requirements "
    "using OCR, graph analysis, and AI-powered requirement generation."
)

# Main content
st.title("Diagram to Requirements Converter")

# Step 1: Upload Diagram
if st.session_state.current_step == 1:
    st.header("Step 1: Upload Technical Diagram")
    st.markdown(
        "Upload a technical diagram (architecture diagram, flowchart, etc.) "
        "to begin the conversion process."
    )
    
    # Use Direct Upload only
    st.info("Using Direct Upload for better OCR results.")
    uploaded_image, uploaded_path = process_direct_upload()
    if uploaded_image is not None and uploaded_path is not None:
        st.session_state.uploaded_image = uploaded_image
        st.session_state.uploaded_image_path = uploaded_path
        if st.button("Proceed to OCR Processing"):
            set_step(2)

# Step 2: OCR Processing
elif st.session_state.current_step == 2:
    st.header("Step 2: OCR Processing")
    st.markdown(
        "Extracting text from the diagram using Optical Character Recognition."
    )
    
    if st.session_state.uploaded_image is not None and st.session_state.uploaded_image_path is not None:
        # Use Direct OCR only
        st.info("Using Direct OCR for best results.")
        extracted_text = process_direct_ocr(st.session_state.uploaded_image_path)
            
        st.session_state.extracted_text = extracted_text
        
        if st.button("Proceed to Graph Construction"):
            set_step(3)
    else:
        st.error("Please upload a diagram first.")
        if st.button("Go back to Upload"):
            set_step(1)

# Step 3: Graph Construction
elif st.session_state.current_step == 3:
    st.header("Step 3: Graph Construction")
    st.markdown(
        "Building a graph representation of the components identified in the diagram."
    )
    
    if st.session_state.extracted_text is not None:
        graph = build_graph(st.session_state.extracted_text)
        st.session_state.graph = graph
        
        if st.button("Proceed to Requirement Generation"):
            set_step(4)
    else:
        st.error("Please complete OCR processing first.")
        if st.button("Go back to OCR Processing"):
            set_step(2)

# Step 4: Requirement Generation
elif st.session_state.current_step == 4:
    st.header("Step 4: Requirement Generation")
    st.markdown(
        "Generating system requirements based on the graph analysis using GPT-4."
    )
    
    if st.session_state.graph is not None:
        requirements = generate_requirements(st.session_state.graph)
        st.session_state.requirements = requirements
        
        if st.button("Proceed to Review & Edit"):
            set_step(5)
    else:
        st.error("Please complete graph construction first.")
        if st.button("Go back to Graph Construction"):
            set_step(3)

# Step 5: Review & Edit
elif st.session_state.current_step == 5:
    st.header("Step 5: Review & Edit Requirements")
    st.markdown(
        "Review and edit the generated requirements before exporting."
    )
    
    if st.session_state.requirements is not None:
        edited_requirements = display_requirements_editor(st.session_state.requirements)
        st.session_state.edited_requirements = edited_requirements
        
        if st.button("Proceed to Export"):
            set_step(6)
    else:
        st.error("Please complete requirement generation first.")
        if st.button("Go back to Requirement Generation"):
            set_step(4)

# Step 6: Export
elif st.session_state.current_step == 6:
    st.header("Step 6: Export Requirements")
    st.markdown(
        "Export the final requirements to JSON format."
    )
    
    if st.session_state.edited_requirements is not None:
        export_to_json(st.session_state.edited_requirements)
    else:
        st.error("Please complete the review & edit step first.")
        if st.button("Go back to Review & Edit"):
            set_step(5)

# Footer
st.markdown("---")
st.markdown(
    "Diagram to Requirements Converter | Prototype Version"
)
