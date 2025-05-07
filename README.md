# Diagram to Requirements Converter

A prototype application that demonstrates the full flow from technical diagrams to traceable system requirements with an editable review UI.

## Overview

This application allows users to:

1. Upload a technical diagram
2. Extract text via OCR
3. Build a simulated graph of components
4. Use GPT-4 to generate traceable system requirements
5. Edit and export those requirements in JSON format

## Technology Stack

- **Python**: Primary programming language
- **Streamlit**: Web application framework
- **OpenAI API**: For GPT-4 integration
- **OCR**: For text extraction from diagrams
- **NetworkX**: For graph construction and visualization

## Project Structure

```
Diagram POC/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
├── README.md               # Project documentation
├── cline_docs/             # Project documentation
│   ├── activeContext.md    # Current work context
│   ├── productContext.md   # Product overview
│   ├── progress.md         # Progress tracking
│   ├── systemPatterns.md   # System architecture
│   └── techContext.md      # Technical details
├── data/                   # Data directory
│   ├── output/             # Exported requirements
│   └── sample_diagrams/    # Sample diagrams for testing
└── src/                    # Source code
    ├── modules/            # Application modules
    │   ├── diagram_upload/ # Diagram upload functionality
    │   ├── ocr_processing/ # OCR text extraction
    │   ├── graph_construction/ # Graph building
    │   ├── requirement_generation/ # Requirement generation
    │   ├── review_edit/    # Requirement editing
    │   └── export/         # Export functionality
```

## Setup and Installation

### Local Development

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set up your OpenAI API key in the `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

5. Run the application:
   ```
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your forked repository, branch, and app.py file
6. Add your OpenAI API key as a secret:
   - In the app settings, add a secret with the name `OPENAI_API_KEY` and your API key as the value
7. Deploy the app
8. Your app will be available at a URL like `https://username-app-name-streamlit-app.streamlit.app`

**Note**: The `.streamlit/config.toml` file is already configured for Streamlit Cloud deployment.

## Usage

1. Upload a technical diagram (PNG, JPG, or PDF)
2. Select OCR processing options
3. Review the extracted text and identified components
4. Visualize the graph representation
5. Generate requirements using GPT-4
6. Edit and refine the generated requirements
7. Export the requirements in JSON format

## Features

- **Interactive UI**: Step-by-step workflow with intuitive navigation
- **OCR Processing**: Extract text from diagrams with preprocessing options
- **Graph Visualization**: Interactive graph representation of components
- **AI-Powered Requirements**: Generate requirements based on diagram analysis
- **Requirement Editing**: Review and edit requirements with bulk or individual editing
- **Export Options**: Export requirements in JSON or CSV format with filtering options

## Notes

This is a prototype application designed to demonstrate the concept of converting technical diagrams to system requirements. In a production environment, additional features and optimizations would be implemented.
