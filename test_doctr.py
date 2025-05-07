import os
import sys
import cv2
import numpy as np
from PIL import Image

def test_doctr(image_path):
    """
    Test DocTR OCR on a single image.
    
    Args:
        image_path (str): Path to the image file.
    """
    print(f"Testing DocTR on image: {image_path}")
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist.")
        return
    
    try:
        # Import doctr modules
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
        
        # Load the image using DocumentFile
        doc = DocumentFile.from_images([image_path])
        
        # Initialize the OCR predictor
        predictor = ocr_predictor(pretrained=True, det_arch='db_resnet50', reco_arch='crnn_vgg16_bn')
        
        # Run OCR prediction
        result = predictor(doc)
        
        # Extract and print text
        print("\nExtracted Text:")
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = ""
                    for word in line.words:
                        line_text += word.value + " "
                    print(line_text)
        
        print("\nOCR completed successfully!")
        
    except Exception as e:
        print(f"Error during OCR processing: {str(e)}")

if __name__ == "__main__":
    # Create a data/temp directory if it doesn't exist
    temp_dir = os.path.join("data", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a sample image with text if the sample diagram doesn't exist
    sample_path = os.path.join("data", "sample_diagrams", "microservice_architecture.png")
    
    if not os.path.exists(sample_path) or os.path.getsize(sample_path) == 0:
        print(f"Creating a sample image at {sample_path}")
        # Create a blank image
        img = np.ones((800, 1000, 3), np.uint8) * 255
        
        # Add some text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'User Interface', (100, 100), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'API Gateway', (100, 200), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Authentication Service', (100, 300), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Product Service', (400, 300), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Order Service', (700, 300), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'User DB', (100, 400), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Product DB', (400, 400), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Order DB', (700, 400), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Draw some lines to connect components
        cv2.line(img, (150, 120), (150, 180), (0, 0, 0), 2)
        cv2.line(img, (150, 220), (150, 280), (0, 0, 0), 2)
        cv2.line(img, (450, 220), (450, 280), (0, 0, 0), 2)
        cv2.line(img, (750, 220), (750, 280), (0, 0, 0), 2)
        cv2.line(img, (150, 320), (150, 380), (0, 0, 0), 2)
        cv2.line(img, (450, 320), (450, 380), (0, 0, 0), 2)
        cv2.line(img, (750, 320), (750, 380), (0, 0, 0), 2)
        
        # Save the image
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        cv2.imwrite(sample_path, img)
        print(f"Sample image created at {sample_path}")
    
    # Test DocTR on the sample diagram
    test_doctr(sample_path)
