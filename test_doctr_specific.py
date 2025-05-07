import os
import sys
import cv2
import numpy as np
from PIL import Image

def test_doctr_specific(image_path):
    """
    Test DocTR OCR on a specific image.
    
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
        extracted_text = ""
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = ""
                    for word in line.words:
                        line_text += word.value + " "
                        print(f"Word: '{word.value}', Confidence: {word.confidence:.4f}")
                    print(line_text)
                    extracted_text += line_text + "\n"
        
        print("\nComplete Extracted Text:")
        print(extracted_text)
        
        print("\nOCR completed successfully!")
        
        # Try with preprocessing
        print("\n\nTrying with preprocessing...")
        
        # Load the image with OpenCV
        img = cv2.imread(image_path)
        
        # Apply preprocessing
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # Apply light sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(thresh, -1, kernel)
        
        # Save the preprocessed image
        preprocessed_path = image_path.replace('.png', '_preprocessed.png')
        cv2.imwrite(preprocessed_path, sharpened)
        print(f"Saved preprocessed image to {preprocessed_path}")
        
        # Run OCR on preprocessed image
        doc_preprocessed = DocumentFile.from_images([preprocessed_path])
        result_preprocessed = predictor(doc_preprocessed)
        
        # Extract and print text from preprocessed image
        print("\nExtracted Text from Preprocessed Image:")
        extracted_text_preprocessed = ""
        for page in result_preprocessed.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = ""
                    for word in line.words:
                        line_text += word.value + " "
                        print(f"Word: '{word.value}', Confidence: {word.confidence:.4f}")
                    print(line_text)
                    extracted_text_preprocessed += line_text + "\n"
        
        print("\nComplete Extracted Text from Preprocessed Image:")
        print(extracted_text_preprocessed)
        
        print("\nPreprocessed OCR completed successfully!")
        
    except Exception as e:
        print(f"Error during OCR processing: {str(e)}")

if __name__ == "__main__":
    # Use the provided image path or default to a sample
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "D:\\Users\\jp\\Downloads\\car-hud-decoding-hardware.png"
    
    # Test DocTR on the specific image
    test_doctr_specific(image_path)
