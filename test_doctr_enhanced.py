import os
import sys
import cv2
import numpy as np
from PIL import Image

def test_doctr_enhanced(image_path):
    """
    Test DocTR OCR on a specific image with enhanced preprocessing.
    
    Args:
        image_path (str): Path to the image file.
    """
    print(f"Testing DocTR with enhanced preprocessing on image: {image_path}")
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist.")
        return
    
    try:
        # Import doctr modules
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
        
        # Initialize the OCR predictor
        predictor = ocr_predictor(pretrained=True, det_arch='db_resnet50', reco_arch='crnn_vgg16_bn')
        
        # Load the image with OpenCV
        img = cv2.imread(image_path)
        
        # Apply enhanced preprocessing specifically for technical diagrams
        
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 3. Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            bilateral, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # 4. Apply morphological operations to enhance text
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # 5. Apply sharpening to enhance text edges
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(eroded, -1, kernel_sharp)
        
        # 6. Apply contrast enhancement
        # Convert to float and normalize
        normalized = sharpened.astype(float) / 255.0
        # Apply power-law transformation (gamma correction)
        gamma = 0.8  # Values < 1 will enhance darker regions
        enhanced = np.power(normalized, gamma)
        # Convert back to uint8
        enhanced = (enhanced * 255).astype(np.uint8)
        
        # Save the preprocessed image
        preprocessed_path = image_path.replace('.png', '_enhanced.png')
        cv2.imwrite(preprocessed_path, enhanced)
        print(f"Saved enhanced preprocessed image to {preprocessed_path}")
        
        # Run OCR on preprocessed image
        doc_preprocessed = DocumentFile.from_images([preprocessed_path])
        result_preprocessed = predictor(doc_preprocessed)
        
        # Extract and print text from preprocessed image
        print("\nExtracted Text from Enhanced Preprocessed Image:")
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
        
        print("\nComplete Extracted Text from Enhanced Preprocessed Image:")
        print(extracted_text_preprocessed)
        
        print("\nEnhanced Preprocessed OCR completed successfully!")
        
        # Try another preprocessing approach
        print("\n\nTrying with alternative preprocessing approach...")
        
        # Load the image with OpenCV
        img = cv2.imread(image_path)
        
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(gray)
        
        # 3. Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)
        
        # 4. Apply Otsu's thresholding
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. Apply morphological operations
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Save the alternative preprocessed image
        alt_preprocessed_path = image_path.replace('.png', '_alt_enhanced.png')
        cv2.imwrite(alt_preprocessed_path, opening)
        print(f"Saved alternative enhanced preprocessed image to {alt_preprocessed_path}")
        
        # Run OCR on alternative preprocessed image
        doc_alt_preprocessed = DocumentFile.from_images([alt_preprocessed_path])
        result_alt_preprocessed = predictor(doc_alt_preprocessed)
        
        # Extract and print text from alternative preprocessed image
        print("\nExtracted Text from Alternative Enhanced Preprocessed Image:")
        extracted_text_alt_preprocessed = ""
        for page in result_alt_preprocessed.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = ""
                    for word in line.words:
                        line_text += word.value + " "
                        print(f"Word: '{word.value}', Confidence: {word.confidence:.4f}")
                    print(line_text)
                    extracted_text_alt_preprocessed += line_text + "\n"
        
        print("\nComplete Extracted Text from Alternative Enhanced Preprocessed Image:")
        print(extracted_text_alt_preprocessed)
        
        print("\nAlternative Enhanced Preprocessed OCR completed successfully!")
        
    except Exception as e:
        print(f"Error during OCR processing: {str(e)}")

if __name__ == "__main__":
    # Use the provided image path or default to a sample
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "D:\\Users\\jp\\Downloads\\car-hud-decoding-hardware.png"
    
    # Test DocTR on the specific image with enhanced preprocessing
    test_doctr_enhanced(image_path)
