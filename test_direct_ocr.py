import os
import sys
import cv2
import numpy as np
from PIL import Image

def test_direct_ocr(image_path):
    """
    Test the direct OCR approach on a specific image.
    This script mimics the exact steps used in the direct_ocr.py module
    but without the Streamlit UI components.
    
    Args:
        image_path (str): Path to the image file.
    """
    print(f"Testing Direct OCR on image: {image_path}")
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist.")
        return
    
    try:
        # Import doctr modules
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
        
        # Initialize the OCR predictor
        print("Loading OCR model...")
        predictor = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        print("OCR model loaded successfully.")
        
        # Load the image with OpenCV (exactly as in the direct_ocr.py module)
        img = cv2.imread(image_path)
        
        # Test all preprocessing methods
        preprocessing_methods = ["Original", "Enhanced", "Alternative"]
        
        for method in preprocessing_methods:
            print(f"\n\nTesting with preprocessing method: {method}")
            
            if method == "Original":
                # Use the original image without preprocessing
                processed_path = image_path
                processed_img = img
            
            elif method == "Enhanced":
                # Apply the enhanced preprocessing
                
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
                processed_img = (enhanced * 255).astype(np.uint8)
                
                # Save the preprocessed image
                processed_path = image_path.replace('.png', f'_direct_enhanced.png')
                cv2.imwrite(processed_path, processed_img)
            
            else:  # Alternative
                # Apply the alternative preprocessing
                
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
                processed_img = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=1)
                
                # Save the preprocessed image
                processed_path = image_path.replace('.png', f'_direct_alternative.png')
                cv2.imwrite(processed_path, processed_img)
            
            print(f"Saved preprocessed image to {processed_path}")
            
            # Perform OCR
            print("Performing OCR...")
            
            # Load the processed image using DocumentFile
            doc = DocumentFile.from_images([processed_path])
            
            # Run OCR prediction
            result = predictor(doc)
            
            # Extract text from the result
            extracted_text = ""
            confidence_scores = []
            
            # Process the DocTR result
            for page_idx, page in enumerate(result.pages):
                for block_idx, block in enumerate(page.blocks):
                    for line_idx, line in enumerate(block.lines):
                        line_text = ""
                        for word_idx, word in enumerate(line.words):
                            line_text += word.value + " "
                            confidence_scores.append(word.confidence)
                            print(f"Word: '{word.value}', Confidence: {word.confidence:.4f}")
                        print(line_text)
                        extracted_text += line_text.strip() + "\n"
            
            # Clean up the extracted text
            extracted_text = extracted_text.strip()
            
            # Calculate average confidence
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            # Print results
            print(f"\nAverage confidence score: {avg_confidence:.4f}")
            print("\nExtracted Text:")
            print(extracted_text)
            
            print(f"\nPreprocessing method '{method}' OCR completed successfully!")
        
    except Exception as e:
        print(f"Error during OCR processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Use the provided image path or default to a sample
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use the sample diagram from the project
        image_path = os.path.join("data", "sample_diagrams", "microservice_architecture.png")
        
        # If the sample doesn't exist, create it
        if not os.path.exists(image_path):
            print(f"Creating a sample image at {image_path}")
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
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            cv2.imwrite(image_path, img)
            print(f"Sample image created at {image_path}")
    
    # Test the direct OCR on the image
    test_direct_ocr(image_path)
