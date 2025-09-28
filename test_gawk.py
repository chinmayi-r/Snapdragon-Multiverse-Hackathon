"""
Test script for Gawk application
Tests basic functionality and dependencies
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("✓ MediaPipe imported successfully")
    except ImportError as e:
        print(f"✗ MediaPipe import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from pdf2image import convert_from_path
        print("✓ pdf2image imported successfully")
    except ImportError as e:
        print(f"✗ pdf2image import failed: {e}")
        return False
    
    try:
        import PyPDF2
        print("✓ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"✗ PyPDF2 import failed: {e}")
        return False
    
    try:
        import mss
        print("✓ MSS imported successfully")
    except ImportError as e:
        print(f"✗ MSS import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_webcam():
    """Test webcam access"""
    print("\nTesting webcam access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ Webcam accessible and working")
                cap.release()
                return True
            else:
                print("✗ Webcam opened but cannot read frames")
                cap.release()
                return False
        else:
            print("✗ Cannot open webcam")
            return False
    except Exception as e:
        print(f"✗ Webcam test failed: {e}")
        return False

def test_screen_capture():
    """Test screen capture functionality"""
    print("\nTesting screen capture...")
    
    try:
        import mss
        with mss.mss() as sct:
            # Get primary monitor
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            if screenshot:
                print("✓ Screen capture working")
                return True
            else:
                print("✗ Screen capture failed")
                return False
    except Exception as e:
        print(f"✗ Screen capture test failed: {e}")
        return False

def test_face_detection():
    """Test MediaPipe face detection"""
    print("\nTesting face detection...")
    
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
        
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = face_detection.process(dummy_image)
        
        print("✓ Face detection initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Face detection test failed: {e}")
        return False

def test_file_creation():
    """Test if we can create necessary directories and files"""
    print("\nTesting file creation...")
    
    try:
        # Create data directory
        os.makedirs("gawk_data", exist_ok=True)
        os.makedirs("gawk_data/videos", exist_ok=True)
        os.makedirs("gawk_data/pdfs", exist_ok=True)
        os.makedirs("gawk_data/metadata", exist_ok=True)
        os.makedirs("gawk_data/pdf_pages", exist_ok=True)
        os.makedirs("gawk_data/heatmaps", exist_ok=True)
        
        print("✓ Data directories created successfully")
        return True
    except Exception as e:
        print(f"✗ File creation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Gawk Application Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_webcam,
        test_screen_capture,
        test_face_detection,
        test_file_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Gawk application is ready to use.")
        return True
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
