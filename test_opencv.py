#!/usr/bin/env python3
"""
Test script to verify OpenCV works in headless environment
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import cv2
import numpy as np

def test_opencv():
    """Test basic OpenCV operations"""
    print("Testing OpenCV in headless environment...")
    
    # Create a test image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    print(f"✅ Created test image with shape: {test_img.shape}")
    
    # Test colormap application (this often triggers the GL error)
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    colorized = cv2.applyColorMap(gray, cv2.COLORMAP_PLASMA)
    print(f"✅ Applied colormap successfully, result shape: {colorized.shape}")
    
    # Test image encoding (for base64 conversion)
    _, buffer = cv2.imencode('.png', colorized)
    print(f"✅ Encoded image to PNG buffer, size: {len(buffer)} bytes")
    
    print("🎉 All OpenCV tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_opencv()
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        exit(1)
