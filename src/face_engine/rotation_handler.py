import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from deepface import DeepFace
import tempfile
import os

class RotationHandler:
    """
    Handles image rotation to find the best orientation for face detection.
    Tries multiple angles and selects the one with highest face detection confidence.
    """
    
    def __init__(self, angles=None, detector_backend='retinaface'):
        """
        Args:
            angles: List of angles to try (in degrees). Default: every 30 degrees
            detector_backend: Face detector to use for confidence scoring
        """
        self.angles = angles or [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        self.detector_backend = detector_backend
        self.temp_dir = tempfile.mkdtemp(prefix="rotation_")
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by specified angle (in degrees) around center.
        Handles edge cases like padding to prevent cropping.
        
        Args:
            image: Input image as numpy array
            angle: Rotation angle in degrees (positive = counter-clockwise)
            
        Returns:
            Rotated image
        """
        if angle == 0:
            return image
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding dimensions to prevent cropping
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Perform rotation with white background
        rotated = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (new_width, new_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)  # White background
        )
        
        return rotated
    
    def detect_face_with_confidence(self, image_path: str) -> Tuple[float, int]:
        """
        Detect face and return confidence score and number of faces.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (max_confidence, num_faces)
        """
        try:
            # Use DeepFace to detect faces
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=False  # Don't align, just detect
            )
            
            if not faces:
                return 0.0, 0
            
            # Get maximum confidence from all detected faces
            max_confidence = max(face.get('confidence', 0) for face in faces)
            num_faces = len(faces)
            
            return max_confidence, num_faces
            
        except Exception as e:
            print(f"Error detecting face: {e}")
            return 0.0, 0
    
    def find_best_rotation(self, image_path: str) -> Dict:
        """
        Try multiple rotation angles and find the one with best face detection.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with:
                - best_angle: Angle with highest confidence
                - best_confidence: Confidence score at best angle
                - best_image_path: Path to rotated image (or original if angle=0)
                - num_faces: Number of faces detected
                - all_results: List of all angle results for debugging
        """
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return {
                'best_angle': 0,
                'best_confidence': 0.0,
                'best_image_path': image_path,
                'num_faces': 0,
                'all_results': []
            }
        
        results = []
        best_angle = 0
        best_confidence = 0.0
        best_image_path = image_path
        best_num_faces = 0
        
        for angle in self.angles:
            try:
                # Rotate image
                rotated = self.rotate_image(image, angle)
                
                # Save rotated image temporarily
                temp_path = os.path.join(self.temp_dir, f"rotated_{angle}.jpg")
                cv2.imwrite(temp_path, rotated)
                
                # Detect faces and get confidence
                confidence, num_faces = self.detect_face_with_confidence(temp_path)
                
                results.append({
                    'angle': angle,
                    'confidence': confidence,
                    'num_faces': num_faces
                })
                
                # Update best if this is better
                # Prioritize: 1) Having faces, 2) Higher confidence, 3) Single face over multiple
                if num_faces > 0:
                    is_better = False
                    
                    if best_num_faces == 0:
                        # First angle with faces
                        is_better = True
                    elif confidence > best_confidence:
                        # Higher confidence
                        is_better = True
                    elif confidence == best_confidence and num_faces == 1 and best_num_faces > 1:
                        # Same confidence but single face is better
                        is_better = True
                    
                    if is_better:
                        best_angle = angle
                        best_confidence = confidence
                        best_num_faces = num_faces
                        best_image_path = temp_path if angle != 0 else image_path
                
                # Clean up temp file if not the best
                if angle != 0 and temp_path != best_image_path:
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                        
            except Exception as e:
                print(f"Error processing angle {angle}: {e}")
                results.append({
                    'angle': angle,
                    'confidence': 0.0,
                    'num_faces': 0,
                    'error': str(e)
                })
        
        print(f"  Rotation analysis: Best angle={best_angle}°, confidence={best_confidence:.3f}, faces={best_num_faces}")
        
        return {
            'best_angle': best_angle,
            'best_confidence': best_confidence,
            'best_image_path': best_image_path,
            'num_faces': best_num_faces,
            'all_results': results
        }
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning up temp dir: {e}")
