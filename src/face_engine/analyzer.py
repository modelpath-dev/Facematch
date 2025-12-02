from deepface import DeepFace
import cv2
import numpy as np
from typing import List, Dict, Optional
from src.face_engine.rotation_handler import RotationHandler

class FaceAnalyzer:
    def __init__(self, 
                 model_name='ArcFace', 
                 detector_backend='retinaface',
                 min_face_confidence=0.5,  # Lowered to be more lenient
                 match_threshold=0.60,      # Slightly relaxed from 0.55
                 min_face_size=30,          # Lowered to handle small faces in ID documents
                 max_face_area_ratio=0.85,  # Slightly more lenient
                 enable_rotation=True):
        """
        Initialize FaceAnalyzer with quality filtering and rotation support.
        
        Args:
            model_name: Embedding model (ArcFace, Facenet, etc.)
            detector_backend: Face detector (retinaface, mtcnn, etc.)
            min_face_confidence: Minimum confidence for face detection (0-1)
            match_threshold: Threshold for face matching (lower = stricter)
            min_face_size: Minimum face dimension in pixels
            max_face_area_ratio: Maximum ratio of face area to image area
            enable_rotation: Whether to try multiple rotations
        """
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.min_face_confidence = min_face_confidence
        self.match_threshold = match_threshold
        self.min_face_size = min_face_size
        self.max_face_area_ratio = max_face_area_ratio
        self.enable_rotation = enable_rotation
        
        # Initialize rotation handler
        if enable_rotation:
            self.rotation_handler = RotationHandler(detector_backend=detector_backend)
        
        # Force model build to load weights upfront
        print(f"Initializing FaceAnalyzer with {model_name} and {detector_backend}...")
        print(f"  - Min face confidence: {min_face_confidence}")
        print(f"  - Match threshold: {match_threshold}")
        print(f"  - Rotation enabled: {enable_rotation}")
        try:
            DeepFace.build_model(model_name)
        except Exception as e:
            print(f"Error building model {model_name}: {e}")

    def _is_valid_face(self, face_data: Dict, image_shape: tuple) -> bool:
        """
        Validate if detected face meets quality criteria.
        
        Args:
            face_data: Face detection result from DeepFace
            image_shape: Shape of the original image (height, width, channels)
            
        Returns:
            True if face passes quality checks
        """
        # Check confidence - handle different field names across DeepFace versions
        confidence = face_data.get('confidence', face_data.get('face_confidence', 0))
        
        # If confidence is still 0, it might be a detection without confidence score
        # In this case, we'll be more lenient and check other criteria
        if confidence > 0 and confidence < self.min_face_confidence:
            print(f"    - Rejected: Low confidence ({confidence:.3f} < {self.min_face_confidence})")
            return False
        
        # Check face size
        facial_area = face_data.get('facial_area', {})
        face_width = facial_area.get('w', 0)
        face_height = facial_area.get('h', 0)
        
        if face_width < self.min_face_size or face_height < self.min_face_size:
            print(f"    - Rejected: Face too small ({face_width}x{face_height} < {self.min_face_size})")
            return False
        
        # Check if face area is too large (likely full-image fallback)
        image_height, image_width = image_shape[:2]
        image_area = image_width * image_height
        face_area = face_width * face_height
        area_ratio = face_area / image_area if image_area > 0 else 0
        
        if area_ratio > self.max_face_area_ratio:
            print(f"    - Rejected: Face area too large ({area_ratio:.2f} > {self.max_face_area_ratio})")
            return False
        
        return True

    def get_face_embeddings(self, img_path: str) -> List[Dict]:
        """
        Detects faces in an image and returns a list of embeddings with metadata.
        Tries multiple rotations if enabled and applies quality filtering.
        """
        results = []
        
        # Step 1: Find best rotation if enabled
        best_image_path = img_path
        rotation_info = None
        
        if self.enable_rotation:
            rotation_info = self.rotation_handler.find_best_rotation(img_path)
            best_image_path = rotation_info['best_image_path']
            
            # If no faces found at any angle, return empty
            if rotation_info['num_faces'] == 0:
                print(f"    - No faces detected at any rotation angle")
                return []
        
        # Step 2: Get embeddings from best rotation
        try:
            # Load image to get dimensions for validation
            image = cv2.imread(best_image_path)
            if image is None:
                print(f"Error: Could not load image {best_image_path}")
                return []
            
            image_shape = image.shape
            
            # DeepFace.represent returns a list of dicts
            embeddings = DeepFace.represent(
                img_path=best_image_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True
            )
            
            # Step 3: Filter and validate faces
            for emb in embeddings:
                # Validate face quality
                if not self._is_valid_face(emb, image_shape):
                    continue
                
                # Extract confidence with fallback for different field names
                confidence = emb.get('confidence', emb.get('face_confidence', 0.9))
                
                # Add to results
                result_data = {
                    'embedding': emb['embedding'],
                    'box': emb['facial_area'],
                    'confidence': confidence,
                    'source_path': img_path,
                    'rotation_angle': rotation_info['best_angle'] if rotation_info else 0,
                    'quality_score': confidence
                }
                
                results.append(result_data)
                
        except ValueError as e:
            print(f"No faces detected in {img_path}: {e}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
        
        # If we found multiple faces, keep only the highest quality one
        if len(results) > 1:
            print(f"    - Multiple faces detected ({len(results)}), keeping highest quality")
            results = [max(results, key=lambda x: x['quality_score'])]
            
        return results

    def verify_embeddings(self, emb1: List[float], emb2: List[float]) -> Dict:
        """
        Compares two embeddings and returns distance and match status.
        Uses stricter threshold to reduce false positives.
        """
        # Calculate Cosine Distance
        a = np.array(emb1)
        b = np.array(emb2)
        
        # Cosine distance = 1 - cosine_similarity
        denominator = (np.linalg.norm(a) * np.linalg.norm(b))
        if denominator == 0:
            return {
                'verified': False, 
                'distance': 1.0, 
                'threshold': self.match_threshold,
                'similarity': 0.0
            }

        cosine_similarity = np.dot(a, b) / denominator
        cosine_distance = 1 - cosine_similarity
        
        # Use configured threshold (stricter than default)
        verified = cosine_distance < self.match_threshold
        
        return {
            'verified': verified,
            'distance': float(cosine_distance),
            'threshold': self.match_threshold,
            'similarity': float(cosine_similarity)
        }
    
    def cleanup(self):
        """Clean up temporary files from rotation handler"""
        if hasattr(self, 'rotation_handler'):
            self.rotation_handler.cleanup()
