import os
from typing import List, Dict, Any
from src.core.data_structures import Applicant, Document
from src.document_processor.extractor import DocumentExtractor
from src.face_engine.analyzer import FaceAnalyzer

class VerificationEngine:
    def __init__(self):
        self.extractor = DocumentExtractor()
        self.analyzer = FaceAnalyzer()

    def process_applicant(self, applicant: Applicant) -> Dict[str, Any]:
        print(f"Processing Applicant: {applicant.role}") 
        
        # 1. Extract Primary Embeddings
        primary_embeddings = []
        primary_faces_count = 0
        
        print(f"  - Processing {len(applicant.primary_docs)} primary documents...")
        for doc in applicant.primary_docs:
            print(f"    - Extracting from {doc.original_filename or doc.file_path}...")
            for img_path in self.extractor.extract_images(doc.file_path):
                faces = self.analyzer.get_face_embeddings(img_path)
                primary_embeddings.extend(faces)
                primary_faces_count += len(faces)
                
                # Cleanup temp image if needed, or keep for debugging
                # os.remove(img_path) 

        print(f"  - Found {primary_faces_count} faces in primary documents.")

        if not primary_embeddings:
            print("  - WARNING: No faces found in primary documents. Cannot verify comparisons.")
        
        # 2. Process Comparison Documents
        comparison_results = []
        
        print(f"  - Processing {len(applicant.comparison_docs)} comparison documents...")
        for doc in applicant.comparison_docs:
            doc_result = {
                'document_class': doc.doc_class,
                'filename': doc.original_filename,
                'file_path': doc.file_path,
                'faces_found': 0,
                'is_match': False,
                'confidence': 0.0,
                'distance': 1.0,
                'rotation_angle': 0,
                'details': 'No faces found in comparison document'
            }
            
            doc_faces = []
            try:
                for img_path in self.extractor.extract_images(doc.file_path):
                    faces = self.analyzer.get_face_embeddings(img_path)
                    doc_faces.extend(faces)
            except Exception as e:
                doc_result['details'] = f"Error processing document: {str(e)}"
            
            doc_result['faces_found'] = len(doc_faces)
            
            # Only compare if both primary and comparison document have faces
            if doc_faces and primary_embeddings:
                # Compare all pairs and find best match
                best_match = None
                best_similarity = -1.0
                min_distance = 2.0
                best_rotation = 0
                
                for p_face in primary_embeddings:
                    for c_face in doc_faces:
                        res = self.analyzer.verify_embeddings(p_face['embedding'], c_face['embedding'])
                        
                        if res['similarity'] > best_similarity:
                            best_similarity = res['similarity']
                            min_distance = res['distance']
                            best_match = res
                            best_rotation = c_face.get('rotation_angle', 0)
                
                if best_match:
                    doc_result['is_match'] = bool(best_match['verified'])
                    doc_result['confidence'] = float(round(best_similarity, 4))
                    doc_result['distance'] = float(round(min_distance, 4))
                    doc_result['rotation_angle'] = best_rotation
                    doc_result['details'] = "Comparison complete"
                else:
                     doc_result['details'] = "Comparison failed (unknown error)"
            elif not doc_faces:
                # Don't skip, report as no faces found
                print(f"    - No human faces detected in {doc.doc_class}")
                doc_result['details'] = "No human faces detected"
                doc_result['is_match'] = False
            elif not primary_embeddings:
                doc_result['details'] = "No primary face available for comparison"
                doc_result['is_match'] = False

            comparison_results.append(doc_result)

        return {
            'role': applicant.role,
            'primary_faces_detected': primary_faces_count,
            'comparisons': comparison_results
        }
    
    def cleanup(self):
        """Clean up temporary files"""
        if hasattr(self, 'analyzer'):
            self.analyzer.cleanup()
