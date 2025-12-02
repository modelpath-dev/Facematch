import os
import cv2
import numpy as np
import tempfile
import zipfile
from typing import List, Generator
from pdf2image import convert_from_path
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as OpenPyxlImage
import mimetypes

class DocumentExtractor:
    def __init__(self, temp_dir: str = None):
        self.temp_dir = temp_dir or os.path.join(tempfile.gettempdir(), "face_verification_extracts")
        os.makedirs(self.temp_dir, exist_ok=True)

    def extract_images(self, file_path: str) -> Generator[str, None, None]:
        """
        Yields paths to images found in the document.
        If the document is an image, yields the file_path itself.
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        # Detect file type
        mime_type, _ = mimetypes.guess_type(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']:
            yield file_path
        elif ext == '.pdf':
            yield from self._extract_from_pdf(file_path)
        elif ext in ['.xlsx', '.xlsm']:
            yield from self._extract_from_excel(file_path)
        elif ext == '.xls':
            print(f"Warning: .xls format not fully supported for image extraction without external tools. Skipping {file_path}")
            # .xls is binary OLE2. Hard to extract images without proprietary libs or conversion.
            # We could try to use some library if needed, but for now skip.
            pass
        else:
            print(f"Unsupported file type: {ext} for {file_path}")

    def _extract_from_pdf(self, file_path: str) -> Generator[str, None, None]:
        try:
            # Convert PDF pages to images
            # We use 300 DPI for better face detection
            images = convert_from_path(file_path, dpi=300)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            for i, img in enumerate(images):
                out_path = os.path.join(self.temp_dir, f"{base_name}_page_{i+1}.jpg")
                img.save(out_path, 'JPEG')
                yield out_path
        except Exception as e:
            print(f"Error extracting from PDF {file_path}: {e}")

    def _extract_from_excel(self, file_path: str) -> Generator[str, None, None]:
        """
        Extract images from XLSX by treating it as a ZIP archive (more reliable for extracting all media).
        """
        try:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            with zipfile.ZipFile(file_path, 'r') as z:
                # Images are usually in xl/media/
                media_files = [f for f in z.namelist() if f.startswith('xl/media/')]
                
                for media in media_files:
                    # Extract to temp dir
                    original_ext = os.path.splitext(media)[1]
                    if original_ext.lower() not in ['.png', '.jpg', '.jpeg', '.emf', '.wmf']:
                        continue
                        
                    out_filename = f"{base_name}_{os.path.basename(media)}"
                    out_path = os.path.join(self.temp_dir, out_filename)
                    
                    with open(out_path, 'wb') as f_out:
                        f_out.write(z.read(media))
                    
                    # Convert EMF/WMF to supported format if necessary? 
                    # OpenCV/DeepFace might not support EMF/WMF.
                    # For now, yield supported types.
                    if original_ext.lower() in ['.png', '.jpg', '.jpeg']:
                        yield out_path
        except Exception as e:
            print(f"Error extracting from Excel {file_path}: {e}")
