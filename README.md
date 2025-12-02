# Face Verification System

A face verification pipeline that compares faces across various document formats including images, PDFs, and Excel files. The system extracts faces from primary documents and verifies them against faces in comparison documents.

## Technologies Used

### Core Libraries

- **DeepFace**: For face detection, embedding extraction, and face verification
- **OpenCV**: For image processing and manipulation
- **TensorFlow/Keras**: Backend for deep learning models
- **ArcFace Model**: State-of-the-art face recognition model for generating embeddings
- **RetinaFace**: High-performance face detection backend

### Document Processing

- **pdf2image**: Converts PDF documents to images for face extraction
- **openpyxl**: Extracts images from Excel (.xlsx) files
- **mimetypes**: Identifies file types for appropriate processing

### Infrastructure & Utilities

- **boto3**: AWS SDK for cloud storage integration
- **python-dotenv**: Environment variable management
- **pandas**: Data processing and manipulation

## System Architecture

The system follows a modular architecture with the following components:

1. **Input Handlers**: Process different input formats (JSON configuration or folder scanning)
2. **Document Processor**: Extracts images from various document types (PDF, Excel, images)
3. **Face Engine**: Detects faces, generates embeddings, and performs verification
4. **Core Engine**: Orchestrates the entire verification pipeline

### Key Features

- Multi-format document support (images, PDFs, Excel files)
- Automatic face detection with quality filtering
- Image rotation handling for improved face detection
- Configurable verification thresholds
- Detailed result reporting with confidence scores

## Setup Instructions

### Prerequisites

- Python 3.7+
- pip package manager

### Installation Steps

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd face-verification-system
   ```
2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   Create a `.env` file in the project root with necessary credentials:

   ```
   # AWS Credentials (if using S3)
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key

   # Other service configurations as needed
   ```

### Running the System

#### Mode 1: JSON Input

Process applicants defined in a JSON configuration file:

```bash
python main.py --mode json --input path/to/input.json --output path/to/output.json
```

#### Mode 2: Folder Input

Automatically process all documents in a folder structure:

```bash
python main.py --mode folder --input path/to/documents/folder --output path/to/output.json
```

### JSON Input Format

When using JSON mode, the input file should follow this structure:

```json
{
  "applicants": [
    {
      "role": "applicant",
      "primary_documents": [
        {
          "file_path": "path/to/id_document.jpg",
          "doc_class": "id"
        }
      ],
      "comparison_documents": [
        {
          "file_path": "path/to/selfie.jpg",
          "doc_class": "selfie"
        }
      ]
    }
  ]
}
```

## How It Works

1. **Document Extraction**: The system identifies document types and extracts images appropriately:

   - Image files: Direct processing
   - PDF files: Conversion to high-resolution images
   - Excel files: Extraction of embedded images
2. **Face Detection**: Using RetinaFace, the system detects faces in extracted images with quality filtering:

   - Confidence thresholding
   - Size validation
   - Area ratio checks
3. **Rotation Handling**: Automatically rotates images to find optimal face detection angles.
4. **Embedding Generation**: ArcFace model generates robust facial embeddings for detected faces.
5. **Verification**: Compares embeddings using cosine similarity with configurable thresholds.
6. **Result Generation**: Produces detailed comparison results with confidence scores and metadata.

## Configuration Options

Key parameters can be adjusted in the FaceAnalyzer class:

- `min_face_confidence`: Minimum confidence for face detection (default: 0.5)
- `match_threshold`: Threshold for face matching (default: 0.60)
- `min_face_size`: Minimum face dimension in pixels (default: 30)
- `enable_rotation`: Whether to try multiple rotations (default: True)

## Output Format

The system generates a JSON output with the following structure:

```json
{
  "status": "success",
  "applicant": {
    "role": "applicant",
    "primary_faces_detected": 1,
    "comparisons": [
      {
        "document_class": "selfie",
        "filename": "selfie.jpg",
        "faces_found": 1,
        "is_match": true,
        "confidence": 0.9234,
        "distance": 0.0766,
        "rotation_angle": 0,
        "details": "Comparison complete"
      }
    ]
  },
  "co_applicants": []
}
```

## Troubleshooting

Common issues and solutions:

1. **No faces detected**: Ensure images are clear and faces are well-lit
2. **Low confidence scores**: Check image quality and resolution
3. **Installation errors**: Make sure all dependencies are installed correctly
4. **Memory issues**: Process smaller batches of documents
