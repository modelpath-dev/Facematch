import argparse
import json
import os
import sys
from src.input_handlers.json_handler import JsonInputHandler
from src.input_handlers.folder_handler import FolderInputHandler
from src.core.engine import VerificationEngine

def main():
    parser = argparse.ArgumentParser(description="Face Verification System")
    parser.add_argument('--mode', choices=['json', 'folder'], required=True, help="Input mode: 'json' or 'folder'")
    parser.add_argument('--input', required=True, help="Path to input file (json) or directory (folder)")
    parser.add_argument('--output', default='output.json', help="Path to output JSON file")
    
    args = parser.parse_args()
    
    # 1. Initialize Input Handler
    handler = None
    if args.mode == 'json':
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found.")
            sys.exit(1)
        handler = JsonInputHandler(args.input)
    elif args.mode == 'folder':
        if not os.path.exists(args.input):
            print(f"Error: Input directory {args.input} not found.")
            sys.exit(1)
        handler = FolderInputHandler(args.input)
        
    # 2. Initialize Engine
    engine = VerificationEngine()
    
    # 3. Process
    results = []
    print("Starting Face Verification Pipeline...")
    
    try:
        for applicant in handler.get_applicants():
            applicant_result = engine.process_applicant(applicant)
            results.append(applicant_result)
    except Exception as e:
        print(f"Critical Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup temporary files
        engine.cleanup()
    
    # 4. Output - Organize by role
    applicant_results = []
    co_applicant_results = []
    
    for result in results:
        role = result.get('role', '').lower()  # Normalize to lowercase for comparison
        if role == 'applicant':
            applicant_results.append(result)
        elif 'applicant' in role and role != 'applicant':  # co-applicant, coapplicant1, etc.
            co_applicant_results.append(result)
    
    output_data = {
        'status': 'success',
        'applicant': applicant_results[0] if applicant_results else None,
        'co_applicants': co_applicant_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Processing complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
