from PyPDF2 import PdfReader
import json


def process_manual(file_path):
    '''
    Run this function the first time a tech manual is processed to extract text.
    This process takes approximately 2 minutes. 
    Input: path to PDF, including pdf filename
    Output: JSON file
    '''

    
    reader = PdfReader(file_path)
    results = {}
    print('Converting PDF to searchable data structure...')
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        results[idx] = text
    print('Conversion complete. Writing to local file...', flush=True)
    with open('ExtractedManual.json', 'w') as file:
        json.dump(results, file)
    print('Done.')
    return 'PDF extraction complete.'