import zipfile
from pathlib import Path

def create_clean_submission():
    # Required files
    files = [
        'main.py',
        'assets/train_features.csv',
        'assets/train_labels.csv'
    ]
    
    # Create new zip file
    with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            zf.write(file)

if __name__ == "__main__":
    create_clean_submission()