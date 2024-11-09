# Import required libraries
from pathlib import Path
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
import numpy as np
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration class to store model parameters and mappings
class Config:
    # Model hyperparameters
    max_length = 512 
    batch_size = 16
    learning_rate = 2e-5
    epochs = 5
    model_name = 'google/electra-base-discriminator'
    seed = 42
    
    # Define output dimensions for different prediction tasks
    num_binary_labels = 21
    num_injury_classes = 6
    num_weapon_classes = 12
    
    # List of binary features to predict, grouped by category
    binary_columns = [
        # Individual identifier is handled separately as index
        
        # Mental health history and current state
        'DepressedMood',                  # Person perceived to be depressed (0/1)
        'MentalIllnessTreatmentCurrnt',   # Current mental health/substance treatment (0/1)
        'HistoryMentalIllnessTreatmnt',   # History of treatment (0/1)
        'SuicideAttemptHistory',          # Previous suicide attempts (0/1)
        'SuicideThoughtHistory',          # History of suicidal thoughts/plans (0/1)
        'SubstanceAbuseProblem',          # Combined alcohol and substance abuse (0/1)
        'MentalHealthProblem',            # Mental health condition present (0/1)
        
        # Specific mental health diagnoses
        'DiagnosisAnxiety',               # Anxiety disorder diagnosis (0/1)
        'DiagnosisDepressionDysthymia',   # Depression/dysthymia diagnosis (0/1)
        'DiagnosisBipolar',               # Bipolar disorder diagnosis (0/1)
        'DiagnosisAdhd',                  # ADHD diagnosis (0/1)
        
        # Contributing factors
        'IntimatePartnerProblem',         # Problems with current/former partner (0/1)
        'FamilyRelationship',             # Family relationship problems (0/1)
        'Argument',                       # Arguments/conflicts (0/1)
        'SchoolProblem',                  # School-related problems (0/1)
        'RecentCriminalLegalProblem',     # Criminal legal problems (0/1)
        
        # Disclosure of intent
        'SuicideNote',                    # Left suicide note (0/1)
        'SuicideIntentDisclosed',         # Disclosed intent in last month (0/1)
        'DisclosedToIntimatePartner',     # Disclosed to partner (0/1)
        'DisclosedToOtherFamilyMember',   # Disclosed to family (0/1)
        'DisclosedToFriend'               # Disclosed to friend (0/1)
    ]
    
    # Mapping dictionaries for categorical variables
    injury_location_map = {
        1: 'House, apartment',
        2: 'Motor vehicle',               # Excluding school bus and public transportation
        3: 'Natural area',                # Field, river, beaches, woods
        4: 'Park, playground',            # Public use area
        5: 'Street/road',                 # Including sidewalk, alley
        6: 'Other'
    }
    
    weapon_type_map = {
        1: 'Blunt instrument',
        2: 'Drowning', 
        3: 'Fall',
        4: 'Fire or burns',
        5: 'Firearm',
        6: 'Hanging, strangulation, suffocation',
        7: 'Motor vehicle including buses, motorcycles',
        8: 'Other transport vehicle, eg, trains, planes, boats',
        9: 'Poisoning',
        10: 'Sharp instrument',
        11: 'Other (e.g. taser, electrocution, nail gun)',
        12: 'Unknown'
    }

# Custom Dataset class for handling text data and labels
class SuicideDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all texts for efficiency
        self.encodings = self.tokenizer(
            list(self.texts),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# Neural network model for multi-task classification
class SuicideClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load pre-trained ELECTRA model
        self.electra = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(0.3)
        
        # Define separate classification heads for each task
        self.binary_classifier = nn.Linear(self.electra.config.hidden_size, config.num_binary_labels)
        self.injury_classifier = nn.Linear(self.electra.config.hidden_size, config.num_injury_classes)
        self.weapon_classifier = nn.Linear(self.electra.config.hidden_size, config.num_weapon_classes)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        # Get ELECTRA embeddings
        outputs = self.electra(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # Take [CLS] token representation
        dropout_output = self.dropout(pooled_output)
        
        # Generate predictions for each task
        binary_logits = self.sigmoid(self.binary_classifier(dropout_output))
        injury_logits = self.injury_classifier(dropout_output)
        weapon_logits = self.weapon_classifier(dropout_output)
        
        return binary_logits, injury_logits, weapon_logits

# Data preprocessing function
def prepare_data(features_df: pd.DataFrame) -> tuple:
    """Prepare data for model prediction"""
    # Handle UID column
    if 'uid' in features_df.columns:
        features_df['uid'] = features_df['uid'].astype(str)
    
    # Combine narrative fields and add location/injury details
    features_df['combined_narrative'] = features_df.apply(
        lambda row: f"{row['NarrativeLE']} {row['NarrativeCME']} Location: {Config.injury_location_map[int(row['InjuryLocationType'])] if 'InjuryLocationType' in row else 'Unknown location'} Injury type: {Config.weapon_type_map[int(row['WeaponType1'])] if 'WeaponType1' in row else 'Unknown'}", 
        axis=1
    )
    
    # Extract text features
    texts = features_df['combined_narrative'].values
    
    # Clean binary columns
    for col in Config.binary_columns:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(int).clip(0, 1)
    
    # Clean categorical columns
    if 'InjuryLocationType' in features_df.columns:
        features_df['InjuryLocationType'] = features_df['InjuryLocationType'].clip(1, 6)
    
    if 'WeaponType1' in features_df.columns:
        features_df['WeaponType1'] = features_df['WeaponType1'].clip(1, 12)
    
    return texts, Config.binary_columns

# Define file paths
SUBMISSION_PATH = Path("data\submission_format.csv")
FEATURES_PATH = Path("data/test_features.csv")
SUBMISSION_FORMAT_PATH = Path("assets/smoke_test_labels_waBGl8d.csv")

# Main prediction function
def generate_predictions(features: pd.DataFrame, submission_format: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions using the model"""
    # Set random seed for reproducibility
    config = Config()
    set_seed(config.seed)
    
    # Setup model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = SuicideClassifier(config)
    
    # Prepare input data
    texts, binary_columns = prepare_data(features)
    
    # Create dataset and dataloader
    dataset = SuicideDataset(texts, np.zeros((len(texts), config.num_binary_labels)), tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=False,  # Keep order consistent
        num_workers=0,  # Single worker for reproducibility
        pin_memory=True
    )
    
    # Generate predictions
    model.eval()
    binary_predictions = []
    injury_predictions = []
    weapon_predictions = []
    
    # Run inference
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            binary_logits, injury_logits, weapon_logits = model(input_ids, attention_mask)
            
            binary_predictions.append(binary_logits.numpy())
            injury_predictions.append(injury_logits.numpy())
            weapon_predictions.append(weapon_logits.numpy())
    
    # Process predictions
    binary_predictions = np.vstack(binary_predictions)
    injury_predictions = np.vstack(injury_predictions)
    weapon_predictions = np.vstack(weapon_predictions)
    
    # Convert to final format
    injury_probs = torch.nn.functional.softmax(torch.tensor(injury_predictions), dim=1).numpy()
    injury_classes = np.argmax(injury_probs, axis=1)  # Add 1 for 1-based indexing
    weapon_probs = torch.nn.functional.softmax(torch.tensor(weapon_predictions), dim=1).numpy()
    weapon_classes = np.argmax(weapon_probs, axis=1) + 1  # Add 1 for 1-based indexing
    
    # Create output dataframe
    predictions_df = pd.DataFrame(np.round(binary_predictions).astype(int), columns=config.binary_columns, index=submission_format.index)
    predictions_df['InjuryLocationType'] = injury_classes
    predictions_df['WeaponType1'] = weapon_classes
    
    # Ensure correct column order
    predictions_df = predictions_df[submission_format.columns]
    
    return predictions_df

# Main execution function
def main():
    # Set random seed at program start
    set_seed(Config.seed)
    
    # Load input data
    features = pd.read_csv(FEATURES_PATH, index_col=0)
    print(f"Loaded test features of shape {features.shape}")

    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col=0)
    print(f"Loaded submission format of shape: {submission_format.shape}")

    # Generate and save predictions
    predictions = generate_predictions(features, submission_format)
    print(f"Saving predictions of shape {predictions.shape} to {SUBMISSION_PATH}")
    predictions.to_csv(SUBMISSION_PATH, index=True)

# Entry point
if __name__ == "__main__":
    main()