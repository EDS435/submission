from pathlib import Path
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
import numpy as np

class Config:
    max_length = 512 
    batch_size = 16
    learning_rate = 2e-5
    epochs = 5
    num_binary_labels = 21  # Number of binary classification tasks
    num_injury_classes = 6
    num_weapon_classes = 12
    model_name = 'google/electra-base-discriminator'
    
    # Define binary columns based on mental health history, diagnoses, contributing factors, and disclosure
    binary_columns = [
        # Mental health history and current state
        'DepressedMood',  # Person perceived to be depressed at time
        'MentalIllnessTreatmentCurrnt',  # Currently in mental health/substance abuse treatment
        'HistoryMentalIllnessTreatmnt',  # History of mental health/substance abuse treatment
        'SuicideAttemptHistory',  # Previous suicide attempts
        'SuicideThoughtHistory',  # History of suicidal thoughts/plans
        'SubstanceAbuseProblem',  # Substance abuse issues
        'MentalHealthProblem',  # Mental health condition at time
        
        # Specific mental health diagnoses
        'DiagnosisAnxiety',
        'DiagnosisDepressionDysthymia', 
        'DiagnosisBipolar',
        'DiagnosisAdhd',
        
        # Contributing factors
        'IntimatePartnerProblem',  # Problems with current/former partner
        'FamilyRelationship',  # Family relationship problems
        'Argument',  # Arguments/conflicts
        'SchoolProblem',  # School-related problems
        'RecentCriminalLegalProblem',  # Criminal legal problems
        
        # Disclosure of intent
        'SuicideNote',  # Left suicide note
        'SuicideIntentDisclosed',  # Disclosed intent in last month
        'DisclosedToIntimatePartner',  # Disclosed to partner
        'DisclosedToOtherFamilyMember',  # Disclosed to family
        'DisclosedToFriend'  # Disclosed to friend
    ]
    
    # Categorical column mappings
    injury_location_map = {
        1: 'House, apartment',
        2: 'Motor vehicle', 
        3: 'Natural area',
        4: 'Park, playground',
        5: 'Street/road',
        6: 'Other'
    }
    
    # Detailed injury types based on weapon used
    weapon_type_map = {
        1: 'Blunt force trauma from blunt instrument',
        2: 'Drowning/asphyxiation by submersion',
        3: 'Impact injuries from fall',
        4: 'Burn injuries from fire',
        5: 'Gunshot wounds from firearm',
        6: 'Asphyxiation from hanging/strangulation/suffocation',
        7: 'Trauma from motor vehicle collision',
        8: 'Trauma from other transport vehicle collision',
        9: 'Toxic effects from poisoning/overdose',
        10: 'Lacerations/stab wounds from sharp instrument',
        11: 'Other injury mechanism',
        12: 'Unknown injury type'
    }

class SuicideDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all texts
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

class SuicideClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.electra = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(0.3)
        
        # Separate classifiers for binary and categorical outputs
        self.binary_classifier = nn.Linear(self.electra.config.hidden_size, config.num_binary_labels)
        self.injury_classifier = nn.Linear(self.electra.config.hidden_size, config.num_injury_classes)
        self.weapon_classifier = nn.Linear(self.electra.config.hidden_size, config.num_weapon_classes)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.electra(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # Take [CLS] token representation
        dropout_output = self.dropout(pooled_output)
        
        # Generate predictions for each type
        binary_logits = self.sigmoid(self.binary_classifier(dropout_output))
        injury_logits = self.injury_classifier(dropout_output)
        weapon_logits = self.weapon_classifier(dropout_output)
        
        return binary_logits, injury_logits, weapon_logits

def prepare_data(features_df: pd.DataFrame) -> tuple:
    """Prepare data for model prediction"""
    # Ensure uid column is present and set as string
    if 'uid' in features_df.columns:
        features_df['uid'] = features_df['uid'].astype(str)
    
    # Combine narratives and add injury details
    features_df['combined_narrative'] = features_df.apply(
        lambda row: f"{row['NarrativeLE']} {row['NarrativeCME']} Location: {Config.injury_location_map[int(row['InjuryLocationType'])] if 'InjuryLocationType' in row else 'Unknown location'} Injury type: {Config.weapon_type_map[int(row['WeaponType1'])] if 'WeaponType1' in row else 'Unknown'}", 
        axis=1
    )
    
    # Get texts
    texts = features_df['combined_narrative'].values
    
    # Ensure all binary columns are 0 or 1
    for col in Config.binary_columns:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(int).clip(0, 1)
    
    # Ensure categorical columns are within valid ranges
    if 'InjuryLocationType' in features_df.columns:
        features_df['InjuryLocationType'] = features_df['InjuryLocationType'].clip(1, 6)
    
    if 'WeaponType1' in features_df.columns:
        features_df['WeaponType1'] = features_df['WeaponType1'].clip(1, 12)
    
    return texts, Config.binary_columns

SUBMISSION_PATH = Path("data\submission_format.csv")
FEATURES_PATH = Path("data/test_features.csv")
SUBMISSION_FORMAT_PATH = Path("assets/smoke_test_labels_waBGl8d.csv")

def generate_predictions(features: pd.DataFrame, submission_format: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions using the model"""
    # Initialize model and tokenizer
    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = SuicideClassifier(config)
    
    # Prepare data
    texts, binary_columns = prepare_data(features)
    
    # Create dataset
    dataset = SuicideDataset(texts, np.zeros((len(texts), config.num_binary_labels)), tokenizer)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
    
    # Generate predictions
    model.eval()
    binary_predictions = []
    injury_predictions = []
    weapon_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            binary_logits, injury_logits, weapon_logits = model(input_ids, attention_mask)
            
            binary_predictions.append(binary_logits.numpy())
            injury_predictions.append(injury_logits.numpy())
            weapon_predictions.append(weapon_logits.numpy())
    
    # Stack predictions
    binary_predictions = np.vstack(binary_predictions)
    injury_predictions = np.vstack(injury_predictions)
    weapon_predictions = np.vstack(weapon_predictions)
    
    # Convert categorical predictions to class indices
    injury_classes = np.argmax(injury_predictions, axis=1) + 1  # Add 1 for 1-based indexing
    weapon_classes = np.argmax(weapon_predictions, axis=1) + 1
    
    # Create predictions dataframe and round binary predictions to integers
    predictions_df = pd.DataFrame(np.round(binary_predictions).astype(int), columns=config.binary_columns, index=submission_format.index)
    predictions_df['InjuryLocationType'] = injury_classes
    predictions_df['WeaponType1'] = weapon_classes
    
    # Ensure columns are in the correct order
    predictions_df = predictions_df[submission_format.columns]
    
    return predictions_df

def main():
    # Load the data files
    features = pd.read_csv(FEATURES_PATH, index_col=0)
    print(f"Loaded test features of shape {features.shape}")

    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH, index_col=0)
    print(f"Loaded submission format of shape: {submission_format.shape}")

    # Generate predictions
    predictions = generate_predictions(features, submission_format)
    print(f"Saving predictions of shape {predictions.shape} to {SUBMISSION_PATH}")
    predictions.to_csv(SUBMISSION_PATH, index=True)

if __name__ == "__main__":
    main()