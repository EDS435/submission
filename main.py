from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Set
import re
from tqdm import tqdm
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import sys

SUBMISSION_PATH = Path("submission.csv")
TRAIN_FEATURES_PATH = Path("assets/smoke_test_features.csv")
TEST_FEATURES_PATH = Path("assets/smoke_test_features.csv")
TRAIN_LABELS_PATH = Path("assets/smoke_test_labels.csv")
SUBMISSION_FORMAT_PATH = Path("data/submission_format.csv")

class HPOMapper:
    def __init__(self):
        
        self.hpo_mappings = {
            'depression': 'HP:0000716',
            'anxiety': 'HP:0000739',
            'bipolar': 'HP:0007302',
            'adhd': 'HP:0007018',
            'substance_abuse': 'HP:0030858',
            'suicide': 'HP:0031347'
        }

        # Binary columns based on training data
        self.binary_columns = [
            'DepressedMood', 'MentalIllnessTreatmentCurrnt', 'HistoryMentalIllnessTreatmnt',
            'SuicideAttemptHistory', 'SuicideThoughtHistory', 'SubstanceAbuseProblem',
            'MentalHealthProblem', 'DiagnosisAnxiety', 'DiagnosisDepressionDysthymia',
            'DiagnosisBipolar', 'DiagnosisAdhd', 'IntimatePartnerProblem',
            'FamilyRelationship', 'Argument', 'SchoolProblem', 'RecentCriminalLegalProblem',
            'SuicideNote', 'SuicideIntentDisclosed', 'DisclosedToIntimatePartner',
            'DisclosedToOtherFamilyMember', 'DisclosedToFriend'
        ]
        
        # Improved classifiers with more trees and better parameters
        self.binary_classifiers = {col: RandomForestClassifier(
            n_estimators=1000,  # More trees
            max_depth=20,       # Deeper trees
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced_subsample',  # Better handling of imbalanced data
            random_state=42,
            n_jobs=-1
        ) for col in self.binary_columns}
        
        self.location_classifier = RandomForestClassifier(
            n_estimators=1000,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        
        self.weapon_classifier = RandomForestClassifier(
            n_estimators=500,  # Increased from 100
            max_depth=20,      # Added parameter
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )

    def extract_features(self, text_data):
        features = {
            # Mental health terms - expanded
            'mental_health_terms': len(re.findall(r'\b(depress|anxi|bipolar|adhd|mental|schizo|ptsd|ocd|disorder)\b', text_data, re.I)),
            'treatment_terms': len(re.findall(r'\b(treat|therap|medic|doctor|hospital|psychiatr|counsel|clinic|prescri)\b', text_data, re.I)),
            'suicide_terms': len(re.findall(r'\b(suicid|kill|die|end|life|hang|overdose|shot)\b', text_data, re.I)),
            
            # Relationship indicators
            'relationship_terms': len(re.findall(r'\b(family|partner|friend|school|work|spouse|girlfriend|boyfriend|wife|husband)\b', text_data, re.I)),
            'argument_terms': len(re.findall(r'\b(argu|fight|conflict|disagree|upset|anger|broke up|separat|divorce)\b', text_data, re.I)),
            
            # Substance abuse
            'substance_terms': len(re.findall(r'\b(drug|alcohol|substance|abuse|addict|overdose|intoxicat|withdraw)\b', text_data, re.I)),
            
            # Communication patterns
            'disclosure_terms': len(re.findall(r'\b(tell|said|note|disclos|express|wrote|text|call|message)\b', text_data, re.I)),
            'intent_terms': len(re.findall(r'\b(plan|intent|want|going to|threaten|attempt)\b', text_data, re.I)),
            
            # Location specifics
            'location_home': len(re.findall(r'\b(house|home|bedroom|apartment|bathroom|garage|basement)\b', text_data, re.I)),
            'location_public': len(re.findall(r'\b(street|park|bridge|building|office|school|hospital|hotel)\b', text_data, re.I)),
            'location_remote': len(re.findall(r'\b(woods|forest|rural|remote|field|mountain|lake|river)\b', text_data, re.I)),
            
            # Weapon specifics
            'weapon_firearm': len(re.findall(r'\b(gun|shot|pistol|rifle|handgun|bullet|firearm|revolver)\b', text_data, re.I)),
            'weapon_poison': len(re.findall(r'\b(overdose|poison|drug|pill|medication|substance)\b', text_data, re.I)),
            'weapon_sharp': len(re.findall(r'\b(knife|cut|blade|razor|slash|stab)\b', text_data, re.I)),
            'weapon_hang': len(re.findall(r'\b(hang|rope|asphyx|strangle|suffocate)\b', text_data, re.I)),
            
            # Additional contextual features
            'time_indicators': len(re.findall(r'\b(morning|night|evening|today|yesterday|week|month)\b', text_data, re.I)),
            'emotional_state': len(re.findall(r'\b(stress|depress|anxiety|worry|sad|hopeless|lonely|upset)\b', text_data, re.I)),
            'history_terms': len(re.findall(r'\b(history|previous|past|chronic|ongoing|recent)\b', text_data, re.I))
        }
        return pd.Series(features)

    def train(self, features, labels):
        """Train the model with extracted features"""
        # Extract features from training data first
        print("Extracting features from training data...")
        X_train = pd.DataFrame([
            self.extract_features(' '.join(filter(None, [
                str(row.get('NarrativeLE', '')), 
                str(row.get('NarrativeCME', ''))
            ]))) for _, row in tqdm(features.iterrows())
        ])
        
        print("Training binary classifiers...")
        for col in tqdm(self.binary_columns):
            self.binary_classifiers[col].fit(X_train, labels[col])
        
        print("Training location classifier...")
        self.location_classifier.fit(X_train, labels['InjuryLocationType'])
        
        print("Training weapon classifier...")
        self.weapon_classifier.fit(X_train, labels['WeaponType1'])

def validate_submission(submission_df: pd.DataFrame, format_df: pd.DataFrame) -> bool:
    """
    Validates that submission matches required format
    
    Args:
        submission_df: Generated submission DataFrame
        format_df: Expected format DataFrame
        
    Returns:
        bool: True if valid, False if invalid
    """
    # Check if columns match
    if not all(submission_df.columns == format_df.columns):
        print("ERROR: Columns don't match expected format")
        print(f"Expected: {format_df.columns.tolist()}")
        print(f"Got: {submission_df.columns.tolist()}")
        return False
    
    # Check if UIDs are unique
    if len(submission_df['uid'].unique()) != len(submission_df):
        print("ERROR: Duplicate UIDs found")
        return False
        
    # Validate data types and ranges
    for col in submission_df.columns:
        if col == 'uid':
            # Check if UIDs are strings
            if not submission_df[col].dtype == object:
                print(f"ERROR: UIDs must be strings")
                return False
            continue
            
        # Check if values are integers
        if not submission_df[col].dtype in ['int64', 'int32']:
            print(f"ERROR: Column {col} contains non-integer values")
            return False
            
        # For binary columns (all except last 2)
        if col not in ['InjuryLocationType', 'WeaponType1']:
            if not submission_df[col].isin([0, 1]).all():
                print(f"ERROR: Binary column {col} contains values other than 0 or 1")
                return False
                
        # Check InjuryLocationType range (1-6)
        elif col == 'InjuryLocationType':
            if not submission_df[col].between(1, 6).all():
                print(f"ERROR: InjuryLocationType contains values outside valid range (1-6)")
                return False
                
        # Check WeaponType1 range (1-12)
        elif col == 'WeaponType1':
            if not submission_df[col].between(1, 12).all():
                print(f"ERROR: WeaponType1 contains values outside valid range (1-12)")
                return False
                
    print("Submission format validation passed!")
    return True

def post_process_predictions(predictions):
    # Only apply treatment history when there's strong evidence
    mask = (predictions['MentalHealthProblem'] == 1) & \
           (predictions['MentalIllnessTreatmentCurrnt'] == 1)
    predictions.loc[mask, 'HistoryMentalIllnessTreatmnt'] = 1
    
    # More focused suicide-related rules
    mask = predictions['SuicideNote'] == 1
    predictions.loc[mask, 'SuicideIntentDisclosed'] = 1  # Keep this rule
    
    # Disclosure consistency (keep this as is)
    disclosure_cols = ['DisclosedToIntimatePartner', 'DisclosedToOtherFamilyMember', 'DisclosedToFriend']
    mask = predictions[disclosure_cols].any(axis=1)
    predictions.loc[mask, 'SuicideIntentDisclosed'] = 1
    
    # Mental health diagnosis implications (keep this as is)
    diagnosis_cols = ['DiagnosisAnxiety', 'DiagnosisDepressionDysthymia', 'DiagnosisBipolar', 'DiagnosisAdhd']
    mask = predictions[diagnosis_cols].any(axis=1)
    predictions.loc[mask, 'MentalHealthProblem'] = 1
    
    return predictions

def main():
    # Load data
    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH)
    train_features = pd.read_csv(TRAIN_FEATURES_PATH, index_col='uid')
    train_labels = pd.read_csv(TRAIN_LABELS_PATH, index_col='uid')
    test_features = pd.read_csv(TEST_FEATURES_PATH)
    
    # Initialize and train the mapper
    mapper = HPOMapper()
    mapper.train(train_features, train_labels)
    
    # Create empty predictions DataFrame with correct columns
    predictions = pd.DataFrame(columns=submission_format.columns)
    
    # Add UIDs from test features
    predictions['uid'] = test_features['uid']
    
    # Generate features for test data
    X_test = pd.DataFrame([
        mapper.extract_features(' '.join(filter(None, [
            str(row.get('NarrativeLE', '')),
            str(row.get('NarrativeCME', ''))
        ]))) for _, row in test_features.iterrows()
    ])
    
    # Generate predictions with adjusted thresholds
    for col in predictions.columns[1:]:  # Skip uid column
        if col == 'InjuryLocationType':
            base_pred = mapper.location_classifier.predict(X_test)
            # Reduce randomness for more stable predictions
            mask = np.random.random(len(base_pred)) > 0.8  # Only modify 20% of predictions
            base_pred[mask] = np.random.randint(1, 7, size=mask.sum())
            predictions[col] = base_pred.astype(int)
            
        elif col == 'WeaponType1':
            base_pred = mapper.weapon_classifier.predict(X_test)
            mask = np.random.random(len(base_pred)) > 0.8  # Only modify 20% of predictions
            base_pred[mask] = np.random.randint(1, 13, size=mask.sum())
            predictions[col] = base_pred.astype(int)
            
        else:  # Binary columns
            base_pred = mapper.binary_classifiers[col].predict_proba(X_test)[:, 1]
            
            # Dynamic thresholds based on column characteristics
            if col in ['SuicideNote', 'SuicideIntentDisclosed']:
                threshold = 0.35  # Increase from 0.25
            elif col in ['MentalHealthProblem', 'DepressedMood']:
                threshold = 0.40  # Increase from 0.30
            elif col in ['SubstanceAbuseProblem', 'DiagnosisAnxiety']:
                threshold = 0.45  # Increase from 0.35
            else:
                threshold = 0.50  # Increase from 0.40
            
            predictions[col] = (base_pred > threshold).astype(int)
    
    # Fill any missing values with defaults
    for col in predictions.columns:
        if col == 'uid':
            continue
        elif col in ['InjuryLocationType']:
            predictions[col] = predictions[col].fillna(6).astype(int)
        elif col == 'WeaponType1':
            predictions[col] = predictions[col].fillna(12).astype(int)
        else:
            predictions[col] = predictions[col].fillna(0).astype(int)
    
    # Ensure all numeric columns are integers
    numeric_columns = predictions.columns.difference(['uid'])
    predictions[numeric_columns] = predictions[numeric_columns].astype(int)
    
    # Post-process predictions
    predictions = post_process_predictions(predictions)
    
    # Validate before saving
    if validate_submission(predictions, submission_format):
        print(f"\nPredictions shape: {predictions.shape}")
        print(f"Columns: {predictions.columns.tolist()}")
        print(f"\nSaving predictions to {SUBMISSION_PATH}")
        
        try:
            predictions.to_csv(SUBMISSION_PATH, index=False, float_format='%.0f')
            if SUBMISSION_PATH.exists():
                print(f"Successfully created {SUBMISSION_PATH}")
            else:
                raise FileNotFoundError("Failed to create submission file")
        except Exception as e:
            print(f"ERROR: {str(e)}")
            sys.exit(1)
    else:
        print("ERROR: Validation failed, file not saved")

if __name__ == "__main__":
    main()