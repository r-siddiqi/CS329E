# Import required libraries for evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch

def evaluate_model(model, tokenizer, test_data):
    model.eval()  # Set model to evaluation mode
    predictions = []
    true_labels = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
            # Tokenize input
            inputs = tokenizer(row['message'], 
                             return_tensors="pt",
                             truncation=True,
                             max_length=512,
                             padding=True)
            
            # Move inputs to the same device as model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Get model outputs
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get predicted class (0, 1, or 2)
            predicted_class = torch.argmax(logits, dim=1).item()
            
            # Map numeric predictions to class labels
            label_map = {0: "Quantum mechanics", 1: "Biophysics", 2: "High-energy physics"}
            predicted_label = label_map[predicted_class]
            
            # Store prediction and true label
            predictions.append(predicted_label)
            true_labels.append(row['topic'])
    
    return predictions, true_labels

# Example usage (to be run in notebook):
if 'test_df' not in locals():
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['topic'])

# Run evaluation
predictions, true_labels = evaluate_model(model, tokenizer, test_df)

# Calculate and display metrics
print("\nClassification Report:")
print(classification_report(true_labels, predictions))

# Create confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(true_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=['Quantum mechanics', 'Biophysics', 'High-energy physics'],
            yticklabels=['Quantum mechanics', 'Biophysics', 'High-energy physics'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"\nOverall Accuracy: {accuracy:.4f}")