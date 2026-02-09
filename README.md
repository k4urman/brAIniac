# Cancer Severity Prediction Neural Network

A PyTorch-based feedforward neural network for predicting cancer severity levels from medical tabular data, with automatic patient risk stratification using K-means clustering.

##  Overview

This project uses deep learning to predict cancer severity scores (0-1 scale) from patient medical records. The model is trained on NIH Glioblastoma dataset containing 21,634 patient records with 164 clinical features.

### Key Features

- **Deep Neural Network**: 3 hidden layers with batch normalization and dropout
- **Automatic Risk Stratification**: K-means clustering into Safe 🟢, Moderate 🟡, and Severe 🔴 groups
- **Continuous Training**: Save/load checkpoints and track training history across sessions
- **Comprehensive Visualizations**: Training curves, prediction analysis, and patient clustering plots
- **High Accuracy**: Achieves ~99.4% accuracy (0.006 MAE on 0-1 scale)

##  Model Architecture

```
Input Layer:        22 features
Hidden Layer 1:     128 neurons (ReLU + BatchNorm + Dropout 0.3)
Hidden Layer 2:     64 neurons  (ReLU + BatchNorm + Dropout 0.3)
Hidden Layer 3:     32 neurons  (ReLU + BatchNorm + Dropout 0.2)
Output Layer:       1 neuron    (Sigmoid activation)

Total Parameters:   13,761
Scaling Rule:       Pyramidal (÷2 per layer)
```

##  Input Features (22)

### Demographics (4)
- Gender, Race, Ethnicity, Age at diagnosis

### Clinical Features (12)
- Vital status, Tumor grade, Morphology, Site of biopsy
- Laterality, Prior malignancy, Prior treatment, Another malignancy
- Metastasis, Disease status, Progression, WHO grade

### Lifestyle & Health (6)
- Alcohol history, Alcohol intensity, Tobacco frequency
- Tobacco onset, Days to death, Karnofsky performance score

##  Severity Score Calculation (0-1 Scale)

The severity level is computed from 5 weighted clinical factors:

| Factor | Weight | Values |
|--------|--------|--------|
| **Tumor Grade** | 30% | G1: 0.05, G2: 0.15, G3: 0.25, G4: 0.30 |
| **Vital Status** | 25% | Alive: 0.0, Dead: 0.25 |
| **Metastasis** | 20% | Yes: 0.20, No: 0.0 |
| **Prior Malignancy** | 15% | Yes: 0.15, No: 0.0 |
| **Disease Status** | 10% | With tumor: 0.10, Tumor free: 0.0 |

**Total Range**: 0.0 (safest) to 1.0 (most severe)

##  Installation

### Prerequisites

```bash
Python 3.7+
```

### Install Dependencies

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib
```

Or for CPU-only PyTorch (smaller, faster):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy scikit-learn matplotlib
```

##  Project Structure

```
.
├── cancer_severity_prediction.ipynb    # Main Jupyter notebook
├── Downloads/
│   └── NIH Glioblastoma data.csv      # Input dataset (21,634 patients)
├── medical_cancer_severity_model.pth  # Latest trained model
├── best_model.pth                     # Best performing model
├── training_history.json              # Training session logs
└── patient_clusters.csv               # K-means cluster assignments
```

##  Usage

### 1. Prepare Your Data

Place your CSV file with the required columns (see Input Features section).

Update the file path in **Cell 8**:

```python
csv_file_path = 'Downloads/NIH Glioblastoma data.csv'
```

### 2. Run the Notebook

Open in Jupyter Notebook and run all cells:

```bash
jupyter notebook cancer_severity_prediction.ipynb
```

Or click: **Kernel → Restart & Run All**

### 3. Training Output

```
Epoch [10/100], Loss: 0.006933
Epoch [20/100], Loss: 0.007218
...
Epoch [100/100], Loss: 0.006926

Evaluation Results (0-1 scale):
Mean Absolute Error (MAE): 0.0058
Root Mean Squared Error (RMSE): 0.0117

 NEW BEST MODEL! MAE improved to 0.0058
```

### 4. Continue Training (Optional)

Run **Cell 17b** to train for additional epochs:

```python
continue_training(model, train_loader, test_loader, criterion, 
                 optimizer, device, checkpoint, additional_epochs=50)
```

## 📈 Understanding the Metrics

### Loss Function
- **Type**: L1Loss (Mean Absolute Error)
- **Range**: 0.0 to 1.0
- **Interpretation**: Average prediction error
  - Loss < 0.01 = Excellent (< 1% error)
  - Loss 0.01-0.05 = Good
  - Loss > 0.05 = Needs improvement

### Epochs
- **Definition**: One complete pass through all training data
- **Example**: 100 epochs = model reviews all 17,307 patients 100 times
- **Purpose**: Each pass refines the model's understanding

### Performance Metrics
- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more
- **Current Performance**: MAE ~0.006 (99.4% accuracy)

## 🔬 K-Means Clustering

Automatically groups patients into risk categories:

### Risk Categories

| Category | Severity Range | Description |
|----------|---------------|-------------|
| 🟢 **Safe** | 0.0 - 0.35 | Low risk, favorable prognosis |
| 🟡 **Moderate** | 0.35 - 0.55 | Medium risk, requires monitoring |
| 🔴 **Severe** | 0.55 - 1.0 | High risk, aggressive treatment needed |

### Output

```
CLUSTER ANALYSIS
Cluster 0 - SAFE 🟢
  Number of patients: 8,542
  Average severity: 0.298
  
Cluster 1 - MODERATE 🟡
  Number of patients: 7,123
  Average severity: 0.502
  
Cluster 2 - SEVERE 🔴
  Number of patients: 5,969
  Average severity: 0.687
```

Results saved to `patient_clusters.csv`

##  Visualizations

The notebook generates 6 plots:

1. **Training Loss Curve** - Loss vs. epoch
2. **Predictions vs. Actual** - Scatter plot with diagonal line
3. **Error Distribution** - Histogram of prediction errors
4. **Patient Clusters (2D PCA)** - Left: by cluster, Right: by severity
5. **Cluster Severity Distributions** - 3 histograms (one per cluster)

##  Training History Tracking

Every training session is logged in `training_history.json`:

```json
{
  "sessions": [
    {
      "date": "2026-02-09 14:32:15",
      "mae": 0.0064,
      "rmse": 0.0129,
      "epochs": 100,
      "improved": true
    }
  ],
  "best_mae": 0.0064,
  "best_model_date": "2026-02-09 14:32:15"
}
```

##  Hyperparameters

You can adjust these in the notebook:

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| Learning Rate | Cell 13 | 0.001 | How fast the model learns |
| Batch Size | Cell 11 | 32 | Samples per training step |
| Epochs | Cell 14 | 100 | Training iterations |
| Hidden Layers | Cell 4 | 128, 64, 32 | Network architecture |
| Dropout | Cell 4 | 0.3, 0.3, 0.2 | Regularization strength |
| Loss Function | Cell 13 | L1Loss | MAE vs MSE |

##  Troubleshooting

### Graphs Not Showing
- Ensure `%matplotlib inline` is in Cell 1
- Restart kernel and run all cells

### JSON Decode Error
- Delete `training_history.json` and `training_history.json.backup`
- The notebook will create fresh files

### CUDA Out of Memory
- Reduce `batch_size` in Cell 11
- Model automatically uses CPU if GPU unavailable

### NaN Loss Values
- Check for missing data in CSV
- The preprocessing handles this automatically, but verify data quality

##  Example Predictions

```python
# Make predictions on new patient data
patient_data = X_test[0:1]  # Single patient
prediction = predict_severity(model, patient_data, device)

print(f"Predicted severity: {prediction[0][0]:.3f}")
# Output: Predicted severity: 0.487 (Moderate risk)
```

##  Model Performance

Tested on NIH Glioblastoma dataset:

- **Training samples**: 17,307 patients
- **Test samples**: 4,327 patients
- **Features**: 22 clinical variables
- **MAE**: 0.0064 (0.64% average error)
- **RMSE**: 0.0129
- **Prediction range**: 0.280 - 0.771

##  Contributing

To improve the model:

1. Add more features to `feature_mapping` in Cell 3
2. Adjust severity weighting in `preprocess_medical_data()`
3. Experiment with different architectures in Cell 4
4. Try different optimizers (SGD, RMSprop) in Cell 13

##  License

This project is for educational and research purposes.

##  Acknowledgments

- Dataset: NIH Glioblastoma Cancer Genome Atlas (TCGA-GBM)
- Framework: PyTorch 2.10.0
- Preprocessing: scikit-learn
- Visualization: matplotlib

---

**Last Updated**: February 2026  
**Version**: 1.0  
**Python**: 3.7+  
**PyTorch**: 2.10.0+
