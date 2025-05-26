# Gender_Classification_Project
A high-accuracy gender classification system using a custom CNN trained on the UTKFace dataset. Achieves 93.4% test accuracy with data augmentation and BatchNorm. Includes model training, evaluation, and a Flask web app for real-time predictions.

# Gender Classification using CNNs 

##  Features
- **High Accuracy**: 93.4% test accuracy on UTKFace dataset.
- **Lightweight CNN**: Only 1.2M parameters (vs ResNet50's 23.5M).
- Deployment: Flask web app for easy predictions.
- **Data Augmentation**: Robust to lighting/pose variations.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AashrithaSura/Gender_Classification_Project.git
   cd Gender_Classification_Project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the pretrained model:  
   [Google Drive Link](https://drive.google.com/file/d/17_TqsZtZ7AJ84DUCxnp14Tz0rHt4JVSx/view?usp=sharing)

---

##  Usage
### Run the Flask App
```bash
python src/app.py
```
Open `http://localhost:5000` in your browser to upload images and get predictions.


---

## Model Architecture
A 6-layer CNN with:
- **Convolutional Blocks**: 32 → 64 → 128 filters + BatchNorm.
- **Dropout**: 50% for regularization.
- **Optimizer**: Adam (LR=0.0001).
---

## Results
| Metric      | Male | Female |
|-------------|------|--------|
| Precision   | 92.5%| 94.4%  |
| Recall      | 94.8%| 92.0%  |
| F1-Score    | 93.6%| 93.2%  |


---

##  Future Work
- [ ] **Multi-Task Learning**: Add age/race prediction.
- [ ] **Edge Deployment**: Optimize with TensorFlow Lite.
- [ ] **Live Camera Feed**: Real-time video processing.


- **Dataset**: [UTKFace](https://susanqq.github.io/UTKFace/)





