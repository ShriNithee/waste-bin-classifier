# ♻️ Waste Bin Classifier

An image classification model that looks at a photo of a waste item, identifies what it is, and tells you **which bin it should go in**, following India's household waste segregation categories (wet, dry, domestic hazardous).

Built with **TensorFlow / Keras** using **transfer learning on MobileNetV2**, trained on Google Colab.

> This is the first module of a larger project: a civic platform for reporting and tracking garbage black spots.

---

## 🧩 The Problem

Waste segregation at source is required by India's Solid Waste Management Rules, but in practice most households mix their waste. Once wet, dry and hazardous waste are mixed, recycling becomes difficult and hazardous items like batteries end up in landfills. A big reason is simple confusion: people don't know which bin an item belongs in.

This model answers that question from a single photo.

---

## 🗂️ Dataset

**Garbage Classification (12 classes)** from Kaggle: [mostafaabla/garbage-classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

- 15,515 images across 12 classes
- Split: 80% training (12,412 images), 20% validation (3,103 images)

| Waste type | Images | Bin |
|---|---|---|
| biological | 985 | 🟢 Green – Wet waste |
| paper | 1,050 | 🔵 Blue – Dry recyclable |
| cardboard | 891 | 🔵 Blue – Dry recyclable |
| plastic | 865 | 🔵 Blue – Dry recyclable |
| metal | 769 | 🔵 Blue – Dry recyclable |
| white-glass | 775 | 🔵 Blue – Dry recyclable |
| green-glass | 629 | 🔵 Blue – Dry recyclable |
| brown-glass | 607 | 🔵 Blue – Dry recyclable |
| clothes | 5,325 | 🔵 Blue – Dry waste |
| shoes | 1,977 | 🔵 Blue – Dry waste |
| battery | 945 | 🔴 Red – Domestic hazardous |
| trash | 697 | ⚫ Black – Non-recyclable reject |

*The dataset is not included in this repository. It downloads automatically when you run the notebook.*

---

## 🧠 Approach

1. **Transfer learning**: started from MobileNetV2 pretrained on ImageNet (1.4M images), so the model already knows edges, textures and shapes.
2. **Data augmentation**: random flips, rotations and zoom to help the model generalise and reduce overfitting.
3. **Stage 1 – Feature extraction (5 epochs)**: base model frozen, only the new classification layer trained (learning rate 1e-3).
4. **Stage 2 – Fine-tuning (5 epochs)**: top 40 layers of MobileNetV2 unfrozen and trained with a small learning rate (1e-5).
5. **Bin mapping**: each predicted waste type is mapped to its disposal bin.
6. **Confidence threshold**: if the model is less than 60% confident, it says it's unsure instead of guessing.

**Model size:** 2.27M parameters (~8.7 MB), small enough to run in a web or mobile app.

---

## 📊 Results

| Metric | Score |
|---|---|
| **Waste-type accuracy** | **94.91%** |
| **Bin-level accuracy** | **98.78%** |
| Macro-average F1 | 0.93 |

### Per-class performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| battery | 0.97 | 0.99 | 0.98 |
| biological | 0.98 | 0.97 | 0.98 |
| brown-glass | 0.92 | 0.92 | 0.92 |
| cardboard | 0.93 | 0.95 | 0.94 |
| clothes | 0.99 | 0.99 | 0.99 |
| green-glass | 0.92 | 0.96 | 0.94 |
| metal | 0.77 | 0.93 | 0.84 |
| paper | 0.95 | 0.87 | 0.91 |
| plastic | 0.91 | 0.81 | 0.86 |
| shoes | 0.96 | 0.98 | 0.97 |
| trash | 0.97 | 0.97 | 0.97 |
| white-glass | 0.87 | 0.82 | 0.84 |

### Training progress

| Epoch | Stage | Train accuracy | Validation accuracy |
|---|---|---|---|
| 5 | Feature extraction | 93.81% | 93.59% |
| 6 | Fine-tuning starts | 88.24% | 93.81% |
| 10 | Fine-tuning ends | 93.85% | **94.91%** |

### Key observations

- **Hazardous waste is reliably caught**: battery recall of 0.99 means almost no batteries are sent to the wrong bin, which is the most important mistake to avoid.
- **Bin-level accuracy is higher than class accuracy** because many errors are between classes that share a bin (e.g. paper vs cardboard), so the user still gets the right answer.
- **Weakest classes are metal, plastic and white-glass**: shiny and transparent materials look visually similar.
- **No overfitting**: validation accuracy stays at or above training accuracy, since augmentation and dropout only apply during training.
- **Class imbalance**: clothes has ~5,300 images while brown-glass has ~600. Class weighting is a planned improvement.
- **Real-world gap**: on a real photo outside the dataset, the model predicted plastic with 68% confidence. Testing on real Indian household waste is the next step.


---

## 🚀 How to Run

1. Open `waste_classifier.ipynb` in [Google Colab](https://colab.research.google.com) (File → Upload notebook).
2. Enable the GPU: **Runtime → Change runtime type → T4 GPU**.
3. Run all cells from top to bottom. Training takes about 15 minutes.
4. In Step 11, upload your own photo of a waste item to get a bin recommendation.

To use the saved model elsewhere:

```python
import json, numpy as np
from tensorflow import keras

model = keras.models.load_model('waste_classifier.keras')
class_names = json.load(open('class_names.json'))

img = keras.utils.load_img('your_photo.jpg', target_size=(224, 224))
probs = model.predict(np.expand_dims(keras.utils.img_to_array(img), 0))[0]
print(class_names[np.argmax(probs)], f"{probs.max():.0%}")
```

---

## 📁 Repository Structure

```
waste-bin-classifier/
├── README.md
├── waste_classifier.ipynb   # training and evaluation notebook
├── waste_classifier.keras   # trained model
├── class_names.json         # class label order
└── results/                 # plots and sample predictions
```

---

## 🛣️ Next Steps

- [ ] Test on 50+ real photos of Indian household waste (milk packets, coconut shells, snack wrappers)
- [ ] Handle class imbalance with class weights
- [ ] Add classes: e-waste, sanitary waste, multi-layer packaging
- [ ] Compare MobileNetV2 with EfficientNetB0 and ResNet50
- [ ] Build a Django web app for photo upload and bin recommendation
- [ ] Extend to garbage detection in street photos (YOLOv8) for the black-spot reporting platform

---

## 🛠️ Tech Stack

Python · TensorFlow / Keras · MobileNetV2 · scikit-learn · Matplotlib · Google Colab · Kaggle

---

## 👤 Author

**ShriNithee** – Pre-final year B.E. Computer Science and Engineering
