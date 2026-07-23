# Lung X-Ray Classification Using Deep Learning

A deep learning image classification project comparing custom Convolutional Neural Networks (CNNs) with pretrained architectures for classifying chest X-ray images into **Normal**, **Opacity**, and **Pneumonia** categories.

The project explores custom CNN design, transfer learning, fine-tuning, and model evaluation using PyTorch. The best-performing model, a fine-tuned **EfficientNet-B0**, achieved **93.3% test accuracy** and a **0.934 macro F1-score**.

## Project Overview

The goal of this project was to investigate how custom-built CNN architectures compare with pretrained deep learning models for multi-class chest X-ray classification.

The project involved:

* Building and training custom CNN architectures from scratch
* Experimenting with multiple established CNN architectures
* Applying transfer learning and fine-tuning
* Comparing predictive performance and parameter efficiency
* Evaluating the final model using accuracy, precision, recall, F1-score, and a confusion matrix

Several pretrained architectures were explored, including:

* AlexNet
* VGG16
* ResNet18
* DenseNet121
* GoogLeNet
* ConvNeXt Tiny
* RegNet
* EfficientNet-B0

EfficientNet-B0 with fine-tuning produced the strongest overall performance.

## Dataset and Preprocessing

The dataset contained **3,475 chest X-ray images** across three classes:

| Class     | Images |
| --------- | -----: |
| Normal    |  1,250 |
| Opacity   |  1,125 |
| Pneumonia |  1,100 |

The data was divided into training, validation, and test sets using a **70/15/15 split**.

Preprocessing included:

* Resizing images to `224 × 224`
* Converting images to grayscale
* Replicating grayscale images across three channels for compatibility with pretrained models
* Pixel normalisation
* Random horizontal flipping and small rotations for training-data augmentation

Augmentation was applied only to the training set to improve generalisation while maintaining unbiased validation and test evaluation.

## Model Development

### Custom CNNs

Two custom CNN architectures were developed and trained from scratch.

The initial CNN provided a lightweight baseline with approximately **306K trainable parameters**.

A deeper architecture was then developed using additional convolutional layers, batch normalisation, and a revised classification stage. This improved validation performance but substantially increased the number of trainable parameters.

### Transfer Learning

Multiple established CNN architectures were investigated using pretrained models.

Two main transfer-learning strategies were explored:

**Feature Extraction**

Most pretrained layers were frozen, with only the final classification layers trained on the chest X-ray dataset.

**Fine-Tuning**

Higher-level pretrained layers were allowed to update during training, enabling learned image features to adapt more closely to the chest X-ray classification task.

Fine-tuning consistently produced stronger results, with EfficientNet-B0 achieving the best overall validation performance.

## Model Comparison

| Model               | Training Strategy  | Trainable Parameters | Validation Accuracy |   Macro F1 |
| ------------------- | ------------------ | -------------------: | ------------------: | ---------: |
| Original Custom CNN | From scratch       |              306,147 |              87.14% |     0.8738 |
| Improved Custom CNN | From scratch       |           12,934,259 |              89.64% |     0.8982 |
| EfficientNet-B0     | Feature extraction |                3,843 |              88.87% |     0.8913 |
| **EfficientNet-B0** | **Fine-tuning**    |        **3,159,583** |          **94.05%** | **0.9418** |

The improved custom CNN outperformed the original architecture, although this came at a significant increase in model complexity.

Fine-tuned EfficientNet-B0 achieved the strongest validation results while using substantially fewer trainable parameters than the improved custom CNN, demonstrating the effectiveness of transfer learning for this task.

## Final Results

The fine-tuned **EfficientNet-B0** was selected for final evaluation on the unseen test set.

| Metric              | Test Result |
| ------------------- | ----------: |
| **Accuracy**        |  **93.30%** |
| **Macro Precision** |  **0.9352** |
| **Macro Recall**    |  **0.9337** |
| **Macro F1-Score**  |  **0.9340** |

The model demonstrated strong performance across all three classes rather than achieving high accuracy through a single dominant category.

Pneumonia was classified particularly effectively, achieving an F1-score of **0.9850**.

The primary classification challenge was distinguishing between **Normal** and **Opacity** X-rays, suggesting that more subtle opacity patterns were harder for the model to differentiate from healthy scans.

## Technologies Used

* Python
* PyTorch
* Torchvision
* Convolutional Neural Networks (CNNs)
* Transfer Learning
* EfficientNet
* Image preprocessing and augmentation
* Scikit-learn
* Matplotlib
* Jupyter Notebook / Google Colab

## Repository Structure

```text
lung-xray-classification/
├── README.md
├── notebooks/
│   └── lung_xray_classification.ipynb
├── report/
│   └── report.pdf
└── requirements.txt
```

## Running the Project

Clone the repository:

```bash
git clone <your-repository-url>
cd lung-xray-classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook notebooks/lung_xray_classification.ipynb
```

The dataset is not included in this repository. The notebook may require dataset paths to be updated before running in a local environment.

## Key Takeaways

This project demonstrated the effectiveness of transfer learning for medical image classification on a relatively limited dataset.

While increasing the complexity of a custom CNN improved its predictive performance, fine-tuning a pretrained EfficientNet-B0 achieved better results with fewer trainable parameters than the improved custom architecture.

The project also highlighted the importance of evaluating models beyond accuracy alone. Macro precision, recall, F1-score, and class-level performance were used to assess how reliably the model performed across all three diagnostic categories.

## Further Improvements

Potential extensions to the project include:

* Exploring more advanced image augmentation techniques
* Using learning-rate scheduling and systematic hyperparameter optimisation
* Investigating class-sensitive loss functions
* Adding explainability methods such as Grad-CAM to visualise image regions influencing predictions
* Comparing additional lightweight architectures for deployment efficiency

## Disclaimer

This project was developed for academic and educational purposes. The models and results presented here are experimental and are **not intended for clinical diagnosis or medical decision-making**.
