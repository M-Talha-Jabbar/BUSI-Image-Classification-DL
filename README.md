# BUSI-Image-Classification-DL

This repository showcases a deep learning project focused on the classification of breast ultrasound images to aid in breast cancer detection. This project explores the progression from a foundational Convolutional Neural Network (CNN) model to an enhanced solution incorporating transfer learning, data augmentation, and regularization techniques to achieve robust performance in medical image analysis.

## Dataset

The project utilizes the **Breast Ultrasound Images (BUSI) Dataset**. This dataset comprises 780 breast ultrasound images from 600 female patients, ranging in age from 25 and 75 years. Images are categorized into 'normal', 'benign' (non-cancerous), and 'malignant' (cancerous) classes. The images are approximately 500x500 pixels in PNG format, with corresponding ground truth information.

**Sample Images from the BUSI Dataset:**

<img width="488" alt="image" src="https://github.com/user-attachments/assets/dc9a9b75-f5d9-4549-b6ac-fda6b9cef65b" />

## Project Setup and Data Preprocessing

The development environment for this project was Google Colab, leveraging its T4 GPU for efficient model training. Google Drive was mounted to access the dataset.

Key data preprocessing steps included:
* **Image Loading & Resizing**: All images were loaded and uniformly resized to 224x224 pixels with 3 color channels.
* **Normalization**: Pixel intensity values were scaled from their original 0-255 range to a [0, 1] range.
* **Label Encoding**: Textual class labels ('benign', 'malignant', 'normal') were converted into numerical categorical formats.
* **Data Splitting**: The dataset was systematically divided into training (499 images), validation (125 images), and testing (156 images) subsets (approximately 64% train, 16% validation, 20% test) to facilitate robust model evaluation and generalization.
* **Reproducibility**: A fixed random seed (42) was consistently applied across all operations to ensure the reproducibility of results.

## Model Development Phase 1: Simple CNN Baseline

### Approach

A basic Convolutional Neural Network (CNN) was developed as a baseline for image classification.

### Model Architecture and Parameters

* **Model Architecture**: A sequential CNN model consisting of two convolutional layers (32 and 64 filters), each followed by a max-pooling layer, then a flatten layer, and finally two dense (fully connected) layers (128 units and 3 output units for classification).
* **Training Parameters**: Learning Rate: 0.0001, Epochs: 100, Batch Size: 16, Optimizer: Adam, Activation Functions: ReLU for convolutional layers and Softmax for the output layer.

### Results

| Model | Training Accuracy | Validation Accuracy | Testing Accuracy |
| :---- | :---------------- | :------------------ | :--------------- |
| CNN   | 100.00%           | 69.60%              | 74.36%           |

### Loss and Accuracy Plots

<img width="622" alt="image" src="https://github.com/user-attachments/assets/8d7818ba-fce0-4ab9-8571-838062804496" />

### Observations

The baseline CNN model achieved a testing accuracy of approximately 74.36%. A significant observation was **overfitting**, where the model performed exceptionally well on training data (100% accuracy) but showed a notable drop in performance on validation and test sets. This behavior was visually confirmed by the training accuracy curves separating from validation accuracy, and validation loss continuing to exceed training loss. Potential contributing factors included the relatively small dataset size (780 images) and inherent complexities like noise and low contrast in ultrasound images.

## Model Development Phase 2: Performance Enhancement with Transfer Learning

### Objective

To significantly enhance the model's performance and achieve a testing accuracy of 85% or above.

### Chosen Model and Rationale

* **Chosen Model**: ResNet50.
* **Rationale**: ResNet50 was selected for its proven high performance in image classification, its deep architecture, and its effective use of **residual connections** to mitigate the vanishing gradient problem in deep networks. The availability of pre-trained weights on ImageNet allowed for **transfer learning**, enabling the model to leverage features learned from a large, diverse dataset, thus improving performance on the BUSI dataset.

### Methodology

* A pre-trained ResNet50 model (without its original top classification layers) was loaded.
* Custom classification layers (GlobalAveragePooling2D, Dense with 1024 ReLU units, Dropout with 0.5 rate, and a final Dense Softmax layer for 3 classes) were added on top of the ResNet50 base.
* Extensive **data augmentation** (including rotation, width/height shifts, shear, zoom, and horizontal flips) was applied during training to increase data diversity and improve generalization.
* A two-phase training strategy was implemented: initially, the ResNet50 base layers were frozen, and only the newly added classification layers were trained. Subsequently, all layers of the ResNet50 base model were unfrozen, and the entire model was **fine-tuned** with a low learning rate.

### Results

| Model    | Training Accuracy | Validation Accuracy | Testing Accuracy |
| :------- | :---------------- | :------------------ | :--------------- |
| ResNet50 | 99.00%            | 87.20%              | 89.74%           |

### Loss and Accuracy Plots

<img width="863" alt="image" src="https://github.com/user-attachments/assets/74cff540-d84a-4d29-b921-52e3a2a09741" />

### Observations

The ResNet50 model successfully achieved the target of over 85% testing accuracy, reaching 89.74%. Both training and validation accuracies showed consistent improvement, and losses decreased, indicating effective learning and generalization. The use of data augmentation significantly contributed to the high accuracy by diversifying the dataset. The model exhibited less obvious overfitting compared to the initial CNN baseline.

## Model Development Phase 3: Overfitting Mitigation

### Objective

To specifically mitigate overfitting observed in previous models (where training accuracy quickly approached 100% and validation accuracy could decrease), without altering the batch size, number of iterations (epochs), or learning rate.

### Chosen Techniques and Rationale

* **Chosen Techniques**: L2 Regularization and Dropout.
* **Rationale**: **L2 regularization** was applied to the dense layers to penalize large weights, thereby preventing the model from becoming too complex and overfitting the training data. **Dropout** was implemented in the dense layers (with a rate of 0.5) to randomly deactivate neurons during training, reducing their co-adaptation and forcing the model to learn more robust features.

### Methodology

The ResNet50 model from Phase 2 was retained. L2 regularization (with a coefficient of 0.01) was added to the dense layers, and the dropout rate was set to 0.5. The two-phase training and fine-tuning approach from Phase 2 was repeated with these additional regularization techniques.

### Results

| Model                               | Training Accuracy | Validation Accuracy | Testing Accuracy |
| :---------------------------------- | :---------------- | :------------------ | :--------------- |
| ResNet50 + Dropout + L2 Regularization | 97.60%            | 80.80%              | 85.26%           |

### Loss and Accuracy Plots

<img width="864" alt="image" src="https://github.com/user-attachments/assets/4ecda5e5-03d6-4579-ad42-1c5b807cfc7b" />

### Observations

The integration of L2 regularization and Dropout techniques successfully **reduced overfitting**, as indicated by the training accuracy decreasing from 99.00% (in Phase 2) to 97.60%, while still maintaining a strong test accuracy of 85.26%. The validation accuracy of 80.80% indicated improved generalization, showcasing the effectiveness of the overfitting mitigation strategies. Some remaining discrepancy between training and validation accuracy might suggest further underlying issues or complexities within the BUSI dataset, potentially related to noise or class imbalance.
