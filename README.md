# **ReplicantNet** 🏙️🧠  

## **Overview**  
ReplicantNet is a **vision-based AI model** designed to classify images from the **Pascal VOC 2008 dataset** while exploring **synthetic data generation** to enhance classification accuracy. Using **transfer learning with VGGNet**, we analyze classification performance and further augment training data using **Variational AutoEncoders (VAEs) and Generative Adversarial Networks (GANs)**. The project evaluates the impact of AI-generated data on model performance using **Mean Average Precision (mAP)** as the primary metric.  

---

## **Features**  
✅ **Transfer Learning with VGGNet** – Fine-tuned for Pascal Classification  
✅ **Synthetic Data Generation** – Using **VAE** and **GANs**  
✅ **Performance Evaluation** – Measuring impact of augmentation on mAP  
✅ **Comparative Analysis** – Studying real vs. AI-generated data  

---

## **Dataset**  
- **Source:** [Pascal VOC 2008](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/index.html)  
- **Split Used:** `train` → Training, `val` → Testing  

---

## **Methodology**  

### **1. Transfer Learning with VGGNet**  
- Implemented **VGGNet** pre-trained on **ImageNet**  
- Fine-tuned for Pascal classification  
- Evaluated on **Mean Average Precision (mAP)**  
- Showcased **Top 10 ranked classified images**  

### **2. Variational AutoEncoder (VAE) Augmentation**  
- Trained a **VAE** model to generate synthetic images per class  
- Augmented dataset with **100, 200, and 500 samples per class**  
- Retrained classifier and analyzed performance changes  

### **3. Generative Adversarial Networks (GANs) Augmentation**  
- Trained a **GAN** to generate realistic images for each class  
- Evaluated performance improvement using the **augmented dataset**  
- Analyzed cases where augmentation helps (or fails)  

### **4. Comparative Analysis & Discussion**  
- **Performance comparison:** Original vs. VAE-augmented vs. GAN-augmented  
- **Curves & Visualization:** mAP scores across different training sets  
- **Discussion:** Why and why not synthetic data improves classification  

---

## **Project Structure**  
```
ReplicantNet/
│── data/                     # Pascal VOC dataset
│── models/                   # Trained VGG, VAE, GAN models
│── notebooks/                # Jupyter Notebooks for training & analysis
│── results/                  # Evaluation metrics & visualization
│── report/                   # LaTeX Report
│── src/                      # Source Code
│   ├── train_vgg.py          # Transfer Learning with VGGNet
│   ├── train_vae.py          # VAE training & augmentation
│   ├── train_gan.py          # GAN training & augmentation
│   ├── evaluate.py           # Model Evaluation
│── README.md                 # Project Documentation
```

---

## **Installation & Usage**  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/yourusername/ReplicantNet.git
cd ReplicantNet
pip install -r requirements.txt
```

### **Train Models:**  
```bash
python src/train_vgg.py    # Train VGGNet on Pascal VOC
python src/train_vae.py    # Train VAE and generate synthetic data
python src/train_gan.py    # Train GAN and generate synthetic data
python src/evaluate.py     # Evaluate models & generate performance reports
```

---

## **Results & Findings**  
📌 **Comparison of real vs. AI-generated training data**  
📌 **Analysis of synthetic data effectiveness for classification**  
📌 **Visualizations of training curves & mAP metrics**  
