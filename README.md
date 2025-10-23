# BLIP-2 Visual Question Answering with Attention  
Fine-tuned Retina VQA Pipeline using [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base)

---

## Overview
This repository implements a **Visual Question Answering (VQA)** pipeline built on the **BLIP-2** architecture from Salesforce.  
It is fine-tuned for **retinal image question-answering tasks** (RetinaVQA dataset), combining visual embeddings from fundus images with natural-language question inputs.

---

## Features
- **BLIP-2 Backbone** (`Salesforce/blip-vqa-base`) for multimodal reasoning  
-  Custom `VQADataset` for synchronized image–text processing  
-  Image normalization and resizing with `BlipImageProcessor`  
-  Text tokenization and attention mask handling with `BlipProcessor`  
-  Integrated DataLoader with collate function for batched training  
-  GPU/CPU-ready execution with PyTorch  
-  Visualization utilities for sample inspection  

---

##  Directory Structure
```
📂 RetinaVQA/
 ┣ 📁 train/
 ┣ 📁 validation/
 ┣ 📁 test/
 ┗ 📄 dataset_info.json
📄 Blip2_With_Attention.ipynb
📄 README.md
```

---

##  Model Architecture
- **Model:** `BlipForQuestionAnswering`  
- **Base:** `Salesforce/blip-vqa-base`  
- **Parameters:** ~247M  
- **Fusion:** Image–Text Cross-Attention Layers  
- **Loss Function:** Cross-Entropy  
- **Optimizer:** AdamW (lr=5e-5)  

---

##  Installation

### 1️ Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2️ Create a Virtual Environment
```bash
python -m venv vqa_env
source vqa_env/bin/activate   # On Windows: vqa_env\Scripts\activate
```

### 3️ Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install transformers datasets tqdm pillow matplotlib
```

---

##  Usage

### 1️ Prepare RetinaVQA Dataset
Ensure your dataset is stored in a Hugging Face-compatible format:
```bash
./RetinaVQA/
├── train/
├── validation/
└── test/
```

### 2️ Run the Script
```bash
Load into Jupyter Notebook and run cell by cell Blip2_With_Attention.ipynb
```

This will Allow to:
- Load `Salesforce/blip-vqa-base`
- Process RetinaVQA data
- Build PyTorch DataLoaders
- Visualize a random sample with question & answer
- Initialize the optimizer for fine-tuning

---

##  Sample Output
Example console output:
```
Question: What type of lesion is visible in the retina?
Answer: Microaneurysm
```
Image visualization will appear via Matplotlib.

---

## Future Extensions
-  Fine-tuning on domain-specific datasets  
-  Add attention visualization (e.g., Grad-CAM for multimodal layers)  
-  Integrate with BLIP-2 Q-Former for contextual embeddings  
-  Extend to multi-disease ophthalmic QA and report generation  

---

## References
- [Salesforce BLIP-2 on Hugging Face](https://huggingface.co/Salesforce/blip-vqa-base)  
- [Hugging Face Transformers Library](https://huggingface.co/docs/transformers)  
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

