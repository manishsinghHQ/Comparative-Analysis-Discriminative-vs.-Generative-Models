# Assignment 1 — Discriminative vs Generative Models
## CIFAR-10 · CNN Classifier vs VAE · PyTorch + Streamlit

---

## 🚀 Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

> Python 3.9+ recommended. CUDA optional (CPU works fine for 5-epoch runs).

---

## 📁 File Structure

```
assignment1_app/
├── app.py              ← Main Streamlit app (single file, all code inside)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── logs/               ← Auto-created at runtime
    ├── cnn_model.pt    ← Saved CNN weights
    ├── vae_model.pt    ← Saved VAE weights
    ├── cnn_metrics.json
    └── vae_metrics.json
```

---

## 🧠 What's Inside

### Models
| Model | Type | Architecture |
|-------|------|-------------|
| `CNN_Classifier` | Discriminative | ResNet-style CNN (ResBlocks, 3 stages, ~2M params) |
| `VAE` | Generative | Conv Encoder + Conv Decoder, β-VAE loss |

### App Tabs
| Tab | Content |
|-----|---------|
| 🔷 CNN Classifier | Train, metrics, per-class accuracy, confusion matrix |
| 🔶 VAE Generative | Train, loss curves, latent space PCA |
| 📊 Comparison | Side-by-side table + multi-plot analysis |
| 🎨 Generated Samples | VAE samples + latent space visualization |
| 📋 Technical Report | Full 4-page report (downloadable as Markdown) |

### Sidebar Controls
- CNN: epochs, learning rate
- VAE: epochs, learning rate, latent dim, β weight
- Reset button to clear all state

---

## 📦 Deliverables Covered

✅ Python code (single `app.py`)  
✅ Training logs (live in app + saved to `logs/`)  
✅ Generated sample images (VAE tab)  
✅ Technical report (Report tab, downloadable)  
✅ Training curves (Loss, Accuracy, KL, Recon)  
✅ Confusion matrix (CNN)  
✅ Latent space PCA visualization (VAE)  
✅ Per-class accuracy breakdown  
✅ Model comparison table  

---

## ⚡ Quick Start (5-min demo)

1. Sidebar → CNN: 5 epochs, lr=0.001 → **Train CNN**
2. Sidebar → VAE: 5 epochs, lr=0.001, latent=128, β=1.0 → **Train VAE**
3. Go to 📊 Comparison tab
4. Go to 🎨 Generated Samples
5. Go to 📋 Technical Report → Download

For full quality: 15-20 epochs each.
