# Audio-Guided Cross-Modal Fusion for fMRI-to-Video Reconstruction

This repository contains the official implementation of the paper:

**Audio-Guided Cross-Modal Fusion for fMRI-to-Video Reconstruction**  
Prosenjith Roy Shuvo, Zareen Tasneem  
Primeasia University, Dhaka, Bangladesh

📄 *Under review*  
📅 December 2025

---

## Overview

Reconstructing dynamic video content from fMRI is fundamentally constrained by the **low temporal resolution** of neural measurements (TR ≈ 2s).  
Existing fMRI-to-video approaches rely almost exclusively on neural signals, which limits accurate motion estimation and leads to temporally unstable reconstructions.

This work introduces an **audio-guided cross-modal fusion framework** that uses **synthesized audio as an auxiliary temporal prior** to compensate for missing high-frequency temporal information in fMRI.

> **Important:**  
> The synthesized audio is **not assumed to be neurally encoded**.  
> It is used purely as an **engineered, model-level temporal prior**.

---

## Key Idea

- fMRI provides **coarse spatial and semantic information**
- Synthesized audio provides **dense temporal structure**
- Cross-attention fuses both modalities
- Motion is predicted from the fused representation
- Motion conditions a frozen video diffusion model

This improves:
- Temporal smoothness
- Motion stability
- Object coherence
- Perceptual video quality

---

## Pipeline Overview

```
Stimulus Video
│
├── fMRI → PCA → fMRI Embedding
│
├── Caption → AudioLDM → Synthesized Audio
│ ↓
│ Wav2Vec2
│ ↓
│ Audio Embedding
│
└── Cross-Attention Fusion (fMRI + Audio)
↓
Motion Decoder
↓
Motion-Guided Video Diffusion
↓
Reconstructed Video
```
---

## Repository Structure

```
audio-guided-fmri-video-reconstruction/
│
├── configs/ # YAML experiment configurations
│ ├── fmri_embedding.yaml
│ ├── audio_embedding.yaml
│ ├── fusion.yaml
│ ├── motion_decoder_fmri.yaml
│ ├── motion_decoder_fusion.yaml
│ ├── video_reconstruction.yaml
│ ├── eval_video_reconstruction.yaml
│ └── eval_video_qualitative.yaml
│
├── scripts/ # Executable experiment scripts
│ ├── run_fmri_embeddings.py
│ ├── run_audio_embeddings.py
│ ├── run_fusion_embeddings.py
│ ├── train_motion_decoder_fmri.py
│ ├── train_motion_decoder_fusion.py
│ ├── generate_videos.py
│ ├── evaluate_video_reconstruction.py
│ ├── evaluate_video_qualitative.py
│ └── summarize_qualitative_eval.py
│
├── src/ # Core library code
│ ├── fmri/ # fMRI preprocessing & embeddings
│ ├── audio/ # Audio synthesis & embeddings
│ ├── fusion/ # Cross-attention fusion modules
│ ├── motion/ # Motion decoders
│ ├── video/ # Video generation & evaluation
│ │ ├── generation.py
│ │ ├── evaluation.py
│ │ ├── metrics.py
│ │ ├── plotting.py
│ │ └── qualitative.py
│ ├── pipeline/ # End-to-end orchestration
│ └── utils/
│
├── notebooks/ # Colab / analysis notebooks
│ └── 11.2_qualitative_evaluation_of_reconstructed_videos.ipynb
│
├── outputs/
│ ├── logs/
│ └── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```
---

## Installation

### Local / Colab Setup

```
git clone https://github.com/prosenjith/audio-guided-fmri-video-reconstruction.git
cd audio-guided-fmri-video-reconstruction
pip install -r requirements.txt
```

### For Google Colab:
```
from google.colab import drive
drive.mount('/content/drive')
```

---
## Data

**Dataset:** Dynamic Natural Vision (DNV)  
**fMRI:** Whole-brain 7T recordings  
**Stimuli:** Naturalistic videos (3–5s segments)  
**Audio:** Synthesized per video segment using AudioLDM  

Due to licensing constraints, raw fMRI data and stimulus videos are **not included** in this repository.

---

## Expected Directory Layout (Example)

```text
MyDrive/Research/
├── data/
│   ├── fmri/
│   ├── stimuli/
│   │   └── videos/
│   └── audio/
├── models/
└── evaluation/
```
---

## Running the Pipeline

1. fMRI Embeddings
`
python scripts/run_fmri_embeddings.py --config configs/fmri_embedding.yaml
`

2. Audio Embeddings
`
python scripts/run_audio_embeddings.py --config configs/audio_embedding.yaml
`

3. Fusion Embeddings
```python scripts/run_fusion_embeddings.py --config configs/fusion.yaml```

4. Train Motion Decoders
```
python scripts/train_motion_decoder_fmri.py
python scripts/train_motion_decoder_fusion.py
```

5. Video Generation
```python scripts/generate_videos.py```

---

## Evaluation

**Quantitative Evaluation**

```python scripts/evaluate_video_reconstruction.py```

Metrics:
- SSIM ↑
- PSNR
- LPIPS ↓

**Qualitative Evaluation**

The qualitative pipeline follows a timeline-based human evaluation protocol:
- Uniform frame sampling
- Timeline visualization (GT vs fMRI-only vs Fusion)
- Manual annotation
- CSV-based logging
```
python scripts/evaluate_video_qualitative.py
python scripts/summarize_qualitative_eval.py
```

Qualitative criteria:
- Motion alignment
- Temporal smoothness
- Object coherence
- Overall perceptual preference

Results Summary
- SSIM: +2.01%
- LPIPS: +1.93%
- PSNR: −0.84% (expected perceptual trade-off)
- Human preference: Fusion preferred in 63–78% of samples
- Fusion consistently improves temporal stability and motion coherence.

---

## Design Philosophy

- ❌ No claim of auditory neural decoding  
- ✅ Audio used as an engineered temporal prior  
- ✅ Frozen diffusion model (engineering simplicity)  
- ✅ Focus on robustness, interpretability, and reproducibility  

---

## Limitations

- Depends on synthesized audio quality  
- Motion conditioning is indirect  
- Limited subject count  
- Short video segments  

---

## Citation

If you use this work, please cite:

```bibtex
@article{shuvo2025audioguided,
  title   = {Audio-Guided Cross-Modal Fusion for fMRI-to-Video Reconstruction},
  author  = {Shuvo, Prosenjith Roy and Tasneem, Zareen},
  journal = {Under Review},
  year    = {2025}
}
```

## Contact

**Prosenjith Roy Shuvo**  
📧 **royprosenjith@gmail.com**

For questions, issues, or collaboration, feel free to open an issue or contact me directly.

---
