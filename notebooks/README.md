## Notebooks Overview

> ⚠️ These notebooks are lightweight wrappers.
> Do not edit model logic here — see `scripts/` and `src/`.

Each notebook corresponds to a single stage of the pipeline and can be run independently after its prerequisites.

All computation logic lives in `scripts/` and `src/`.
Notebooks only orchestrate execution and visualization.

Execution order:
1. 00_environment_check.ipynb
2. 01_fmri_embeddings.ipynb
3. 02_audio_embeddings.ipynb
4. 03_fusion_embeddings.ipynb
5. 04_motion_decoders.ipynb
6. 05_video_reconstruction.ipynb
7. 06_quantitative_evaluation.ipynb
8. 07_qualitative_evaluation.ipynb
