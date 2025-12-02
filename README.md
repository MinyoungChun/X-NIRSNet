# X-NIRSNet: Novel Deep Learning Architecture Using Cross-Residual Blocks for Decoding Mental States with Functional Near-Infrared Spectroscopy

![X-NIRSNet Architecture Diagram](./figures/model.jpg)

# Abstract
Background: Functional near-infrared spectroscopy (fNIRS) is increasingly being used in passive brain–computer interface (pBCI) systems to decode users’ various cognitive and affective states in real-world scenarios. Although deep learning (DL) approaches have been applied to improve the decoding accuracy, many existing fNIRS-based DL models continue to rely on handcrafted features or employ architectures that fail to capture complex interrelations between hemodynamic signals.
Methods: In this study, a novel DL architecture named X-NIRSNet that incorporates cross-residual blocks (cross-ResBlocks) to extract complementary features between oxy-hemoglobin (ΔHbO) and deoxy-hemoglobin (ΔHbR) changes was proposed. The model was validated on two prefrontal fNIRS datasets (stress and mental fatigue) using a subject-independent classification scheme to ensure generalizability. In addition, the spatiotemporal relevance of the model decisions was visualized using gradient-weighted class activation mapping (Grad-CAM).
Results: X-NIRSNet achieved classification accuracies of 88.57% for stress and 76.39% for mental fatigue, outperforming three widely used DL architectures including a state-of-the-art model. Grad-CAM analyses revealed that the model focused on neurophysiologically relevant regions, highlighting the dorsolateral prefrontal and orbitofrontal cortices for stress and mental fatigue, respectively. Furthermore, X-NIRSNet appeared to learn temporal transition patterns in the fNIRS signals, effectively functioning as a temporal edge detector.
Conclusions: The proposed X-NIRSNet enables accurate and interpretable decoding of mental states from fNIRS signals. By combining strong classification performance with neurophysiologically meaningful interpretability, X-NIRSNet demonstrates substantial potential for advancing fNIRS-based pBCI systems from controlled laboratory settings toward practical real-world applications.

1. Basic Run (Default Settings)
python main.py

2. Custom Experiment
python main.py --channels 13 --fs 6.138 --sec 60 --num_subjects 20 --trials_per_sub 40

| Argument | Default | Description |
| :---: | :---: | :---: |
| `--channels` | `13` | Number of fNIRS channels |
| `--fs` | `6.138` | Sampling rate (Hz) |
| `--sec` | `60` | Duration of each trial (seconds) |
| `--num_subjects` | `10` | Total number of subjects (for dummy generation) |
| `--trials_per_sub` | `20` | Number of trials per subject |
| `--num_classes` | `2` | Number of classes |
| `--batch_size` | `16` | Batch size for training |
| `--lr` | `0.001` | Learning rate |
| `--epochs` | `50` | Number of training epochs per fold |
| `--seed` | `912` | Random seed for reproducibility |
| `--filter_size` | `32` | Filter size |
| `--hidden_units` | `256, 128` | Hidden units |

```bibtex
@misc{chun2025xnirsnet,
  author = {Minyoung Chun, Hyerin Nam, Chaeyoon Kim, Wooseok Hyung, Chang-Hwan Im},
  title = {X-NIRSNet: Novel Deep Learning Architecture Using Cross-Residual Blocks for Decoding Mental States},
  year = {2025},
  publisher = {None},
  journal = {None},
  howpublished = {\url{[https://github.com/MinyoungChun/X-NIRSNet](https://github.com/MinyoungChun/X-NIRSNet)}}
}