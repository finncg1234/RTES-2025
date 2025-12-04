## Repository Contents

### GitHub Repository
**Main project folder:** [autoencoder_can](https://github.com/finncg1234/RTES-2025/tree/main/autoencoder_can)

### Core Files
| File | Description |
|------|-------------|
| `README.md` | Instructions for running code and links to deliverables |
| `autoencoder.py` | PyTorch autoencoder model definition (encoder, decoder, BFW) |
| `can_main.py` | Training and evaluation pipeline |
| `dataset.py` | Data preprocessing and feature extraction |
| `config.py` | Configuration classes and enumerations |

### Results
| Resource | Description |
|----------|-------------|
| `RTES-2025-stats.xlsx` | Results from all hyperparameter combinations |
| `out/` | Trained model weights and evaluation outputs |

### Deliverables
- **Final Report:** `RTES2025_final.pdf`
- **Final Presentation:** [YouTube Link](https://www.youtube.com/watch?v=gu0wGI2u8mg)


## Running the Autoencoder

### Training the Model

To train an autoencoder model on attack-free CAN bus data:
```bash
python main.py -train -v 2016-chevrolet-silverado -d extra-attack-free -l 0 -f 1 -b 32 -r 10 -e 50 -n 10 -model dummy
```

### Evaluating the Model

To evaluate a trained model on attack data:
```bash
python main.py -run -v 2016-chevrolet-silverado -d dos -l 1 -f 1 -b 32 -r 10 -e 50 -n 10 -model ./out/2016-chevrolet-silverado/weights/fe-1_b-32_input-size-60_epochs-50_lr-0_001.pth
```

### Training and Evaluating Together
```bash
python main.py -train -run -v 2016-chevrolet-silverado -d dos -l 1 -f 1 -b 32 -r 10 -e 50 -n 10 -model ./out/2016-chevrolet-silverado/weights/fe-1_b-32_input-size-60_epochs-50_lr-0_001.pth
```

### Command-Line Parameters

| Parameter | Required | Description | Example Values |
|-----------|----------|-------------|----------------|
| `-train` | No | Flag to train the model | (no value needed) |
| `-run` | No | Flag to evaluate the model | (no value needed) |
| `-nobfw` | No | Disable Batch-Wise Feature Weighting | (no value needed) |
| `-f` | Yes | Feature extraction strategy:<br>• `0` = Naive (no CAN data)<br>• `1` = Naive (with CAN data) | `0` or `1` |
| `-b` | Yes | Batch size for training | `16`, `32`, `64`, `128` |
| `-r` | Yes | Learning rate × 10⁻⁴<br>(e.g., `10` → 0.001) | `1`, `5`, `10`, `50` |
| `-e` | Yes | Number of training epochs | `50`, `100`, `200` |
| `-n` | Yes | Number of CAN messages per input vector | `5`, `10`, `20` |
| `-v` | Yes | Vehicle identifier | `2016-chevrolet-silverado` |
| `-d` | Yes | Dataset type/attack scenario | `extra-attack-free`, `dos`, `fuzzy`, `gear`, `rpm`, `speed`, `standstill`, `interval`, `combined` |
| `-l` | Yes | Dataset has labels:<br>• `0` = Unlabeled (training data)<br>• `1` = Labeled (test data) | `0` or `1` |
| `-model` | Yes | Path to model weights (when running)<br>Any value when training | `./out/.../model.pth` |

### Notes

- **Training data** should use `-l 0` (unlabeled) with `extra-attack-free` dataset
- **Test data** should use `-l 1` (labeled) with attack datasets (dos, fuzzy, etc.)
- The `-model` parameter is only used when `-run` is specified, but must be provided regardless
- Learning rate is scaled by 10⁻⁴, so `-r 10` becomes 0.001
- Input size is calculated automatically: `N × 6` (with CAN data) or `N × 2` (without)
- Results are saved to `./out/{vehicle}/{dataset}/results.csv`
- Model weights are saved to `./out/{vehicle}/weights/`

### Example Workflow

1. **Train on attack-free data:**
```bash
   python main.py -train -v 2016-chevrolet-silverado -d extra-attack-free -l 0 -f 1 -b 32 -r 10 -e 100 -n 10 -model dummy
```

2. **Evaluate on DoS attacks:**
```bash
   python main.py -run -v 2016-chevrolet-silverado -d dos -l 1 -f 1 -b 32 -r 10 -e 100 -n 10 -model ./out/2016-chevrolet-silverado/weights/fe-1_b-32_input-size-60_epochs-100_lr-0_001.pth
```

3. **Evaluate on all attack types:**
```bash
   for attack in dos fuzzy gear rpm speed standstill interval combined; do
       python main.py -run -v 2016-chevrolet-silverado -d $attack -l 1 -f 1 -b 32 -r 10 -e 100 -n 10 -model ./out/2016-chevrolet-silverado/weights/fe-1_b-32_input-size-60_epochs-100_lr-0_001.pth
   done
```

### Output Files

- **Training loss plot:** `{model_path}_training_loss.png`
- **Model weights:** `{model_path}.pth`
- **Results CSV:** `./out/{vehicle}/{dataset}/results.csv`
  - Contains: feature extraction, batch size, learning rate, epochs, messages per input, BFW status, AUC, max F1, threshold

### Feature Extraction Strategies

- **Strategy 0 (NO_CAN_DATA):** Only uses timestamp delta and CAN ID (2 features per message)
- **Strategy 1 (NAIVE):** Uses timestamp delta, CAN ID, and first 4 bytes of data field (6 features per message)

### Disabling BFW

To train/evaluate without Batch-Wise Feature Weighting, add the `-nobfw` flag:
```bash
python main.py -train -nobfw -v 2016-chevrolet-silverado -d extra-attack-free -l 0 -f 1 -b 32 -r 10 -e 50 -n 10 -model dummy
```

This allows direct comparison between models with and without BFW.
