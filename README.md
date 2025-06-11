### Download FLORES-200 dataset

```bash
pip install datasets
python src/download_data.py
```

### Run the char-level "prepare" step with the prepare.py

```bash
# Process all .txt files in data/flores directory and save outputs to work directory
python prepare.py --input_dir=data/flores --output_dir=work
```

This will:
- Scan all .txt files in the input_dir (data/flores by default)
- Create character-to-index mappings 
- Create `train.bin`, `val.bin`, and `meta.pkl` in the output_dir (work by default)

### Train the model

**IMPORTANT**: The `config/train_char.py` file contains optimized hyperparameters for Flores 200 dataset. 
Use minimal command line overrides to preserve the tuned configuration.

For MacBook with M1/M2 chip (RECOMMENDED):
```bash
python train.py config/train_char.py \
  --out_dir=work \
  --device=mps \
  --compile=False \
  --dtype=float32
```

For NVIDIA GPU systems:
```bash
python train.py config/train_char.py \
  --out_dir=work \
  --device=cuda \
  --compile=True \
  --dtype=bfloat16
```

For CPU-only systems:
```bash
python train.py config/train_char.py \
  --out_dir=work \
  --device=cpu \
  --compile=False \
  --dtype=float32 \
  --batch_size=8 \
  --gradient_accumulation_steps=8
```

**Configuration Details:**
- Model: 12 layers, 12 heads, 768 embedding dimensions (~90M parameters)
- Context: 512 characters (2x more than basic config)
- Training: 15,000 iterations with optimized learning rate schedule
- Expected accuracy: 60-70% (vs 18% with old config)
- Training time: ~3-4 hours on M1, ~1-2 hours on modern GPU

**Key Notes:**
- `--device=mps` enables GPU acceleration on Apple Silicon chips
- `--device=cuda` for NVIDIA GPUs with higher batch sizes and compilation
- `--out_dir=work` saves checkpoint to the same directory as the data files
- **Don't override** `max_iters`, `block_size`, or `batch_size` unless necessary (they're optimized in config)


NOTE: I have removed the training option from `myprogram.py` since it was janky and using the subprocess. 

### Predicting using `myprogram.py` script

```bash
python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output output/pred.txt
```
> Values are hardcoded for now. Need to make this dynamic.

### Evaluating prediction accuracy
```bash
python grade.py output/pred.txt example/answer.txt --verbose
```

### Submitting instructions 

Run `submit.sh` to package the code into submit directory 
  
### Validating docker build before submitting
```bash
docker run --rm -v $PWD/src:/job/src -v $PWD/work:/job/work -v $PWD/example:/job/data -v $PWD/output:/job/output cse517-proj/demo bash /job/src/predict.sh /job/data/input.txt /job/output/pred.txt
```


## Checkpoint 3
- Make sure dockerfile runs to spec - refer to feedback from Checkpoint 2
  - [x] code cleanup / refactor 
  - [x] Changed block size to 256 according to feedback from TAs
  
- Train on multiple datasets - different languages
    - prepare datasets - do we consolidate all the datasets?
  
- Figuring out how/where the checkpoint is stored and how's it used for prediction
  - Moved all the required embeddings and checkpoint files under `/work` directory.

