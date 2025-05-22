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

For MacBook with M1/M2 chip:
```bash
python train.py config/train_char.py \
  --out_dir=work \
  --device=mps \
  --compile=False \
  --batch_size=8 \
  --block_size=256 \
  --max_iters=200 \
  --lr_decay_iters=2000 \
  --eval_interval=100
```

For other systems (CPU/CUDA):
```bash
python train.py \
  config/train_char.py \
  --out_dir=work \
  --device=cpu \       # or `cuda` if you have a NVIDIA GPU
  --compile=False \
  --batch_size=8 \
  --block_size=256 \
  --max_iters=2000 \
  --lr_decay_iters=2000
```

Note:
- `--device=mps` enables GPU acceleration on Apple Silicon chips
- `--out_dir=work` saves checkpoint to the same directory as the data files
- `--eval_interval=100` creates checkpoints more frequently (every 100 iterations)
- For CUDA systems, you can use larger batch sizes and more iterations


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

