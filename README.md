### Download FLORES-200 dataset

```bash
pip install datasets
python src/download_data.py
```

### Run the char-level "prepare" step (copied from shakespeare-char)

> Hardcoded `eng_Latn.txt` for now. Need to make this dynamic.
```bash
python data/flores/prepare.py 
```
This will create `train.bin`, `val.bin`, and `meta.pkl` in the `data/flores` directory.

### Train the model

```bash
python train.py \
  config/train_flores_eng_char.py \
  --dataset=flores \
  --out_dir=out-flores-eng \
  --device=cpu        \  # or `cuda` / `mps` if you have a GPU
  --compile=False     \ 
  --batch_size=8      \
  --block_size=128    \
  --max_iters=2000   \
  --lr_decay_iters=2000
```
> Also hardcoded `eng_Latn_char` in the `train_flores_eng_char.py` config for now. Need to make this dynamic.
> --dataset=flores - path to where the train.bin, val.bin, and meta.pkl are located
> --out_dir=out-flores-eng - path to where the model checkpoints and logs will be saved

### Sample from the model

```bash
python sample.py \
  --out_dir=out-flores-eng \
  --device=cpu    
```

### Training the model using `myprogram.py` script
- Make sure the data is downloaded and you have run the prepare step

```bash
python src/myprogram.py train --work_dir=out-flores-eng
```
> Values are hardcoded for now. Need to make this dynamic.

### Predicting using `myprogram.py` script

```bash
python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output pred.txt```
```
> Values are hardcoded for now. Need to make this dynamic.


