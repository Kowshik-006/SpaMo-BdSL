# Banglagov dataset preprocessing

Use these steps to run the SpaMo pipeline on the **Banglagov** dataset.

## Dataset layout (your side)

- **1922 directories**, each named by **Sentence ID** (as in the Excel).
- Each directory has **5 videos** (same sentence signed by 5 different signers).  
  Video files can be named in any way (e.g. `video0.mp4`, `1.mp4`, …); they will be **sorted by filename** and assigned to signer index 0–4 in that order.
- One **Excel file** with (at least) these columns:
  - **Sentence ID** – same as the directory name
  - **Natural sentence** – natural language translation of the sign video
  - **Sign Sentence** – gloss sequence

Column names can have different casing/spacing (e.g. "Sentence ID", "Natural sentence", "Sign Sentence"); the script will try to map them.

---

## Step 1: Build annotation files from Excel

This creates `train_info.npy`, `dev_info.npy`, `test_info.npy` and the `_info_ml.npy` variants under `preprocess/Banglagov/`.

```bash
python preprocess/banglagov/build_anno_from_excel.py \
  --excel /path/to/your/annotations.xlsx \
  --video_root /path/to/root/where/1922/dirs/lie \
  --out_dir ./preprocess/Banglagov \
  --train_ratio 0.7 \
  --dev_ratio 0.15
```

- Splits are at **sentence level**: all 5 signers of a sentence stay in the same split.
- You get **1922 × 5 = 9610** samples total (unless some rows are dropped).
- Install **pandas** (or **openpyxl**) if needed: `pip install pandas openpyxl`.

---

## Step 2: Extract frames from videos

The feature scripts expect frames under a fixed layout. This step writes one folder per sample (e.g. `.../banglagov/train/<sentence_id>_signer_0/` with `00001.png`, `00002.png`, …).

```bash
python preprocess/banglagov/extract_frames.py \
  --anno_dir ./preprocess/Banglagov \
  --video_root /home/mdsr/CSLT-Kowshik-Adiba/banglagov/Sign_Videos  \
  --frames_root /home/mdsr/CSLT-Kowshik-Adiba/banglagov/frame_root \
  --video_ext .mp4
```

- **`--frames_root`**: base directory for all extracted frames.  
  Frames will be under:  
  `frames_root/features/fullFrame-256x256px/banglagov/{train,dev,test}/<fileid>/`.
- If your videos are `.avi`, use `--video_ext .avi`.
- Requires **opencv-python**: `pip install opencv-python`.

---

## Step 3: Extract ViT (spatial) features

Use the **frames root** (where you extracted frames) as `--video_root`:

```bash
python scripts/vit_extract_feature.py \
  --anno_root ./preprocess/Banglagov \
  --video_root /home/mdsr/CSLT-Kowshik-Adiba/banglagov/frame_root \
  --save_dir /home/mdsr/CSLT-Kowshik-Adiba/SpaMo-BdSL/save/banglagov_vit \
  --s2_mode s2wrapping \
  --scales 1 2 \
  --model_name openai/clip-vit-large-patch14
```

- Saves under: `save_dir/clip-vit-large-patch14_feat_Banglagov/{train,dev,test}/<fileid>_s2wrapping.npy`.

---

## Step 4: Extract MAE (spatiotemporal) features

```bash
python scripts/mae_extract_feature.py \
  --anno_root ./preprocess/Banglagov_subset \
  --video_root /home/mdsr/CSLT-Kowshik-Adiba/banglagov/frame_root \
  --save_dir /home/mdsr/CSLT-Kowshik-Adiba/SpaMo-BdSL/save/banglagov_mae_sub \
  --overlap_size 8
```

- Saves under: `save_dir/mae_feat_Banglagov/{train,dev,test}/<fileid>_overlap-8.npy`.

---

## Step 5: Train / evaluate

Use the same dataset class as Phoenix14T, with paths and language set for Banglagov. Example config snippet:

```yaml
data:
  params:
    batch_size: 4
    train:
      target: dataset.p14t.Phoenix14T
      params:
        anno_root: ./preprocess/Banglagov_subset
        vid_root: /home/mdsr/CSLT-Kowshik-Adiba/banglagov/frame_root
        feat_root: /home/mdsr/CSLT-Kowshik-Adiba/SpaMo-BdSL/save/banglagov_vit_sub/clip-vit-large-patch14_feat_Banglagov_subset
        mae_feat_root: /home/mdsr/CSLT-Kowshik-Adiba/SpaMo-BdSL/save/banglagov_mae_sub/mae_feat_Banglagov_subset
        mode: train
        spatial: true
        spatiotemporal: true
        spatial_postfix: _s2wrapping
        spatiotemporal_postfix: _overlap-8
        lang: Bengali
    validation:
      target: dataset.p14t.Phoenix14T
      params:
        anno_root: ./preprocess/Banglagov_subset
        vid_root: /home/mdsr/CSLT-Kowshik-Adiba/banglagov/frame_root
        feat_root: /home/mdsr/CSLT-Kowshik-Adiba/SpaMo-BdSL/save/banglagov_vit_sub/clip-vit-large-patch14_feat_Banglagov_subset
        mae_feat_root: /home/mdsr/CSLT-Kowshik-Adiba/SpaMo-BdSL/save/banglagov_mae_sub/mae_feat_Banglagov_subset
        mode: dev
        spatial: true
        spatiotemporal: true
        spatial_postfix: _s2wrapping
        spatiotemporal_postfix: _overlap-8
        lang: Bengali
    test:
      target: dataset.p14t.Phoenix14T
      params:
        ...  # same as validation with mode: test
```

- **`vid_root`** here is the same **frames root** as in steps 3–4 (the directory that contains `features/fullFrame-256x256px/banglagov/`).
- **`lang: Bengali`** is used in the translation prompt; change if your target language name differs.

Then run training as usual, e.g.:

```bash
python main.py -c configs/finetune_banglagov.yaml -e bleu
```

---

## Summary

| Step | What it does |
|------|----------------|
| 1 | Excel → `preprocess/Banglagov/{train,dev,test}_info.npy` and `_info_ml.npy` |
| 2 | Videos → frames under `frames_root/features/fullFrame-256x256px/banglagov/` |
| 3 | Frames → ViT features (spatial) |
| 4 | Frames → MAE features (spatiotemporal) |
| 5 | Train with config pointing at Banglagov anno_root, feat roots, and `lang: Bengali` |

If your Excel has different column names, adjust the mapping in `build_anno_from_excel.py` (see `col_map` and the checks for `sentence_id`, `natural_sentence`, `sign_sentence`).
