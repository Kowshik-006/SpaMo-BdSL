import argparse
import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from omegaconf import OmegaConf
from transformers import AutoImageProcessor, CLIPVisionModel
from transformers import VideoMAEModel, VideoMAEImageProcessor

from utils.s2wrapper import forward as multiscale_forward
from utils.helpers import sliding_window_for_list, instantiate_from_config


def extract_frames(video_path, resize=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize:
            frame = cv2.resize(frame, resize)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return frames


def extract_vit_features(frames, device='cuda:0', batch_size=8, cache_dir=None):
    model_name = 'openai/clip-vit-large-patch14'

    model = CLIPVisionModel.from_pretrained(
        model_name, output_hidden_states=True, cache_dir=cache_dir
    ).to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    @torch.no_grad()
    def forward_features(inputs):
        return model(inputs).hidden_states[-1]

    all_feats = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:min(i + batch_size, len(frames))]
        inputs = image_processor(batch, return_tensors="pt").to(device).pixel_values
        with torch.no_grad():
            outputs = multiscale_forward(
                forward_features, inputs, scales=[1, 2], num_prefix_token=1
            )
            all_feats.append(outputs[:, 0].cpu())

    del model, image_processor
    torch.cuda.empty_cache()

    return torch.cat(all_feats, dim=0)


def extract_mae_features(frames, device='cuda:0', batch_size=4,
                          overlap_size=8, cache_dir=None):
    model_name = 'MCG-NJU/videomae-large'

    image_processor = VideoMAEImageProcessor.from_pretrained(
        model_name, cache_dir=cache_dir
    )
    model = VideoMAEModel.from_pretrained(
        model_name, cache_dir=cache_dir
    ).to(device).eval()

    frame_list = list(frames)
    if len(frame_list) < 16:
        frame_list += [frame_list[-1]] * (16 - len(frame_list))

    chunks = sliding_window_for_list(frame_list, window_size=16, overlap_size=overlap_size)

    all_feats = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:min(i + batch_size, len(chunks))]
        with torch.no_grad():
            inputs = image_processor(images=batch, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True).hidden_states[-1]
            all_feats.append(outputs[:, 0].cpu())

    del model, image_processor
    torch.cuda.empty_cache()

    return torch.cat(all_feats, dim=0)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Translate a sign language video to Bangla text"
    )
    parser.add_argument(
        '--video_path', required=True,
        help='Path to the input sign language video'
    )
    parser.add_argument(
        '--ckpt_path', required=True,
        help='Path to the trained model checkpoint (.ckpt)'
    )
    parser.add_argument(
        '--config', default='configs/finetune_banglagov.yaml',
        help='Path to the training config YAML'
    )
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cache_dir', default='./hf_cache')
    parser.add_argument(
        '--vit_batch_size', type=int, default=8,
        help='Batch size for ViT feature extraction'
    )
    parser.add_argument(
        '--mae_batch_size', type=int, default=4,
        help='Batch size for MAE feature extraction'
    )
    return parser


def main():
    args = get_parser().parse_args()
    device = args.device

    # --- Step 1: Extract frames ---
    print(f"[1/4] Extracting frames from: {args.video_path}")
    frames = extract_frames(args.video_path, resize=(256, 256))
    print(f"      {len(frames)} frames extracted")

    # --- Step 2: Extract ViT (spatial) features ---
    print("[2/4] Extracting ViT spatial features (S2 wrapping)...")
    vit_features = extract_vit_features(
        frames, device=device,
        batch_size=args.vit_batch_size, cache_dir=args.cache_dir
    )
    print(f"      ViT features: {vit_features.shape}")

    # --- Step 3: Extract MAE (motion) features ---
    print("[3/4] Extracting MAE motion features (overlap=8)...")
    mae_features = extract_mae_features(
        frames, device=device,
        batch_size=args.mae_batch_size, overlap_size=8, cache_dir=args.cache_dir
    )
    print(f"      MAE features: {mae_features.shape}")

    # --- Step 4: Load model and translate ---
    print("[4/4] Loading model and generating translation...")
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)

    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device).eval()

    sample = {
        'pixel_value': vit_features,
        'glor_value': mae_features,
        'num_frames': len(vit_features),
        'id': 'inference',
        'text': '',
        'gloss': '',
        'lang': 'Bangla',
    }

    with torch.no_grad():
        inputs = model.get_inputs([sample])

        visual_outputs, visual_masks = model.prepare_visual_inputs(inputs)
        visual_outputs = model.fusion_proj(visual_outputs)

        input_embeds, input_masks, _, _ = model.prepare_inputs(
            visual_outputs, visual_masks, inputs, 'test', 0
        )

        generated = model.t5_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=input_masks,
            num_beams=5,
            max_length=model.max_txt_len,
        )

        translation = model.t5_tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )[0]

    print(f"\n{'=' * 60}")
    print(f"  Translation:  {translation}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
