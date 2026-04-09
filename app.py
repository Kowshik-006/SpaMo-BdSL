import os
import sys
import cv2
import torch
import tempfile
import gradio as gr
from PIL import Image
import argparse

sys.path.append(os.path.dirname(__file__))

from omegaconf import OmegaConf
from transformers import AutoImageProcessor, CLIPVisionModel
from transformers import VideoMAEModel, VideoMAEImageProcessor

from utils.s2wrapper import forward as multiscale_forward
from utils.helpers import sliding_window_for_list, instantiate_from_config

# =====================================================================
# MODIFY THESE THREE VALUES FOR YOUR ML PC
# =====================================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d","--dataset", 
    type=str, 
    choices=["banglagov", "btvsl"], 
    default="banglagov",
    help="Select SLT dataset: banglagov or btvsl"
)
args, _ = parser.parse_known_args()

if args.dataset == "banglagov":
    CKPT_PATH = "logs/2026-03-24T12-21-42_finetune_banglagov/checkpoints/last.ckpt"
    CONFIG_PATH = "configs/finetune_banglagov.yaml"
else:
    CKPT_PATH = "logs/2026-04-08T17-46-17_finetune_btvsl/checkpoints/last.ckpt"
    CONFIG_PATH = "configs/finetune_btvsl.yaml"
DEVICE = "cuda:0"
CACHE_DIR = "./hf_cache"       # set to None to use ~/.cache/huggingface
VIT_BATCH_SIZE = 8
MAE_BATCH_SIZE = 4
# =====================================================================


def load_models():
    """Load all models once at startup and keep them in memory."""
    print("Loading ViT model...")
    vit_model_name = 'openai/clip-vit-large-patch14'
    vit_model = CLIPVisionModel.from_pretrained(
        vit_model_name, output_hidden_states=True, cache_dir=CACHE_DIR
    ).to(DEVICE).eval()
    vit_processor = AutoImageProcessor.from_pretrained(
        vit_model_name, cache_dir=CACHE_DIR
    )

    print("Loading MAE model...")
    mae_model_name = 'MCG-NJU/videomae-large'
    mae_processor = VideoMAEImageProcessor.from_pretrained(
        mae_model_name, cache_dir=CACHE_DIR
    )
    mae_model = VideoMAEModel.from_pretrained(
        mae_model_name, cache_dir=CACHE_DIR
    ).to(DEVICE).eval()

    print("Loading BanglaT5 SLT model...")
    config = OmegaConf.load(CONFIG_PATH)
    slt_model = instantiate_from_config(config.model)
    checkpoint = torch.load(CKPT_PATH, map_location='cpu')
    slt_model.load_state_dict(checkpoint['state_dict'])
    slt_model = slt_model.to(DEVICE).eval()

    print("All models loaded successfully.\n")
    return vit_model, vit_processor, mae_model, mae_processor, slt_model


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (256, 256))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames could be extracted from the video.")

    return frames


@torch.no_grad()
def get_vit_features(frames, vit_model, vit_processor):
    def forward_features(inputs):
        return vit_model(inputs).hidden_states[-1]

    all_feats = []
    for i in range(0, len(frames), VIT_BATCH_SIZE):
        batch = frames[i:min(i + VIT_BATCH_SIZE, len(frames))]
        inputs = vit_processor(batch, return_tensors="pt").to(DEVICE).pixel_values
        outputs = multiscale_forward(
            forward_features, inputs, scales=[1, 2], num_prefix_token=1
        )
        all_feats.append(outputs[:, 0].cpu())

    return torch.cat(all_feats, dim=0)


@torch.no_grad()
def get_mae_features(frames, mae_model, mae_processor):
    frame_list = list(frames)
    if len(frame_list) < 16:
        frame_list += [frame_list[-1]] * (16 - len(frame_list))

    chunks = sliding_window_for_list(frame_list, window_size=16, overlap_size=8)

    all_feats = []
    for i in range(0, len(chunks), MAE_BATCH_SIZE):
        batch = chunks[i:min(i + MAE_BATCH_SIZE, len(chunks))]
        inputs = mae_processor(images=batch, return_tensors="pt").to(DEVICE)
        outputs = mae_model(**inputs, output_hidden_states=True).hidden_states[-1]
        all_feats.append(outputs[:, 0].cpu())

    return torch.cat(all_feats, dim=0)


@torch.no_grad()
def translate(slt_model, vit_features, mae_features):
    sample = {
        'pixel_value': vit_features.to(DEVICE),
        'glor_value': mae_features.to(DEVICE),
        'num_frames': len(vit_features),
        'id': 'inference',
        'text': '',
        'gloss': '',
        'lang': 'Bangla',
    }

    inputs = slt_model.get_inputs([sample])

    visual_outputs, visual_masks = slt_model.prepare_visual_inputs(inputs)
    visual_outputs = slt_model.fusion_proj(visual_outputs)

    input_embeds, input_masks, _, _ = slt_model.prepare_inputs(
        visual_outputs, visual_masks, inputs, 'test', 0
    )

    generated = slt_model.t5_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=input_masks,
        num_beams=5,
        max_length=slt_model.max_txt_len,
    )

    translation = slt_model.t5_tokenizer.batch_decode(
        generated, skip_special_tokens=True
    )[0]

    return translation


vit_model, vit_processor, mae_model, mae_processor, slt_model = load_models()


def predict(video_file):
    """Gradio callback: receives uploaded video, returns Bangla translation."""
    if video_file is None:
        return "কোনো ভিডিও আপলোড করা হয়নি। (No video uploaded.)"

    try:
        frames = extract_frames(video_file)
        status = f"Extracted {len(frames)} frames"

        vit_features = get_vit_features(frames, vit_model, vit_processor)
        status += f" → ViT features {vit_features.shape}"

        mae_features = get_mae_features(frames, mae_model, mae_processor)
        status += f" → MAE features {mae_features.shape}"

        result = translate(slt_model, vit_features, mae_features)
        print(f"{status} → Translation: {result}")
        return result

    except Exception as e:
        return f"Error: {str(e)}"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Video(label="ইশারা ভাষার ভিডিও আপলোড করুন (Upload Sign Language Video)"),
    outputs=gr.Textbox(label="বাংলা অনুবাদ (Bangla Translation)", lines=3),
    title="SpaMo-BdSL: বাংলা ইশারা ভাষা অনুবাদ",
    description="বাংলা ইশারা ভাষার ভিডিও আপলোড করুন এবং বাংলা অনুবাদ পান।\n\n"
                "Upload a Bangla Sign Language video to get the Bangla text translation.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # accessible from other machines on the network
        server_port=7860,
        share=True,              # generates a public URL via Gradio's tunnel
    )
