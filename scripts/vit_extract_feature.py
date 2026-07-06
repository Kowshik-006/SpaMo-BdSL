import argparse
import os
import os.path as osp
import glob
import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, CLIPVisionModel

import sys
sys.path.append('./')

from utils.s2wrapper import forward as multiscale_forward
from utils.helpers import read_video, get_img_list
from utils.feature_extract_resume import (
    save_features_atomic,
    is_feature_done,
    cleanup_stale_temp_files,
    count_completed,
    build_output_paths,
    handle_keyboard_interrupt,
)


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)



class ViTFeatureReader(object):
    def __init__(
        self, 
        model_name='openai/clip-vit-large-patch14', 
        cache_dir=None,
        device='cuda:0', 
        s2_mode='s2wrapping',
        scales=[1, 2],
        nth_layer=-1
    ):
        self.s2_mode = s2_mode
        self.device = device
        self.scales = scales
        self.nth_layer = nth_layer
        
        self.model = CLIPVisionModel.from_pretrained(
            model_name, output_hidden_states=True, cache_dir=cache_dir
        ).to(device).eval()
        
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def forward_features(self, inputs):
        outputs = self.model(inputs).hidden_states
        outputs = outputs[self.nth_layer]
        return outputs

    @torch.no_grad()
    def get_feats(self, video):
        inputs = self.image_processor(list(video), return_tensors="pt").to(self.device).pixel_values
        if self.s2_mode == "s2wrapping":
            outputs = multiscale_forward(self.forward_features, inputs, scales=self.scales, num_prefix_token=1)
        else:
            outputs = self.forward_features(inputs)
        return outputs[:, 0]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', help='location of tsv files', required=True)
    parser.add_argument('--video_root', help='location of tsv files', required=True)
    parser.add_argument('--device', help='device to use', default='cuda:0')
    parser.add_argument('--s2_mode', default='')
    parser.add_argument('--scales', nargs='+', type=int, help='List of scales', default=[])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nth_layer', type=int, default=-1)
    parser.add_argument('--cache_dir', help='cache dir for model', default=None)
    
    parser.add_argument('--save_dir', help='where to save the output', required=True)
    parser.add_argument('--model_name', help='ViT model name', default='openai/clip-vit-large-patch14')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-extract features even when output files already exist',
    )

    return parser


def build_vit_postfix(args, st):
    postfix = ""
    if args.s2_mode != "":
        postfix = f"_{args.s2_mode}"
    if len(args.scales) == 3:
        postfix = f'{postfix}_large'
    if st is not None:
        postfix = f'_{st}{postfix}'
    return postfix


def get_start_time(data_item, ds_name):
    if ds_name == 'How2Sign':
        return str(data_item['original_info']['START_REALIGNED'])
    return None


def get_iterator(args, mode, item_save_dir):
    batch_size = args.batch_size
    
    data = np.load(os.path.join(args.anno_root, f'{mode}_info.npy'), allow_pickle=True).item()
    ds_name = osp.split(args.anno_root)[-1]
    num = len(data) if ds_name in ('Banglagov', 'BTVSL') else len(data) - 1
    reader = ViTFeatureReader(
        args.model_name, 
        device=args.device, 
        s2_mode=args.s2_mode, 
        scales=args.scales,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir
    )
    
    def iterate():
        for i in range(num):
            st = get_start_time(data[i], ds_name)
            output_path = osp.join(
                item_save_dir,
                f"{data[i]['fileid']}{build_vit_postfix(args, st)}.npy",
            )
            if not args.force and is_feature_done(output_path):
                continue

            fname = data[i]['folder']
            
            if ds_name in ('Phoenix14T', 'CSL-Daily', 'Banglagov', 'BTVSL'):
                image_list = get_img_list(ds_name, args.video_root, fname)
                videos = [Image.open(image).convert('RGB') for image in image_list]
                
                video_feats = []
                for j in range(0, len(videos), batch_size):
                    video_batch = videos[j:min(j + batch_size, len(videos))]
                    feats = reader.get_feats(video_batch).cpu().numpy()
                    video_feats.append(feats)
                
                yield np.concatenate(video_feats, axis=0), data[i]['fileid'], st, output_path
            
            else:
                if ds_name == 'How2Sign':
                    start_time, end_time = data[i]['original_info']['START_REALIGNED'], data[i]['original_info']['END_REALIGNED']
                    videos = read_video(fname, start_time=start_time, end_time=end_time)
            
                if len(videos) > 0:
                    video_feats = []
                    for j in range(0, len(videos), batch_size):
                        video_batch = videos[j:min(j + batch_size, len(videos))]
                        feats = reader.get_feats(video_batch).cpu().numpy()
                        video_feats.append(feats)
                    yield np.concatenate(video_feats, axis=0), data[i]['fileid'], st, output_path
                else:
                    yield [], data[i]['fileid'], st, output_path
    
    return iterate, num, data


def main():
    parser = get_parser()
    args = parser.parse_args()
    mode = ["dev", "test", "train"]

    try:
        for m in mode:
            ds_name = osp.split(args.anno_root)[-1]
            _model_name = os.path.split(args.model_name)[-1]
            fname = f'{_model_name}_feat_{ds_name}'
            item_save_dir = osp.join(args.save_dir, fname, m)
            os.makedirs(item_save_dir, exist_ok=True)
            cleanup_stale_temp_files(item_save_dir)
        
            if ds_name == 'How2Sign':
                if m == 'dev': _m = 'val'
                else: _m = m
            elif ds_name == 'NIASL2021':
                if m == 'dev': _m = 'validation' 
            else:
                _m = m

            generator, num, data = get_iterator(args, _m, item_save_dir)
            output_paths = build_output_paths(
                item_save_dir,
                data,
                num,
                lambda item: build_vit_postfix(args, get_start_time(item, ds_name)),
            )
            done = count_completed(output_paths)
            remaining = num - done
            if done and not args.force:
                print(f"[{m}] Resuming: {done}/{num} complete, {remaining} remaining.")

            iterator = generator()
            progress = tqdm.tqdm(iterator, total=remaining if not args.force else num)
            for vit_feat in progress:
                feats, _id, _st, output_path = vit_feat
                save_features_atomic(output_path, feats)
    except KeyboardInterrupt:
        handle_keyboard_interrupt()


if __name__ == "__main__":
    main()