import os
import os.path as osp
import tqdm
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, CLIPVisionModel
from transformers.utils import logging as hf_logging

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

hf_logging.set_verbosity_error()
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


def configure_gpu(device='cuda:0'):
    if not torch.cuda.is_available():
        print('CUDA not available; falling back to CPU.')
        return 'cpu', False

    device_id = int(device.split(':')[-1]) if ':' in device else 0
    name = torch.cuda.get_device_name(device_id)
    major, minor = torch.cuda.get_device_capability(device_id)
    use_fp16 = True
    print(f'Using {name} on {device} (compute capability {major}.{minor}), fp16={use_fp16}')
    return device, use_fp16


class ViTFeatureReader(object):
    def __init__(
        self,
        model_name='openai/clip-vit-large-patch14',
        cache_dir=None,
        device='cuda:0',
        s2_mode='s2wrapping',
        scales=(1, 2),
        nth_layer=-1,
        use_fp16=False,
    ):
        self.s2_mode = s2_mode
        self.device = device
        self.scales = scales
        self.nth_layer = nth_layer
        self.use_fp16 = use_fp16 and device != 'cpu'

        load_kwargs = {'output_hidden_states': True, 'cache_dir': cache_dir}
        if self.use_fp16:
            load_kwargs['torch_dtype'] = torch.float16

        print(f'Loading CLIP vision encoder from {model_name} ...')
        self.model = CLIPVisionModel.from_pretrained(model_name, **load_kwargs).to(device).eval()
        self.image_processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        print('CLIP vision encoder ready.')

    @torch.no_grad()
    def forward_features(self, inputs):
        if self.use_fp16 and inputs.dtype != torch.float16:
            inputs = inputs.to(dtype=torch.float16)
        outputs = self.model(inputs).hidden_states
        return outputs[self.nth_layer]

    @torch.no_grad()
    def get_feats(self, video):
        inputs = self.image_processor(list(video), return_tensors="pt").pixel_values.to(self.device)
        if self.use_fp16:
            inputs = inputs.to(dtype=torch.float16)
        if self.s2_mode == "s2wrapping":
            outputs = multiscale_forward(
                self.forward_features, inputs, scales=self.scales, num_prefix_token=1
            )
        else:
            outputs = self.forward_features(inputs)
        return outputs[:, 0].float()


class Args:
    def __init__(self):
        self.anno_root = '/kaggle/input/datasets/kowshiksaha006/isharakhobor-frames/preprocess'
        self.video_root = '/kaggle/input/datasets/kowshiksaha006/isharakhobor-frames/frame_root'
        self.device, self.use_fp16 = configure_gpu('cuda:0')
        self.s2_mode = 's2wrapping'
        self.scales = [1, 2]
        self.batch_size = 8
        self.nth_layer = -1
        self.cache_dir = '/kaggle/working/hf_cache'
        self.save_dir = '/kaggle/working/save/isharakhobor'
        self.model_name = 'openai/clip-vit-large-patch14'
        self.force = False


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


def get_iterator(args, mode, item_save_dir, reader):
    batch_size = args.batch_size

    data = np.load(os.path.join(args.anno_root, f'{mode}_info.npy'), allow_pickle=True).item()
    ds_name = 'isharakhobor'
    num = len(data) if ds_name in ('Banglagov', 'BTVSL', 'isharakhobor') else len(data) - 1

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

            if ds_name in ('Phoenix14T', 'CSL-Daily', 'Banglagov', 'BTVSL', 'isharakhobor'):
                image_list = get_img_list(ds_name, args.video_root, fname)
                videos = [Image.open(image).convert('RGB') for image in image_list]

                video_feats = []
                for j in range(0, len(videos), batch_size):
                    video_batch = videos[j:min(j + batch_size, len(videos))]
                    feats = reader.get_feats(video_batch).cpu().numpy()
                    video_feats.append(feats)

                if args.device != 'cpu':
                    torch.cuda.empty_cache()
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
                    if args.device != 'cpu':
                        torch.cuda.empty_cache()
                    yield np.concatenate(video_feats, axis=0), data[i]['fileid'], st, output_path
                else:
                    yield [], data[i]['fileid'], st, output_path

    return iterate, num, data


def main():
    args = Args()
    reader = ViTFeatureReader(
        args.model_name,
        device=args.device,
        s2_mode=args.s2_mode,
        scales=args.scales,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir,
        use_fp16=args.use_fp16,
    )

    mode = ["dev", "test", "train"]
    try:
        for m in mode:
            ds_name = 'isharakhobor'
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

            generator, num, data = get_iterator(args, _m, item_save_dir, reader)
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
