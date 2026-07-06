import os
import numpy as np
import torch
import argparse
import tqdm
import os.path as osp

import sys
sys.path.append('./')

from transformers import VideoMAEModel, VideoMAEImageProcessor

from utils.helpers import read_isharakhobor_video, sliding_window_for_list
from utils.feature_extract_resume import (
    save_features_atomic,
    is_feature_done,
    cleanup_stale_temp_files,
    count_completed,
    build_output_paths,
    handle_keyboard_interrupt,
)


class VideoMAEFeatureReader(object):
    def __init__(
        self,
        model_name='MCG-NJU/videomae-large',
        cache_dir=None,
        device='cuda:0',
        overlap_size=0,
        nth_layer=-1,
    ):
        self.device = device
        self.overlap_size = overlap_size
        self.nth_layer = nth_layer

        self.image_processor = VideoMAEImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = VideoMAEModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def get_feats(self, video):
        inputs = self.image_processor(images=video, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True).hidden_states
        outputs = outputs[self.nth_layer]
        outputs = outputs[:, 0]
        return outputs

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


def get_parser():
    parser = argparse.ArgumentParser(
        description="Extract VideoMAE features for isharakhobor directly from source videos"
    )
    parser.add_argument('--anno_root', required=True, help='Annotation directory')
    parser.add_argument(
        '--video_root', required=True,
        help='Path to CLIPS/ directory (not extracted frames)',
    )
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--model_name', default='MCG-NJU/videomae-large')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--overlap_size', type=int, default=8)
    parser.add_argument('--nth_layer', type=int, default=-1)
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument(
        '--resize', type=int, nargs=2, default=None,
        help='Optional resize W H applied while decoding video frames',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-extract features even when output files already exist',
    )
    return parser


def build_mae_postfix(args, st):
    postfix = f'_overlap-{args.overlap_size}'
    if st is not None:
        postfix = f'_{st}{postfix}'
    return postfix


def get_iterator(args, mode, item_save_dir):
    batch_size = args.batch_size
    data = np.load(os.path.join(args.anno_root, f'{mode}_info.npy'), allow_pickle=True).item()
    num = len(data)

    reader = VideoMAEFeatureReader(
        args.model_name,
        device=args.device,
        overlap_size=args.overlap_size,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir,
    )

    def iterate():
        for i in range(num):
            output_path = osp.join(
                item_save_dir,
                f"{data[i]['fileid']}{build_mae_postfix(args, None)}.npy",
            )
            if not args.force and is_feature_done(output_path):
                continue

            frames = read_isharakhobor_video(args.video_root, data[i], resize=args.resize)
            if len(frames) == 0:
                yield np.array([]), data[i]['fileid'], None, output_path
                continue

            if len(frames) < 16:
                frames.extend([frames[-1]] * (16 - len(frames)))

            frame_chunks = sliding_window_for_list(
                frames, window_size=16, overlap_size=args.overlap_size
            )
            videos = [chunk for chunk in frame_chunks]

            video_feats = []
            for j in range(0, len(videos), batch_size):
                video_batch = videos[j:min(j + batch_size, len(videos))]
                feats = reader.get_feats(video_batch).cpu().numpy()
                video_feats.append(feats)

            yield np.concatenate(video_feats, axis=0), data[i]['fileid'], None, output_path

    return iterate, num, data


def main():
    parser = get_parser()
    args = parser.parse_args()

    ds_name = osp.basename(osp.normpath(args.anno_root))
    if ds_name != 'isharakhobor' and not ds_name.endswith('isharakhobor'):
        print(
            f"Warning: anno_root basename is '{ds_name}'; "
            "expected a directory named 'isharakhobor' for downstream training."
        )

    try:
        for mode in ("dev", "test", "train"):
            item_save_dir = osp.join(args.save_dir, 'mae_feat_isharakhobor', mode)
            os.makedirs(item_save_dir, exist_ok=True)
            cleanup_stale_temp_files(item_save_dir)

            generator, num, data = get_iterator(args, mode, item_save_dir)
            output_paths = build_output_paths(
                item_save_dir,
                data,
                num,
                lambda item: build_mae_postfix(args, None),
            )
            done = count_completed(output_paths)
            remaining = num - done
            if done and not args.force:
                print(f"[{mode}] Resuming: {done}/{num} complete, {remaining} remaining.")

            iterator = generator()
            progress = tqdm.tqdm(iterator, total=remaining if not args.force else num)
            for mae_feat in progress:
                feats, _id, _st, output_path = mae_feat
                save_features_atomic(output_path, feats)
    except KeyboardInterrupt:
        handle_keyboard_interrupt()


if __name__ == "__main__":
    main()
