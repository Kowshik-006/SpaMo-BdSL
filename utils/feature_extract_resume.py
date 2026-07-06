import os
import os.path as osp
import sys

import numpy as np


def save_features_atomic(path, feats):
    """Write features atomically so crashes do not leave partial outputs."""
    if path.endswith('.npy'):
        tmp_path = path[:-4] + '.tmp.npy'
    else:
        tmp_path = path + '.tmp.npy'
    np.save(tmp_path, feats)
    os.replace(tmp_path, path)


def is_feature_done(path):
    """Return True when a non-empty feature file already exists."""
    return osp.isfile(path) and osp.getsize(path) > 0


def cleanup_stale_temp_files(directory):
    """Remove leftover temp files from interrupted writes."""
    if not osp.isdir(directory):
        return
    for name in os.listdir(directory):
        if name.endswith('.tmp.npy'):
            try:
                os.remove(osp.join(directory, name))
            except OSError:
                pass


def count_completed(output_paths):
    return sum(is_feature_done(path) for path in output_paths)


def build_output_paths(save_dir, data, num, postfix_fn):
    """Build expected output paths for each dataset index."""
    paths = []
    for i in range(num):
        fileid = data[i]['fileid']
        postfix = postfix_fn(data[i])
        paths.append(osp.join(save_dir, f'{fileid}{postfix}.npy'))
    return paths


def handle_keyboard_interrupt():
    print(
        "\nInterrupted. Progress saved for completed items. "
        "Re-run the same command to resume.",
        file=sys.stderr,
    )
    sys.exit(130)
