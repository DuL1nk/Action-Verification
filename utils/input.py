import torch
import pdb

def frames_preprocess(frames, model_dim, model_name):
    # Raw frames size: [bs, c, h, w, len_clip, num_clip]
    # Reshape for 2D backbone: [bs * num_clip, c, h, w], len_clip = 1
    # Reshape for 3D backbone: [bs * num_clip, c, h, w, len_clip]

    bs, c, h, w, len_clip, num_clip = frames.size()
    # pdb.set_trace()
    if model_dim == '2D':
        assert len_clip == 1, 'Expect len_clip = 1 for 2D backbone, but get %d instead' % len_clip
        frames = frames.squeeze(-2)
        frames = frames.permute(0, 4, 1, 2, 3)
        frames = frames.reshape(-1, c, h, w)
    elif model_dim == '2.5D':
        assert len_clip > 1, 'Expect len_clip > 1 for 2.5D backbone, but get %d instead' % len_clip
        frames = frames.permute(0, 5, 1, 2, 3, 4)
        frames = frames.reshape(-1, c, h, w, len_clip)
    elif model_dim == '3D':
        assert len_clip == 1, 'Expect len_clip = 1 for 3D backbone, but get %d instead' % len_clip
        frames = frames.squeeze()
        if model_name == 'c3d' or model_name == 'i3d':
            frames = frames.permute(0, 1, 4, 2, 3)


    return frames