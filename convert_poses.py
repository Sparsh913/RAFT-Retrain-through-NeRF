import json
import numpy as np
from pathlib import Path

def ind(c2w):
    if len(c2w) == 3:
        c2w += [[0, 0, 0, 1]]
    return c2w

train_transforms = json.loads(open('poses2/transforms_train.json').read())
eval_transforms = json.loads(open('poses2/transforms_eval.json').read())
transforms = train_transforms + eval_transforms
transforms = sorted(transforms, key=lambda x: int(Path(x['file_path']).stem.split('_')[-1]))

out = {
        'camera_type': 'perspective',
        'render_height': 1080,
        'render_width': 1920,
        'seconds': len(transforms),
        'camera_path': [
            {'camera_to_world': ind(pose['transform']), 'fov': 50, 'aspect': 1}
            for pose in transforms
            ]
        }

outstr = json.dumps(out, indent=4)
with open('camera_path2.json', mode='w') as f:
    f.write(outstr)