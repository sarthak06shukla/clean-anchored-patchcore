"""Run one fitted category-specific bank on an image; save score and heatmap."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
from core import Extractor, transform_image, score_features

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True,help='Completed experiment directory')
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--method',default='adaptive',choices=['clean','pooled','uniform','adaptive','no_guard','no_anchor'])
    parser.add_argument('--output',type=Path,default=Path('prediction'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    config=json.loads((args.run/'config_locked.json').read_text())
    info=json.loads((args.run/'fit_summary.json').read_text())
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=Extractor(device);bank=torch.from_numpy(np.load(args.run/f'bank_{args.method}.npy')).to(device)
    size=config['image_size'];im=transform_image(args.image,'clean',size)
    d,score=score_features(model(im),bank)
    heat=F.interpolate(d.reshape(1,1,size//8,size//8),size=(size,size),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
    heat=gaussian_filter(heat,sigma=config['map_smoothing_sigma'])
    threshold=info['thresholds'][args.method]['value']
    result={'image':args.image.name,'category':config['category'],'method':args.method,'score':score,
            'threshold':threshold,'anomalous':bool(threshold is not None and score>threshold),
            'note':'Threshold calibrated on clean normal images; overlay scale is visualization-only, not a pixel decision threshold.'}
    (args.output/'prediction.json').write_text(json.dumps(result,indent=2));np.save(args.output/'anomaly_map.npy',heat)
    from matplotlib import colormaps
    scale=(heat-heat.min())/max(float(heat.max()-heat.min()),1e-12)
    color=colormaps['inferno'](scale)[:,:,:3]
    overlay=.55*np.asarray(im)/255+.45*color
    Image.fromarray((overlay*255).astype(np.uint8)).save(args.output/'overlay.png')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
