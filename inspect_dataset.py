import argparse, json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .data import list_files, load_npz, label_fraction, patch_label, make_features, compute_nbr
from .config import ARTIFACT_DIR, SPLITS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data_dir', default='data/Sen2Fire'); args = ap.parse_args()
    out = ARTIFACT_DIR; out.mkdir(exist_ok=True)
    summary = {}
    for split in ('train','val','test'):
        files = list_files(args.data_dir, split); counts = Counter(); fractions=[]
        for p in files:
            im, lab = load_npz(p); y=patch_label(lab); counts[y]+=1; fractions.append(label_fraction(lab))
        summary[split] = {'patches':len(files),'no_fire':counts[0],'fire':counts[1], 'mean_fire_pixel_fraction':float(np.mean(fractions))}
    (out/'dataset_summary.json').write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(7,4)); splits=list(summary)
    ax.bar([s+' Fire' for s in splits],[summary[s]['fire'] for s in splits],label='Fire')
    ax.bar([s+' No-Fire' for s in splits],[summary[s]['no_fire'] for s in splits],label='No-Fire')
    ax.set_ylabel('Patch count'); ax.set_title('Sen2Fire patch-level class distribution'); ax.legend(); fig.tight_layout(); fig.savefig(out/'class_distribution.png',dpi=160); plt.close(fig)
    files=list_files(args.data_dir,'train')[:8]
    fig, axes=plt.subplots(2,4,figsize=(14,7))
    for ax,p in zip(axes.flat,files):
        im,lab=load_npz(p); rgb=make_features(im)[...,:3]; rgb=(rgb-rgb.min())/(rgb.max()-rgb.min()+1e-6)
        ax.imshow(rgb); ax.set_title(f"{p.name}\nlabel={patch_label(lab)} ({label_fraction(lab):.4%})"); ax.axis('off')
    fig.tight_layout(); fig.savefig(out/'sample_patches.png',dpi=160); plt.close(fig)
    # NBR sample maps are diagnostic plots; they are not treated as definitive active-fire labels.
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, p in zip(axes.flat, files):
        im, lab = load_npz(p); nbr = compute_nbr(im)
        view = ax.imshow(nbr, cmap='RdYlBu', vmin=-1, vmax=1)
        ax.set_title(f'{p.name} | fire={patch_label(lab)}'); ax.axis('off')
    fig.colorbar(view, ax=axes.ravel().tolist(), shrink=.7, label='NBR')
    fig.savefig(out/'nbr_visualization.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
