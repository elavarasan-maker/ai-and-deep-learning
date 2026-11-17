import argparse, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix,
                             roc_auc_score, roc_curve, classification_report)
from .config import MODEL_DIR, ARTIFACT_DIR
from .data import list_files, PatchSequence, load_npz, patch_label


def evaluate_one(model_path, stem, files, y, seq):
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(seq, verbose=1).ravel()
    pred = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    result = {
        'model': stem, 'accuracy': float(accuracy_score(y, pred)),
        'class_0_no_fire': {'precision': float(precision[0]), 'recall': float(recall[0]), 'f1': float(f1[0])},
        'class_1_fire': {'precision': float(precision[1]), 'recall': float(recall[1]), 'f1': float(f1[1])},
        'macro_f1': float(np.mean(f1)),
        'roc_auc': float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else None,
        'threshold': 0.5, 'test_patches': len(y), 'test_fire_count': int(y.sum())
    }
    (ARTIFACT_DIR / f'{stem}_metrics.json').write_text(json.dumps(result, indent=2))
    (ARTIFACT_DIR / f'{stem}_classification_report.txt').write_text(
        classification_report(y, pred, labels=[0, 1], target_names=['No-Fire', 'Fire'], zero_division=0))
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No-Fire', 'Fire'], yticklabels=['No-Fire', 'Fire'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(f'{stem} confusion matrix')
    fig.tight_layout(); fig.savefig(ARTIFACT_DIR / f'{stem}_confusion_matrix.png', dpi=160); plt.close(fig)
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, probs)
        fig, ax = plt.subplots(figsize=(5, 4)); ax.plot(fpr, tpr, label=f"AUC={result['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], 'k--'); ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
        ax.legend(); ax.set_title(f'{stem} ROC curve'); fig.tight_layout()
        fig.savefig(ARTIFACT_DIR / f'{stem}_roc_curve.png', dpi=160); plt.close(fig)
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data_dir', default='data/Sen2Fire'); args = ap.parse_args()
    prep = np.load(MODEL_DIR / 'preprocessing.npz'); mean, std = prep['mean'], prep['std']
    files = list_files(args.data_dir, 'test'); seq = PatchSequence(files, mean, std, batch_size=8)
    y = np.asarray([patch_label(load_npz(p)[1]) for p in files])
    results = []
    for filename, stem in [('baseline_final.keras', 'baseline'), ('efficientnet_final.keras', 'efficientnet')]:
        path = MODEL_DIR / filename
        if path.exists(): results.append(evaluate_one(path, stem, files, y, seq))
    if not results: raise FileNotFoundError('No trained models found in models/. Run training first.')
    (ARTIFACT_DIR / 'metrics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == '__main__': main()
