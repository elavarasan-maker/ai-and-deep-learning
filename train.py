import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from .config import ARTIFACT_DIR, MODEL_DIR, BATCH_SIZE, SEED
from .data import list_files, PatchSequence, get_train_stats, patch_label, load_npz
from .models import build_baseline, build_efficientnet


def class_weights(files):
    y=[]
    for p in files:
        _, lab=load_npz(p); y.append(patch_label(lab))
    classes=np.array([0,1]); weights=compute_class_weight(class_weight='balanced',classes=classes,y=np.asarray(y))
    return {int(c):float(w) for c,w in zip(classes,weights)}, y


def callbacks(name):
    return [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=6,restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,min_lr=1e-7),
            tf.keras.callbacks.ModelCheckpoint(str(MODEL_DIR/f'{name}_best.keras'),monitor='val_loss',save_best_only=True)]


def plot_history(histories, name):
    import matplotlib.pyplot as plt
    for metric in ('accuracy','loss'):
        fig,ax=plt.subplots(figsize=(7,4))
        for label,h in histories.items():
            ax.plot(h.history.get(metric,[]),label=f'{label} train')
            ax.plot(h.history.get('val_'+metric,[]),label=f'{label} val',linestyle='--')
        ax.set_title(f'{name}: {metric}'); ax.set_xlabel('Epoch'); ax.legend(); fig.tight_layout(); fig.savefig(ARTIFACT_DIR/f'{name}_{metric}.png',dpi=160); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data_dir',default='data/Sen2Fire'); ap.add_argument('--epochs',type=int,default=25); ap.add_argument('--batch_size',type=int,default=BATCH_SIZE); ap.add_argument('--skip_baseline',action='store_true'); args=ap.parse_args()
    ARTIFACT_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True); np.random.seed(SEED); tf.random.set_seed(SEED)
    train_files=list_files(args.data_dir,'train'); val_files=list_files(args.data_dir,'val')
    mean,std=get_train_stats(args.data_dir)
    np.savez(MODEL_DIR/'preprocessing.npz',mean=mean,std=std)
    cw,y=class_weights(train_files); (ARTIFACT_DIR/'training_metadata.json').write_text(json.dumps({'class_weights':cw,'train_count':len(train_files),'val_count':len(val_files),'mean':mean.tolist(),'std':std.tolist()},indent=2))
    tr=PatchSequence(train_files,mean,std,args.batch_size,training=True,shuffle=True); va=PatchSequence(val_files,mean,std,args.batch_size)
    histories={}
    if not args.skip_baseline:
        model=build_baseline(); model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='binary_crossentropy',metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
        h=model.fit(tr,validation_data=va,epochs=args.epochs,class_weight=cw,callbacks=callbacks('baseline')); model.save(MODEL_DIR/'baseline_final.keras'); histories['baseline']=h; plot_history(histories,'baseline')
    model,base=build_efficientnet(); model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='binary_crossentropy',metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
    h1=model.fit(tr,validation_data=va,epochs=min(10,args.epochs),class_weight=cw,callbacks=callbacks('efficientnet_frozen'))
    base.trainable=True
    for layer in base.layers[:-40]: layer.trainable=False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss='binary_crossentropy',metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
    h2=model.fit(tr,validation_data=va,epochs=args.epochs,class_weight=cw,callbacks=callbacks('efficientnet_finetuned'))
    model.save(MODEL_DIR/'efficientnet_final.keras'); histories={'frozen':h1,'fine_tuned':h2}; plot_history(histories,'efficientnet')
    print('Training finished. Run: python -m src.evaluate --data_dir',args.data_dir)

if __name__=='__main__': main()
