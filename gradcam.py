import argparse
import cv2, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from .config import MODEL_DIR, ARTIFACT_DIR
from .data import load_npz, make_features, compute_nbr


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('npz'); args=ap.parse_args()
    model=tf.keras.models.load_model(MODEL_DIR/'efficientnet_final.keras')
    prep=np.load(MODEL_DIR/'preprocessing.npz'); mean,std=prep['mean'].reshape(1,1,5),prep['std'].reshape(1,1,5)
    image,label=load_npz(args.npz); x=make_features(image); xn=(x-mean)/std
    base=next(l for l in model.layers if isinstance(l,tf.keras.Model) and 'efficientnet' in l.name.lower())
    base_tail=tf.keras.Model(base.input,[base.get_layer('top_conv').output,base.output])
    inp=model.input
    with tf.GradientTape() as tape:
        adapted=model.get_layer('spectral_adapter')(inp)
        conv_out,base_out=base_tail(adapted)
        z=model.get_layer('global_average_pooling2d')(base_out)
        z=model.get_layer('dense')(z)
        z=model.get_layer('dropout')(z,training=False)
        score=model.layers[-1](z)[:,0]
    grads=tape.gradient(score,conv_out); weights=tf.reduce_mean(grads,axis=(1,2))
    cam=tf.reduce_sum(conv_out*weights[:,None,None,:],axis=-1)[0]; cam=tf.maximum(cam,0); cam/=tf.reduce_max(cam)+1e-8
    heat=cv2.resize(cam.numpy(),(224,224)); rgb=x[...,:3]; rgb=(rgb-rgb.min())/(rgb.max()-rgb.min()+1e-6)
    overlay=cv2.applyColorMap(np.uint8(255*heat),cv2.COLORMAP_JET); overlay=cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)/255.; blend=.55*rgb+.45*overlay
    nbr=cv2.resize(compute_nbr(image),(224,224))
    fig,axs=plt.subplots(1,3,figsize=(12,4)); axs[0].imshow(rgb); axs[0].set_title('RGB'); axs[1].imshow(nbr,cmap='RdYlBu',vmin=-1,vmax=1); axs[1].set_title('NBR'); axs[2].imshow(blend); axs[2].set_title('Grad-CAM')
    for a in axs:a.axis('off')
    fig.tight_layout(); ARTIFACT_DIR.mkdir(exist_ok=True); out=ARTIFACT_DIR/'gradcam_example.png'; fig.savefig(out,dpi=160); print(out)

if __name__=='__main__': main()
