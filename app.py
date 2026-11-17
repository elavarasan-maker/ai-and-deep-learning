from pathlib import Path
import io, csv, uuid
import numpy as np
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import tensorflow as tf
from src.data import load_npz, make_features, compute_nbr
from src.config import MODEL_DIR

app=Flask(__name__); app.secret_key='change-this-before-deployment'
MODEL_PATH=MODEL_DIR/'efficientnet_final.keras'; PREP_PATH=MODEL_DIR/'preprocessing.npz'
model=None; mean=None; std=None
if MODEL_PATH.exists() and PREP_PATH.exists():
    model=tf.keras.models.load_model(MODEL_PATH); p=np.load(PREP_PATH); mean=p['mean'].reshape(1,1,5); std=p['std'].reshape(1,1,5)


def predict_file(file):
    raw=file.read(); file.seek(0)
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:
        if not {'image','label'}.issubset(z.files): raise ValueError('Upload an NPZ containing image and label arrays.')
        image=np.asarray(z['image'])
        label=np.asarray(z['label'])
    if image.ndim!=3: raise ValueError('Invalid image array shape.')
    if image.shape[0] in (12,13): image=np.moveaxis(image[:12],0,-1)
    elif image.shape[-1] in (12,13): image=image[...,:12]
    else: raise ValueError('Expected 12 Sentinel-2 bands.')
    if model is None: raise RuntimeError('Model is not trained yet. Run training first.')
    x=make_features(image); prob=float(model.predict(((x-mean)/std)[None,...],verbose=0)[0,0])
    risk='EXTREME' if prob>=.85 else 'HIGH' if prob>=.60 else 'LOW'
    return {'filename':file.filename,'prediction':'Fire' if prob>=.5 else 'No-Fire','confidence':prob,'risk':risk,'nbr_mean':float(np.nanmean(compute_nbr(image)))}

@app.route('/',methods=['GET','POST'])
def index():
    results=[]
    if request.method=='POST':
        uploads=request.files.getlist('files')
        for f in uploads:
            if not f.filename: continue
            try: results.append(predict_file(f))
            except Exception as e: flash(f'{f.filename}: {e}')
        if results:
            # Store only this request's report in memory and provide download link through session-independent token.
            token=uuid.uuid4().hex; REPORTS[token]=results
            return render_template('index.html',results=results,report_url=url_for('download_report',token=token))
    return render_template('index.html',results=results)

REPORTS={}
@app.route('/report/<token>')
def download_report(token):
    rows=REPORTS.get(token)
    if rows is None: return 'Report expired or not found',404
    s=io.StringIO(); writer=csv.DictWriter(s,fieldnames=['filename','prediction','confidence','risk','nbr_mean']); writer.writeheader(); writer.writerows(rows)
    return send_file(io.BytesIO(s.getvalue().encode()),mimetype='text/csv',as_attachment=True,download_name='wildfire_alert_report.csv')

if __name__=='__main__': app.run(debug=True)
