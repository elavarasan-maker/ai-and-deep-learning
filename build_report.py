from pathlib import Path
import json
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; OUT=ROOT/'reports'/'Wildfire_Detection_Project_Report.docx'

def add_fig(doc,name,caption):
    p=ART/name
    if p.exists():
        doc.add_picture(str(p),width=Inches(6.0)); doc.add_paragraph(caption,style='Caption')
    else: doc.add_paragraph(f'[Figure pending: {name}. Generate it by running the relevant pipeline step.]')

def main():
    doc=Document(); sec=doc.sections[0]; sec.top_margin=Inches(.7); sec.bottom_margin=Inches(.7)
    styles=doc.styles; styles['Normal'].font.name='Calibri'; styles['Normal'].font.size=Pt(10)
    doc.add_heading('WILDFIRE DETECTION USING MULTISPECTRAL SATELLITE IMAGERY',0)
    doc.add_paragraph('Deep Learning Classification with a Baseline CNN and EfficientNetB0',style='Subtitle')
    doc.add_paragraph('Project report | Student: ____________________ | Institution: ____________________ | Date: ____________________')
    doc.add_page_break()
    chapters=[
    ('Abstract','This project investigates patch-level wildfire detection from multispectral Sentinel-2 satellite imagery. The Sen2Fire benchmark provides 2,466 512×512 patches with 12 Sentinel-2 spectral bands and a Sentinel-5P aerosol channel. Five input channels (B4, B3, B2, B8, B12) are selected; NBR is computed separately for spectral analysis. A four-block CNN is compared with ImageNet-pretrained EfficientNetB0 using a learnable spectral adapter. Results are populated from the actual test run; no performance is assumed in advance.'),
    ('1. Introduction','Wildfire detection from satellite data is challenging because active flames occupy a small fraction of an image, clouds and bright surfaces create false alarms, and satellite observations are intermittent. Multispectral channels provide information beyond visible RGB.'),
    ('2. Problem Definition','The task is binary patch classification: Fire (1) or No-Fire (0). A patch is positive when its fire-mask fraction meets the documented threshold. The threshold is configurable and must be reported.'),
    ('3. Dataset Description','Sen2Fire contains 2,466 patches. The official split uses scene1+scene2 for training (1,458), scene3 for validation (504), and scene4 for testing (504). This scene-based split is retained to reduce leakage from overlapping neighboring patches.'),
    ('4. Dataset Inspection and EDA','Document the NPZ schema, dimensions, data types, label values, missing values, class counts, and positive-pixel fractions. Include representative fire and no-fire samples.'),
    ('5. Spectral Band Selection','The selected stack is B4 red, B3 green, B2 blue, B8 NIR, and B12 SWIR2. The order follows the dataset B1–B12 convention. Aerosol B13 is not used in the primary model.'),
    ('6. NBR Computation','NBR = (NIR − SWIR2) / (NIR + SWIR2 + ε). NBR is included as an analysis layer. It is commonly used for vegetation/burn change analysis and is not, by itself, a definitive active-flame detector.'),
    ('7. Preprocessing','Patches are resized to 224×224. Per-band mean and standard deviation are computed on training data only. Augmentation is restricted to training: 90-degree rotations, horizontal/vertical flips, and mild multiplicative brightness changes.'),
    ('8. Class Imbalance','Class weights are computed from training patch labels using balanced weighting. Report class counts and weights from artifacts/training_metadata.json.'),
    ('9. Baseline CNN','The baseline uses four convolutional blocks with 32, 64, 128, and 256 filters, batch normalization, max pooling, dropout, global average pooling, a 256-unit dense layer, and sigmoid output.'),
    ('10. EfficientNetB0 Transfer Learning','EfficientNetB0 is initialized with ImageNet weights. A trainable 1×1 spectral adapter maps the five multispectral channels to three channels for the pretrained backbone. The backbone is initially frozen.'),
    ('11. Fine-Tuning','After the frozen stage, the final 40 backbone layers are unfrozen and trained with Adam at 1e-4. Early stopping and learning-rate reduction are used.'),
    ('12. Evaluation Protocol','Report accuracy, per-class precision/recall/F1, macro-F1, ROC-AUC, confusion matrix, and ROC curve. Fire recall is emphasized, but precision and false alarms must also be considered.'),
    ('13. Results','Insert the actual metrics from artifacts/metrics.json. Do not fill target values unless the measured test result reaches them.'),
    ('14. Grad-CAM','Grad-CAM overlays are qualitative diagnostics. Check whether activation aligns with fire-mask regions and inspect false positives on clouds, smoke, bright soil, and other confusing surfaces.'),
    ('15. Flask Deployment','The Flask prototype accepts Sen2Fire-format NPZ files, predicts class/confidence, reports mean NBR, assigns a configurable risk band, supports multiple files, and exports a CSV alert report.'),
    ('16. Limitations','The dataset labels are derived from MOD14A1 and are not perfect ground truth. Patch labels discard within-patch spatial detail. Scene geography may limit generalization. NBR should not be treated as a direct active-fire measurement. The prototype is not an operational warning system.'),
    ('17. Future Work','Explore pixel-level segmentation, temporal sequences, VIIRS/FIRMS cross-validation, cloud masking, calibration, threshold selection for high recall, and external validation across continents and seasons.'),
    ('18. Error Analysis','Review false negatives and false positives. Categorize errors caused by cloud, smoke, bright soil, sparse fire pixels, sensor artifacts, and geolocation mismatch.'),
    ('19. Threshold Selection','The default threshold is 0.50. Use validation data to select a threshold that balances Fire recall and false-alarm burden; do not tune on the test set.'),
    ('20. Precision–Recall Analysis','Because fire patches may be rare, precision–recall curves and per-class metrics are more informative than accuracy alone.'),
    ('21. Ablation Study','Compare RGB-only, RGB+NIR+SWIR2, and optional NBR-augmented inputs. Keep all comparisons on the same scene split and random seed.'),
    ('22. Normalization Statistics','Include the exact train-only mean and standard deviation saved in models/preprocessing.npz and training_metadata.json.'),
    ('23. Augmentation Study','Explain 90-degree rotations, horizontal/vertical flips, and mild brightness changes. Confirm no augmentation was applied to validation or test data.'),
    ('24. Model Complexity and Runtime','Record parameter counts, training time per epoch, inference time per patch, memory use, and hardware.'),
    ('25. Deployment Architecture','Describe the browser upload, Flask route, preprocessing, model inference, risk band, and CSV report generation.'),
    ('26. Risk Bands and Alert Semantics','Explain that risk bands are probability-based UI categories and are not calibrated fire-danger ratings or official emergency alerts.'),
    ('27. Reproducibility and Ethics','Record package versions, data citation, license review, random seed, data provenance, and the limits of automated fire detection.'),
    ('28. Conclusion','Summarize the measured performance and error patterns after running the complete experiments. State whether the required thresholds were achieved, and distinguish research performance from operational readiness.')]
    for title,body in chapters:
        doc.add_heading(title,level=1); doc.add_paragraph(body)
        if title.startswith('4.'): add_fig(doc,'class_distribution.png','Figure: Patch-level class distribution.')
        if title.startswith('4.'): add_fig(doc,'sample_patches.png','Figure: Example training patches.')
        if title.startswith('6.'): add_fig(doc,'nbr_visualization.png','Figure: NBR sample visualization (generated after data inspection).')
        if title.startswith('9.'): add_fig(doc,'baseline_accuracy.png','Figure: Baseline training/validation accuracy.'); add_fig(doc,'baseline_loss.png','Figure: Baseline training/validation loss.')
        if title.startswith('10.') or title.startswith('11.'): add_fig(doc,'efficientnet_accuracy.png','Figure: EfficientNetB0 training/validation accuracy.'); add_fig(doc,'efficientnet_loss.png','Figure: EfficientNetB0 training/validation loss.')
        if title.startswith('13.'): add_fig(doc,'baseline_confusion_matrix.png','Figure: Baseline CNN confusion matrix.'); add_fig(doc,'efficientnet_confusion_matrix.png','Figure: EfficientNetB0 confusion matrix.'); add_fig(doc,'baseline_roc_curve.png','Figure: Baseline CNN ROC curve.'); add_fig(doc,'efficientnet_roc_curve.png','Figure: EfficientNetB0 ROC curve.')
        if title.startswith('14.'): add_fig(doc,'gradcam_example.png','Figure: RGB, NBR and Grad-CAM example.')
        doc.add_paragraph('Experimental notes / interpretation: ______________________________________________________________')
        doc.add_page_break()
    doc.add_heading('Appendix A – Reproducibility',level=1); doc.add_paragraph('Record Python/TensorFlow versions, hardware, random seed, training duration, selected checkpoint, threshold, normalization statistics, and exact commands used.')
    doc.add_heading('Appendix B – Dataset Citation',level=1); doc.add_paragraph('Xu, Y., Berg, A., & Haglund, L. (2024). Sen2Fire: A Challenging Benchmark Dataset for Wildfire Detection using Sentinel Data. IGARSS 2024. DOI: 10.1109/IGARSS53475.2024.10641441.')
    doc.add_page_break()
    # Mandatory last section and final text in document.
    doc.add_heading('Dataset Link',level=1)
    doc.add_paragraph('Sen2Fire dataset (Zenodo): https://zenodo.org/records/10881058')
    doc.save(OUT); print(OUT)
if __name__=='__main__': main()
