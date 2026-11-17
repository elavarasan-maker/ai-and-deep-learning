# Build status (2026-11-17)

## Completed in this starter build
- Selected and documented the Sen2Fire multispectral dataset.
- Implemented dataset download/extraction script.
- Implemented NPZ validation, five-band selection, NBR calculation, train-only normalization statistics, patch-label derivation, augmentation, and official scene splits.
- Implemented baseline CNN and EfficientNetB0 transfer-learning model with a trainable 5-to-3 spectral adapter.
- Implemented class weights, training callbacks, fine-tuning, evaluation metrics, confusion matrices, ROC curves, Grad-CAM visualization, Flask single/batch upload, and CSV report export.
- Created a 30+ section DOCX report template and report builder that embeds generated figures when they exist.

## Not yet completed (requires downloading/running the 6.3 GB dataset)
- Actual dataset inspection and class counts.
- Model training and saved model weights.
- Measured test metrics and comparison results.
- Generated plots, sample predictions, and Grad-CAM outputs.
- Live Flask validation with trained weights.
- Final evidence-complete report with all screenshots embedded.

Do not submit the starter report as a completed experimental report. Run the pipeline, inspect all figures, fill the student/institution details, and verify every reported metric first.
