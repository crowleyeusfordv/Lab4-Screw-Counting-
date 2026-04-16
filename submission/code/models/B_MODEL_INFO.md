# B Detector Model Info

- file: `detector.pt`
- role: B module screw detector weight
- source run: `runs/segment/yolo_train/experiments/y11s_finetune_img7_conservative_e20/weights/best.pt`
- copied on: `2026-04-16`
- size: about 20 MB
- sha256: `53ef74f62849bde3cce9e631bf3f1a1ebc4d8bc724d922f6a826466596f52a8d`

## Notes

- This repo expects detector weights at `submission/code/models/detector.pt`.
- If needed, you can override weight path with `--detector_weights` in `run.py`.
- This model is integrated for upload handoff.
- Class-wise detector threshold config is integrated at `submission/code/configs/b_detector_thresholds.json` and auto-loaded by `modules/detector.py` when available.
