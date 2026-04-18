#!/usr/bin/env python3
"""
tools/debug_detector_image.py - visualize detector outputs on one image or a directory of images.

Examples:
  python tools/debug_detector_image.py \
      --image /path/to/image.jpg \
      --output ./debug_vis.jpg

  python tools/debug_detector_image.py \
      --image /path/to/image_dir \
      --output ./debug_vis_dir \
      --detector_weights ./models/detector.pt \
      --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.detector import DEFAULT_WEIGHTS, Detector
from utils.visualizer import CLASS_COLORS, Visualizer, draw_bbox


SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("debug_detector_image")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize detector outputs on a single image or a directory of images."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image path, or a directory that contains images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output image path in single-image mode, or output directory in directory mode.",
    )
    parser.add_argument(
        "--detector_weights",
        type=str,
        default="submission/code/models/detector.pt",
        help="Detector weights path. Defaults to submission/code/models/detector.pt.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Inference device, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--no_fp16",
        action="store_true",
        default=False,
        help="Disable FP16 inference.",
    )
    parser.add_argument(
        "--use_sahi",
        action="store_true",
        default=False,
        help="Enable SAHI slicing during detection.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Detector Debug",
        help="Title shown in the visualization banner.",
    )
    return parser.parse_args()


def _normalize_class_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _class_name_to_color(class_name: str) -> tuple[int, int, int]:
    norm = _normalize_class_name(class_name)
    if norm.startswith("type") and norm[4:].isdigit():
        idx = int(norm[4:]) - 1
        if idx in CLASS_COLORS:
            return CLASS_COLORS[idx]
    return (0, 255, 128)


def _class_name_sort_key(name: str) -> tuple[int, str]:
    norm = _normalize_class_name(name)
    if norm.startswith("type") and norm[4:].isdigit():
        return (int(norm[4:]), norm)
    return (999, norm)


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def _collect_image_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if not _is_supported_image(input_path):
            raise ValueError(f"Unsupported image file: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    image_paths = sorted(path for path in input_path.rglob("*") if _is_supported_image(path))
    if not image_paths:
        raise FileNotFoundError(f"No supported images found under: {input_path}")
    return image_paths


def _resolve_output_path(image_path: Path, input_path: Path, output_path: Path) -> Path:
    if input_path.is_file():
        return output_path

    relative_path = image_path.relative_to(input_path)
    return output_path / relative_path.parent / f"{relative_path.stem}_debug{relative_path.suffix}"


def _visualize_image(
    image_path: Path,
    output_path: Path,
    detector: Detector,
    title: str,
    frame_id: int,
) -> int:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    detections = detector.detect(frame, frame_id=frame_id, enable_tracking=False)

    canvas = frame.copy()
    per_class = Counter()
    for det in detections:
        class_name = det.class_name or "unknown"
        per_class[class_name] += 1
        label = f"{class_name} {det.confidence:.2f}"
        color = _class_name_to_color(class_name)
        draw_bbox(
            canvas,
            det.bbox,
            color=color,
            thickness=2,
            label=label,
            label_bg=True,
            font_scale=0.55,
        )

    summary_lines = [
        title,
        f"image: {image_path.name}",
        f"mode: {'YOLO' if detector.is_yolo_mode else 'fallback'}",
        f"detections: {len(detections)}",
    ]
    for cls_name in sorted(per_class.keys(), key=_class_name_sort_key):
        summary_lines.append(f"{cls_name}: {per_class[cls_name]}")

    canvas = Visualizer.add_text_banner(
        canvas,
        summary_lines,
        position="top-left",
        font_scale=0.6,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to write image: {output_path}")

    logger.info("input: %s", image_path)
    logger.info("output: %s", output_path)
    logger.info("detections: %d", len(detections))
    if per_class:
        logger.info(
            "per-class: %s",
            ", ".join(f"{k}={per_class[k]}" for k in sorted(per_class.keys(), key=_class_name_sort_key)),
        )
    return len(detections)


def main() -> int:
    args = _parse_args()

    input_path = Path(args.image)
    output_path = Path(args.output)
    detector_weights = Path(args.detector_weights) if args.detector_weights else None

    try:
        image_paths = _collect_image_paths(input_path)
        if input_path.is_dir() and output_path.exists() and not output_path.is_dir():
            raise ValueError("When --image is a directory, --output must be a directory path.")

        detector = Detector(
            weights_path=detector_weights or DEFAULT_WEIGHTS,
            use_fp16=not args.no_fp16,
            use_sahi=args.use_sahi,
            device=args.device,
        )

        total_detections = 0
        for index, image_path in enumerate(image_paths):
            if len(image_paths) > 1:
                logger.info("[%d/%d] processing %s", index + 1, len(image_paths), image_path)

            image_output_path = _resolve_output_path(
                image_path=image_path,
                input_path=input_path,
                output_path=output_path,
            )
            total_detections += _visualize_image(
                image_path=image_path,
                output_path=image_output_path,
                detector=detector,
                title=args.title,
                frame_id=index,
            )

        logger.info(
            "processed %d image(s), total detections: %d",
            len(image_paths),
            total_detections,
        )
    except Exception as exc:
        logger.error("visualization failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
