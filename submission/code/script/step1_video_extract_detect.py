#!/usr/bin/env python3
"""
Step 1 debug script:
1. read one video
2. extract keyframes
3. run detector on keyframes
4. save raw frames, detection overlays, and a summary json

Examples:
  python script/step1_video_extract_detect.py --input ../../video_exp/IMG_2374.MOV --output ./debug_step1/IMG_2374 --keyframe_strategy uniform --uniform_count 12
"""

from __future__ import annotations

import argparse
import json
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
from pipeline import _extract_keyframes_motion, extract_keyframes_uniform
from utils.video_io import VIDEO_EXTENSIONS, VideoReader
from utils.visualizer import CLASS_COLORS, Visualizer, draw_bbox


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("step1_video_extract_detect")


def _parse_args() -> argparse.Namespace:
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Step 1: read video, extract keyframes, and run detector on keyframes."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input video path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./debug_step1_output",
        help="Output directory for frames, detections, and summary.json.",
    )
    parser.add_argument(
        "--keyframe_strategy",
        type=str,
        default="uniform",
        choices=["motion", "uniform"],
        help="Keyframe extraction strategy.",
    )
    parser.add_argument(
        "--uniform_count",
        type=int,
        default=15,
        help="Number of keyframes when --keyframe_strategy uniform.",
    )
    parser.add_argument(
        "--detector_weights",
        type=str,
        default="./models/detector.pt",
        help="Detector weights path.",
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
    return parser.parse_args()


def _normalize_class_name(name: str) -> str:
    """Normalize class name."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _class_name_to_color(class_name: str) -> tuple[int, int, int]:
    """Class name to color."""
    norm = _normalize_class_name(class_name)
    if norm.startswith("type") and norm[4:].isdigit():
        idx = int(norm[4:]) - 1
        if idx in CLASS_COLORS:
            return CLASS_COLORS[idx]
    return (0, 255, 128)


def _class_name_sort_key(name: str) -> tuple[int, str]:
    """Class name sort key."""
    norm = _normalize_class_name(name)
    if norm.startswith("type") and norm[4:].isdigit():
        return (int(norm[4:]), norm)
    return (999, norm)


def _make_frame_stem(video_name: str, frame_id: int) -> str:
    """Make frame stem."""
    return f"{video_name}_frame{frame_id:06d}"


def _write_json(path: Path, payload: dict) -> None:
    """Write json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_detection_canvas(
    frame,
    detections,
    video_name: str,
    frame_id: int,
    detector_mode: str,
) -> tuple[cv2.typing.MatLike, dict[str, int]]:
    """Build detection canvas."""
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
        "Step 1 - Video / Keyframes / Detection",
        f"video: {video_name}",
        f"frame_id: {frame_id}",
        f"mode: {detector_mode}",
        f"detections: {len(detections)}",
    ]
    for class_name in sorted(per_class.keys(), key=_class_name_sort_key):
        summary_lines.append(f"{class_name}: {per_class[class_name]}")

    canvas = Visualizer.add_text_banner(
        canvas,
        summary_lines,
        position="top-left",
        font_scale=0.6,
    )
    return canvas, dict(per_class)


def main() -> int:
    """Main."""
    args = _parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.is_file():
        logger.error("Input video does not exist: %s", input_path)
        return 1
    if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
        logger.error("Unsupported video file: %s", input_path)
        return 1

    frames_dir = output_dir / "frames"
    detections_dir = output_dir / "detections"
    frames_dir.mkdir(parents=True, exist_ok=True)
    detections_dir.mkdir(parents=True, exist_ok=True)

    detector = Detector(
        weights_path=Path(args.detector_weights),
        use_fp16=not args.no_fp16,
        use_sahi=args.use_sahi,
        device=args.device,
    )
    detector_mode = "YOLO" if detector.is_yolo_mode else "fallback"

    try:
        with VideoReader(input_path) as reader:
            if args.keyframe_strategy == "motion":
                keyframe_ids = _extract_keyframes_motion(reader)
                actual_strategy = "motion"
                if not keyframe_ids:
                    logger.warning("Motion keyframe extraction returned no frames. Fallback to uniform sampling.")
                    keyframe_ids = extract_keyframes_uniform(reader, target_count=args.uniform_count)
                    actual_strategy = "uniform_fallback"
            else:
                keyframe_ids = extract_keyframes_uniform(reader, target_count=args.uniform_count)
                actual_strategy = "uniform"

            if not keyframe_ids:
                logger.error("No keyframes extracted from video: %s", input_path)
                return 1

            logger.info("Video: %s", input_path.name)
            logger.info("Keyframes: %d", len(keyframe_ids))
            logger.info("Detector mode: %s", detector_mode)

            frame_summaries: list[dict[str, object]] = []
            total_detections = 0
            overall_histogram = Counter()

            for index, (frame_id, frame_hr, _frame_lr) in enumerate(
                reader.iter_frames_at(keyframe_ids, yield_low_res=False),
                start=1,
            ):
                frame_stem = _make_frame_stem(input_path.stem, frame_id)
                raw_frame_path = frames_dir / f"{frame_stem}.jpg"
                det_frame_path = detections_dir / f"{frame_stem}_det.jpg"

                detections = detector.detect(frame_hr, frame_id=frame_id, enable_tracking=False)
                canvas, per_class = _build_detection_canvas(
                    frame=frame_hr,
                    detections=detections,
                    video_name=input_path.name,
                    frame_id=frame_id,
                    detector_mode=detector_mode,
                )

                cv2.imwrite(str(raw_frame_path), frame_hr)
                cv2.imwrite(str(det_frame_path), canvas)

                total_detections += len(detections)
                overall_histogram.update(per_class)

                frame_summaries.append(
                    {
                        "index": index - 1,
                        "frame_id": frame_id,
                        "frame_name": raw_frame_path.name,
                        "detection_name": det_frame_path.name,
                        "n_detections": len(detections),
                        "class_histogram": per_class,
                    }
                )

                logger.info(
                    "[%d/%d] frame_id=%d detections=%d",
                    index,
                    len(keyframe_ids),
                    frame_id,
                    len(detections),
                )

            summary = {
                "step": 1,
                "task": "video_read_extract_detect",
                "input_video": str(input_path),
                "output_dir": str(output_dir),
                "video_name": input_path.name,
                "video_meta": {
                    "frame_count": reader.meta.frame_count,
                    "fps": reader.meta.fps,
                    "width": reader.meta.width,
                    "height": reader.meta.height,
                    "rotation": reader.meta.rotation,
                },
                "keyframe": {
                    "strategy_requested": args.keyframe_strategy,
                    "strategy_used": actual_strategy,
                    "uniform_count": args.uniform_count,
                    "count": len(keyframe_ids),
                    "frame_ids": keyframe_ids,
                },
                "detector": {
                    "mode": detector_mode,
                    "weights_path": str(Path(args.detector_weights)),
                    "device": args.device,
                    "use_fp16": not args.no_fp16,
                    "use_sahi": args.use_sahi,
                },
                "detection": {
                    "total_detections": total_detections,
                    "overall_class_histogram": {
                        name: overall_histogram[name]
                        for name in sorted(overall_histogram.keys(), key=_class_name_sort_key)
                    },
                },
                "frames": frame_summaries,
            }
    except Exception as exc:
        logger.error("Step 1 failed: %s", exc, exc_info=True)
        return 1

    _write_json(output_dir / "summary.json", summary)
    logger.info("Saved raw keyframes to: %s", frames_dir)
    logger.info("Saved detection overlays to: %s", detections_dir)
    logger.info("Saved summary to: %s", output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
