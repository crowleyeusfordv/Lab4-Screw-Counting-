#!/usr/bin/env python3
"""
Step 2 debug script:
1. read one video
2. extract keyframes
3. run detector on keyframes
4. run registration to build a shared reference coordinate system
5. project detection centers onto the reference frame

Examples:
  python script/step2_register_project.py --input ../../video_exp/IMG_2374.MOV --output ./debug_step2/IMG_2374 --keyframe_strategy uniform --uniform_count 12 --anchor_count 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import modules.registration as registration_module
from modules.detector import DEFAULT_WEIGHTS, Detector
from modules.registration import FrameRegistration
from pipeline import _extract_keyframes_motion, extract_keyframes_uniform
from utils.video_io import VIDEO_EXTENSIONS, VideoReader
from utils.visualizer import CLASS_COLORS, Visualizer, draw_bbox


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("step2_register_project")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: read video, extract keyframes, run detector, then registration and projection."
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
        default="./debug_step2_output",
        help="Output directory for frames, detections, registration artifacts, and summary.json.",
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
        default=30,
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
    parser.add_argument(
        "--feature_type",
        type=str,
        default=registration_module.FEATURE_TYPE,
        choices=["AKAZE", "ORB"],
        help="Registration feature type.",
    )
    parser.add_argument(
        "--anchor_strategy",
        type=str,
        default=registration_module.ANCHOR_STRATEGY,
        choices=["first", "middle"],
        help="Preferred root-anchor strategy.",
    )
    parser.add_argument(
        "--anchor_count",
        type=int,
        default=10,
        help="Number of anchors used to build the reference coordinate system.",
    )
    parser.add_argument(
        "--inlier_ratio_threshold",
        type=float,
        default=registration_module.INLIER_RATIO_THRESHOLD,
        help="Registration validity threshold.",
    )
    parser.add_argument(
        "--min_match_count",
        type=int,
        default=registration_module.MIN_MATCH_COUNT,
        help="Minimum match count for registration.",
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


def _make_frame_stem(video_name: str, frame_id: int) -> str:
    return f"{video_name}_frame{frame_id:06d}"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_detection_canvas(
    frame: np.ndarray,
    detections,
    video_name: str,
    frame_id: int,
    detector_mode: str,
) -> tuple[np.ndarray, dict[str, int]]:
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
        "Step 2 - Detection Before Registration",
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


def _frame_color(index: int, total: int) -> tuple[int, int, int]:
    denom = max(total, 1)
    hue = int(round(179 * (index % denom) / denom))
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_projected_centers(
    reference_frame: np.ndarray,
    all_detections,
    registrations,
    reference_name: str,
    reference_frame_id: int,
    anchor_indices: list[int],
) -> np.ndarray:
    canvas = reference_frame.copy()
    valid_regs = sum(1 for reg in registrations if reg.valid)

    for frame_index, (frame_dets, reg) in enumerate(zip(all_detections, registrations)):
        color = _frame_color(frame_index, len(registrations))
        for det in frame_dets:
            center = reg.project_point(det.center())
            if center is None:
                continue
            cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
            cv2.circle(canvas, (cx, cy), 4, color, -1)
            cv2.circle(canvas, (cx, cy), 7, color, 1)

    summary_lines = [
        "Projected detections after registration",
        f"reference: {reference_name}",
        f"reference_frame_id: {reference_frame_id}",
        f"anchors: {len(anchor_indices)}",
        f"frames: {len(registrations)}",
        f"valid_registrations: {valid_regs}",
    ]
    return Visualizer.add_text_banner(canvas, summary_lines, position="top-left", font_scale=0.6)


def main() -> int:
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

    registration_module.ANCHOR_STRATEGY = args.anchor_strategy
    registrar = FrameRegistration(
        feature_type=args.feature_type,
        inlier_ratio_threshold=args.inlier_ratio_threshold,
        min_match_count=args.min_match_count,
    )

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

            frame_ids: list[int] = []
            image_names: list[str] = []
            frames_hr: list[np.ndarray] = []
            frames_lr: list[np.ndarray] = []
            full_res_scales: list[float] = []
            all_detections = []
            frame_summaries: list[dict[str, object]] = []
            total_detections = 0
            overall_histogram = Counter()

            logger.info("Video: %s", input_path.name)
            logger.info("Keyframes: %d", len(keyframe_ids))
            logger.info("Detector mode: %s", detector_mode)
            logger.info("Registration: feature=%s, anchor_count=%d, anchor_strategy=%s", args.feature_type, max(1, args.anchor_count), args.anchor_strategy)

            for index, (frame_id, frame_hr, frame_lr) in enumerate(
                reader.iter_frames_at(keyframe_ids, yield_low_res=True),
                start=1,
            ):
                if frame_hr is None or frame_lr is None:
                    logger.warning("Failed to read frame %d, skipping.", frame_id)
                    continue

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

                frame_ids.append(frame_id)
                image_names.append(raw_frame_path.name)
                frames_hr.append(frame_hr)
                frames_lr.append(frame_lr)
                full_res_scales.append(reader.meta.low_res_scale)
                all_detections.append(detections)

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

            if not frames_hr:
                logger.error("No valid keyframes were loaded from video: %s", input_path)
                return 1

            logger.info("Running registration...")
            registrations = registrar.register_sequence(
                keyframe_images=frames_lr,
                keyframe_ids=frame_ids,
                full_res_scales=full_res_scales,
                anchor_count=max(1, args.anchor_count),
            )
            reg_stats = registrar.get_stats()
            reg_sequence_info = registrar.get_last_sequence_info()
            valid_reg_count = sum(1 for reg in registrations if reg.valid)

            frame_anchor_indices = reg_sequence_info.get("frame_anchor_indices", [None] * len(registrations))
            for frame_summary, reg, used_anchor_idx in zip(frame_summaries, registrations, frame_anchor_indices):
                frame_summary["registration_valid"] = reg.valid
                frame_summary["registration_inlier_ratio"] = round(float(reg.inlier_ratio), 4)
                frame_summary["used_anchor_index"] = used_anchor_idx
                if used_anchor_idx is not None and 0 <= used_anchor_idx < len(frame_ids):
                    frame_summary["used_anchor_frame_id"] = frame_ids[used_anchor_idx]
                    frame_summary["used_anchor_name"] = image_names[used_anchor_idx]

            reference_index = int(reg_sequence_info.get("reference_index", 0))
            reference_frame_hr = frames_hr[reference_index]
            reference_frame_id = frame_ids[reference_index]
            reference_name = image_names[reference_index]

            anchor_indices = [
                int(idx)
                for idx in reg_sequence_info.get("anchor_indices", [reference_index])
                if idx is not None
            ]
            anchor_frame_ids = [
                frame_ids[idx]
                for idx in anchor_indices
                if 0 <= idx < len(frame_ids)
            ]

            cv2.imwrite(str(output_dir / "reference_frame.jpg"), reference_frame_hr)

            projected_canvas = _draw_projected_centers(
                reference_frame=reference_frame_hr,
                all_detections=all_detections,
                registrations=registrations,
                reference_name=reference_name,
                reference_frame_id=reference_frame_id,
                anchor_indices=anchor_indices,
            )
            cv2.imwrite(str(output_dir / "projected_centers.jpg"), projected_canvas)

            summary = {
                "step": 2,
                "task": "video_extract_detect_register_project",
                "input_video": str(input_path),
                "output_dir": str(output_dir),
                "video_name": input_path.name,
                "video_meta": {
                    "frame_count": reader.meta.frame_count,
                    "fps": reader.meta.fps,
                    "width": reader.meta.width,
                    "height": reader.meta.height,
                    "rotation": reader.meta.rotation,
                    "low_res_scale": reader.meta.low_res_scale,
                },
                "keyframe": {
                    "strategy_requested": args.keyframe_strategy,
                    "strategy_used": actual_strategy,
                    "uniform_count": args.uniform_count,
                    "count": len(frame_ids),
                    "frame_ids": frame_ids,
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
                "registration": {
                    "feature_type": args.feature_type,
                    "anchor_strategy": args.anchor_strategy,
                    "anchor_count_requested": max(1, args.anchor_count),
                    "mode": reg_sequence_info.get("mode", "single_anchor"),
                    "reference_index": reference_index,
                    "reference_frame_id": reference_frame_id,
                    "reference_name": reference_name,
                    "anchor_indices": anchor_indices,
                    "anchor_frame_ids": anchor_frame_ids,
                    "valid_anchor_indices": reg_sequence_info.get("valid_anchor_indices", anchor_indices),
                    "anchor_parent_indices": reg_sequence_info.get("anchor_parent_indices", {}),
                    "inlier_ratio_threshold": args.inlier_ratio_threshold,
                    "min_match_count": args.min_match_count,
                    "valid_frames": valid_reg_count,
                    "invalid_frames": len(registrations) - valid_reg_count,
                    "stats": reg_stats,
                },
                "frames": frame_summaries,
            }
    except Exception as exc:
        logger.error("Step 2 failed: %s", exc, exc_info=True)
        return 1

    _write_json(output_dir / "summary.json", summary)
    logger.info("Saved raw keyframes to: %s", frames_dir)
    logger.info("Saved detection overlays to: %s", detections_dir)
    logger.info("Saved reference frame to: %s", output_dir / "reference_frame.jpg")
    logger.info("Saved projected centers to: %s", output_dir / "projected_centers.jpg")
    logger.info("Saved summary to: %s", output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
