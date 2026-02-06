import cv2
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import supervision as sv

from sam2.build_sam import build_sam2_video_predictor
from insightface.app import FaceAnalysis

class SAM2VideoProcessor:
    def __init__(self, video_predictor, face_analyzer, output_dir):
        self.video_predictor = video_predictor
        self.face_analyzer = face_analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _read_first_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Cannot read video from {}".format(video_path))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def _save_frame(self, frame_rgb, video_id):
        image_path = self.output_dir / f"{video_id}_0.png"
        Image.fromarray(frame_rgb).save(image_path)
        print("Saved frame to", image_path)
        return image_path

    def _save_mask(self, video_id, idx, mask):
        mask_path = self.output_dir / f"{video_id}_person{idx}.png"
        vis_mask = mask * 255
        Image.fromarray(vis_mask).save(mask_path)
        print("Saved mask to", mask_path)
        return mask_path

    def detect_faces(self, first_frame, face_num):
        faces = self.face_analyzer.get(first_frame)
        print("Detected face boxes:")
        for i, face in enumerate(faces[:2]):
            x1, y1, x2, y2 = map(int, face["bbox"])
            print(f"!!!Face {i}: bbox = ({x1}, {y1}, {x2}, {y2})")

        num = len(faces)
        if num != face_num:
            print(f"{first_frame} is not a valid image, only {num} faces")
        faces_sorted = sorted(faces, key=lambda x: (x["bbox"][0] + x["bbox"][2]) / 2)
        return faces_sorted

    def process_video(self, video_path):
        frame_rgb = self._read_first_frame(video_path)
        video_path = Path(video_path)
        video_id = video_path.stem
        print("!!!Processing video", video_id)
        frame_path = self._save_frame(frame_rgb, video_id)
        faces_sorted = self.detect_faces(frame_rgb, face_num=2)
        first_state = self.video_predictor.init_state(video_path=str(video_path))
        self.video_predictor.reset_state(first_state)
        persons = []
        for idx, face in enumerate(faces_sorted, start=1):
            box = np.array(face["bbox"], dtype=np.float32)
            # kps = np.array(face["kps"], dtype=np.float32)
            # labels = np.ones(len(kps), dtype=np.int32)
            self.video_predictor.add_new_points_or_box(
                inference_state=first_state,
                frame_idx=0,
                obj_id=idx,
                box=box
            )
        for frame_idx, obj_ids, mask_logits in self.video_predictor.propagate_in_video(first_state, start_frame_idx=0):
            if frame_idx != 0:
                continue
            for i, obj_id in enumerate(obj_ids):
                mask = (mask_logits[i] > -1).cpu().numpy()
                mask = mask.squeeze()
                mask = (mask.astype(np.uint8)) * 255
                mask_path = self._save_mask(video_id, obj_id, mask)
                persons.append({
                    "person_id": obj_id,
                    "bbox": [float(x) for x in faces_sorted[obj_id - 1]["bbox"]],
                    "mask_path": mask_path.as_posix()
                })
            break
        video_data = {
            "video_id": video_id,
            "video_path": video_path.as_posix(),
            "frame0_path": frame_path.as_posix(),
            "persons": persons
        }
        return video_data

    def extract_frames(self, video_path, frame_dir):
        video_info = sv.VideoInfo.from_video_path(video_path)
        frame_gen = sv.get_video_frames_generator(video_path)
        with sv.ImageSink(target_dir_path=frame_dir, overwrite=True, image_name_pattern="{:06d}.jpg") as sink:
            for frame in frame_gen:
                sink.save_image(frame)
        return video_info.total_frames

class SAM2DatasetPreprocessor:
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(self, processor, output_dir):
        self.processor = processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def video_list(self, video_dir):
        return sorted([
            p for p in video_dir.iterdir()
            if p.suffix.lower() in self.VIDEO_EXTS
        ])

    def process_videos(self, video_dir):
        video_dir = Path(video_dir)
        video_list = sorted([
            p for p in video_dir.iterdir()
            if p.suffix.lower() in self.VIDEO_EXTS
        ])
        all_results = []
        for video in tqdm(video_list, desc="Processing videos"):
            try:
                data = self.processor.process_video(video)
                if data:
                    all_results.append(data)
                else:
                    print(f"Failed to process video {video}")
                all_results.append(data)
            except Exception as e:
                print(f"[ERROR] {video}: {e}", flush=True)
        json_path = self.output_dir / "dataset_index.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print("Saved dataset to", json_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="The path to the sam2 model configuration.")
    parser.add_argument("--sam2_ckpt", type=str, required=True, help="The path to the sam2 checkpoint directory.")
    parser.add_argument("--face_encoder", type=str, required=True, help="The path to the face encoder checkpoint directory.")
    parser.add_argument("--video_dir", type=str, required=True, help="The path to the video directory.")
    parser.add_argument("--output", type=str, required=True, help="The path to the output directory.")
    args = parser.parse_args()

    video_predictor = build_sam2_video_predictor(args.config, args.sam2_ckpt)
    face_analyzer = FaceAnalysis(root=args.face_encoder, providers=['CUDAExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(320, 320))

    processor = SAM2VideoProcessor(
        video_predictor,
        face_analyzer,
        output_dir=args.output
    )
    dataset = SAM2DatasetPreprocessor(
        processor,
        output_dir=args.output
    )

    with torch.no_grad():
        dataset.process_videos(args.video_dir)

if __name__ == "__main__":
    main()
# python sam2_video.py --config configs/sam2.1/sam2.1_hiera_l.yaml --sam2_ckpt weights/sam2/sam2.1_hiera_large.pt --face_encoder weights/face_encoder --video_dir raw_video --output sam2_output