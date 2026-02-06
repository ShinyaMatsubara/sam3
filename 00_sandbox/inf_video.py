import glob
import os
import time

import cv2
from PIL import Image

from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import load_frame, render_masklet_frame, save_masklet_video

# ---- Config ----
video_path = "/home/shinya_matsubara/code/sam3/sam3/02_input/01/test_03.mp4"
# prompt_text = "person, crane, suspended load"
prompt_text = "person wearing yellow helmet"
prompt_frame_index = 0
out_dir = "/home/shinya_matsubara/code/sam3/sam3/01_out/00/video"
output_stride = 1  # save every Nth frame
overlay_alpha = 0.5
output_video_path = os.path.join(out_dir, "overlay.mp4")


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_video_frames(video_path_value):
    if isinstance(video_path_value, str) and video_path_value.lower().endswith(".mp4"):
        cap = cv2.VideoCapture(video_path_value)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    frames = glob.glob(os.path.join(video_path_value, "*.jpg"))
    frames += glob.glob(os.path.join(video_path_value, "*.png"))
    try:
        frames.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        frames.sort()
    return frames


log("Loading model (this can take a while)...")
video_predictor = build_sam3_video_predictor()
log("Model loaded.")
log("Starting session...")
if not os.path.exists(video_path):
    raise FileNotFoundError(f"video_path not found: {video_path}")
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]
log(f"Session started: {session_id}")

log("Adding text prompt...")
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=prompt_frame_index,
        text=prompt_text,
    )
)
log("Prompt added. Propagating in video...")

outputs_per_frame = {}
start_time = time.time()
last_log_time = start_time
count = 0
for response in video_predictor.handle_stream_request(
    request=dict(
        type="propagate_in_video",
        session_id=session_id,
    )
):
    outputs_per_frame[response["frame_index"]] = response["outputs"]
    count += 1
    now = time.time()
    if count % 10 == 0 or (now - last_log_time) >= 5:
        log(
            f"Propagated {count} frames... last frame={response['frame_index']} "
            f"elapsed={int(now - start_time)}s"
        )
        last_log_time = now

os.makedirs(out_dir, exist_ok=True)
log("Loading video frames for visualization...")
video_frames = load_video_frames(video_path)
if len(video_frames) == 0:
    raise RuntimeError(f"No frames loaded from video_path: {video_path}")
if len(outputs_per_frame) == 0:
    raise RuntimeError("No outputs returned from propagate_in_video (empty results).")

log("Saving overlay video...")
# Save a single overlay video
save_masklet_video(
    video_frames=video_frames,
    outputs=outputs_per_frame,
    out_path=output_video_path,
    alpha=overlay_alpha,
)
log(f"Done. Saved video to {output_video_path}")

_ = video_predictor.handle_request(
    request=dict(
        type="close_session",
        session_id=session_id,
    )
)
