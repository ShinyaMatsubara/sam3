import os
import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image (force RGB to match model expectations)
image = Image.open("/home/shinya_matsubara/code/sam3/sam3/02_input/00/test.png").convert("RGB")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="workers, cranes, and suspended loads")
# output = processor.set_text_prompt(state=inference_state, prompt="yellow hard hat")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# Save outputs
out_dir = "/home/shinya_matsubara/code/sam3/sam3/01_out/00"
os.makedirs(out_dir, exist_ok=True)

# Save the original image for reference
image.save(os.path.join(out_dir, "input.png"))

# Save each mask as a binary PNG
combined_mask = None
for i, mask in enumerate(masks):
    if torch.is_tensor(mask):
        mask = mask.detach().cpu()
    # Ensure 2D HxW for PIL
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    mask_bin = (mask > 0).numpy().astype("uint8")
    if combined_mask is None:
        combined_mask = mask_bin
    else:
        combined_mask = (combined_mask | mask_bin)
    mask_img = Image.fromarray(mask_bin * 255)
    mask_img.save(os.path.join(out_dir, f"mask_{i:03d}.png"))

# Save a single visualization image with masks overlaid
if combined_mask is not None:
    base = image.convert("RGBA")
    mask_alpha = Image.fromarray((combined_mask * 120).astype("uint8"), mode="L")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    overlay.putalpha(mask_alpha)
    vis = Image.alpha_composite(base, overlay)
    vis.save(os.path.join(out_dir, "overlay.png"))

#################################### For Video ####################################

# from sam3.model_builder import build_sam3_video_predictor

# video_predictor = build_sam3_video_predictor()
# video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# # Start a session
# response = video_predictor.handle_request(
#     request=dict(
#         type="start_session",
#         resource_path=video_path,
#     )
# )
# response = video_predictor.handle_request(
#     request=dict(
#         type="add_prompt",
#         session_id=response["session_id"],
#         frame_index=0, # Arbitrary frame index
#         text="<YOUR_TEXT_PROMPT>",
#     )
# )
# output = response["outputs"]






