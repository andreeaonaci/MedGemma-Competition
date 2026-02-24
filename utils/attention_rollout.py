import torch
import torch.nn.functional as F
import cv2

class AttentionRollout:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    def generate(self, inputs):

        with torch.no_grad():
            output = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True
            )

        # Try to get attentions from output
        attentions = None
        if hasattr(output, 'attentions') and output.attentions is not None:
            # attentions: tuple of (layers, batch, heads, tokens, tokens)
            attentions = output.attentions
        elif isinstance(output, dict) and 'attentions' in output:
            attentions = output['attentions']
        else:
            raise RuntimeError("Model did not return attention maps. Check model and output_attentions support.")

        # Stack attentions: [layers, batch, heads, tokens, tokens]
        attn_stack = torch.stack(attentions)
        # Average heads
        attn_stack = attn_stack.mean(dim=2)
        # Add identity matrix
        num_tokens = attn_stack.size(-1)
        eye = torch.eye(num_tokens, device=self.device)
        attn_stack = attn_stack + eye
        # Normalize rows
        attn_stack = attn_stack / attn_stack.sum(dim=-1, keepdim=True)
        # Rollout multiplication
        joint_attn = attn_stack[0]
        for i in range(1, attn_stack.size(0)):
            joint_attn = torch.matmul(attn_stack[i], joint_attn)
        # CLS → patch tokens
        cls_attn = joint_attn[:, 0, 1:]
        heatmap = cls_attn.squeeze(0)
        # Normalize safely
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap
        

def overlay_heatmap(original_image, heatmap_tensor):

    h, w = original_image.shape[:2]

    num_patches = heatmap_tensor.shape[0]
    patch_dim = int(torch.sqrt(torch.tensor(num_patches, dtype=torch.float32)))

    heatmap = heatmap_tensor.reshape(patch_dim, patch_dim)

    heatmap = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode="bilinear",
        align_corners=False
    ).squeeze()

    heatmap = heatmap.cpu()
    heatmap = heatmap / (heatmap.max() + 1e-8)

    heatmap_uint8 = (heatmap * 255).byte().numpy()

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(original_image, 0.6, heatmap_color, 0.4, 0)

    return overlay