import torch
import cv2
import numpy as np
import folder_paths
from PIL import Image
import os

class FaceAlignmentNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "include_hair": ("BOOLEAN", {"default": True}),
                "blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "align_faces"
    CATEGORY = "image/face"
    
    def align_faces(self, source_image, target_image, include_hair=True, blend_strength=0.5):
        # Convert ComfyUI tensors to numpy
        source_np = (source_image[0].cpu().numpy() * 255).astype(np.uint8)
        target_np = (target_image[0].cpu().numpy() * 255).astype(np.uint8)
        
        # Convert BGR to RGB
        source_np = cv2.cvtColor(source_np, cv2.COLOR_RGB2BGR)
        target_np = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
        
        try:
            # Import face-alignment (will install if not available)
            try:
                import face_alignment
            except ImportError:
                print("Installing face-alignment...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "face-alignment"])
                import face_alignment
            
            # Initialize face alignment
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                device=device,
                face_detector='sfd'
            )
            
            # Get landmarks
            landmarks1 = fa.get_landmarks(source_np)
            landmarks2 = fa.get_landmarks(target_np)
            
            if not landmarks1 or not landmarks2:
                print("Warning: Could not detect faces, returning original")
                return (target_image,)
            
            landmarks1 = landmarks1[0]
            landmarks2 = landmarks2[0]
            
            # Calculate transformation
            M, _ = cv2.estimateAffinePartial2D(landmarks1, landmarks2)
            
            # Warp source to target
            aligned = cv2.warpAffine(
                source_np, 
                M, 
                (target_np.shape[1], target_np.shape[0]),
                flags=cv2.INTER_LINEAR
            )
            
            # Convert back to RGB
            aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            aligned_tensor = torch.from_numpy(aligned).float() / 255.0
            aligned_tensor = aligned_tensor.unsqueeze(0)
            
            return (aligned_tensor,)
            
        except Exception as e:
            print(f"Error in face alignment: {e}")
            return (target_image,)

NODE_CLASS_MAPPINGS = {
    "FaceAlignmentNode": FaceAlignmentNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceAlignmentNode": "Face Alignment"
}