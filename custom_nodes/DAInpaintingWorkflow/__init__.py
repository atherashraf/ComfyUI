from .DASDXLInpaintWorkflowNode import DASDXLInpaintWorkflowNode
from .DAMaskMergeSelectNode import DAMaskMergeSelectNode

NODE_CLASS_MAPPINGS = {
    "DAMaskMergeSelectNode": DAMaskMergeSelectNode,
    "DASDXLInpaintWorkflowNode": DASDXLInpaintWorkflowNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DAMaskMergeSelectNode": "DA Mask Merge Select Node",
    "DASDXLInpaintWorkflowNode": "DA Inpainting workflow Node",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]