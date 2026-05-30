from .DAFaceAlignAndResizeNode import DAFaceAlignAndResizeNode
from .DAPoseTransferPrepareNode import DAPoseTransferPrepareNode
from .DASDXLPoseTransferNode import DASDXLPoseTransferNode
from .DASDXLCameraTransferNode import DASDXLCameraTransferNode
# from .DAMaskMergeSelectNode import DAMaskMergeSelectNode
# from .DASDXLInpaintWorkflowNode import DASDXLInpaintWorkflowNode
from .DAPersonAlignToTargetNode import DAPersonAlignToTargetNode
NODE_CLASS_MAPPINGS = {
    "DAFaceAlignAndResizeNode": DAFaceAlignAndResizeNode,
    "DAPoseTransferPrepareNode": DAPoseTransferPrepareNode,
    "DASDXLPoseTransferNode": DASDXLPoseTransferNode,
    "DASDXLCameraTransferNode": DASDXLCameraTransferNode,
    # "DAMaskMergeSelectorNode": DAMaskMergeSelectNode,
    # "DASDXLInpaintWorkflowNode": DASDXLInpaintWorkflowNode,
    "DAPersonAlignToTargetNode": DAPersonAlignToTargetNode,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DAFaceAlignAndResizeNode": "DA Face Align and Resize",
    "DAPoseTransferPrepareNode": "DA Pose Transfer Prepare",
    "DASDXLPoseTransferNode": "DA SDXL Pose Transfer",
    "DASDXLCameraTransferNode": "DA SDXL Camera Transfer",
    # "DAMaskMergeSelectNode": "DA Mask Merge Selector",
    # "DASDXLInpaintWorkflowNode": "DA SDXL Inpaint Workflow",
    "DAPersonAlignToTargetNode": "DA Person Align To Target",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]