// hooks/useInpaintingState.ts
import * as React from "react";

export interface CheckpointOption {
    id: string;
    name: string;
    file: string;
    size: string;
    recommended: boolean;
    resolution: string;
    description?: string;
}

export const useInpaintingState = () => {
    const [selectedCheckpoint, setSelectedCheckpoint] = React.useState<string>("lustify-sdxl");
    const [positivePrompt, setPositivePrompt] = React.useState("");
    const [negativePrompt, setNegativePrompt] = React.useState("");

    // Your checkpoints data based on your file list
    const checkpoints: CheckpointOption[] = [
        {
            id: "512-inpainting",
            name: "512 Inpainting",
            file: "512-inpainting-ema.safetensors",
            size: "5.2 GB",
            recommended: true,
            resolution: "512x512",
            description: "Best for general-purpose inpainting"
        },
        {
            id: "sd15-inpainting",
            name: "SD 1.5 Inpainting",
            file: "sd-v1-5-inpainting.ckpt",
            size: "4.3 GB",
            recommended: true,
            resolution: "512x512",
            description: "Official Stable Diffusion 1.5 inpainting"
        },
        {
            id: "inpainting-diffusion",
            name: "Inpainting Diffusion",
            file: "inpainting_diffusion_pytorch_model.safetensors",
            size: "4.2 GB",
            recommended: true,
            resolution: "512x512",
            description: "Dedicated inpainting model"
        },
        {
            id: "lustify-sdxl",
            name: "Lustify SDXL",
            file: "lustifySDXLNSFW_v20-inpainting.safetensors",
            size: "6.9 GB",
            recommended: false,
            resolution: "1024x1024+",
            description: "SDXL-based, higher quality, NSFW/artistic"
        },
        {
            id: "hunyuan-3d",
            name: "Hunyuan 3D v2.1",
            file: "hunyuan_3d_v2.1.safetensors",
            size: "7.4 GB",
            recommended: false,
            resolution: "Various",
            description: "3D-focused generation"
        },
        {
            id: "sd35-large",
            name: "SD 3.5 Large",
            file: "sd3.5_large_fp8_scaled.safetensors",
            size: "14.9 GB",
            recommended: false,
            resolution: "1024x1024+",
            description: "General SD3.5 model (not specialized)"
        },
        {
            id: "flux-dev",
            name: "Flux Dev",
            file: "flux1-dev-fp8.safetensors",
            size: "17.2 GB",
            recommended: false,
            resolution: "Various",
            description: "General Flux model"
        },
        {
            id: "ltx-video",
            name: "LTX Video 2B",
            file: "ltx-video-2b-v0.9.safetensors",
            size: "9.4 GB",
            recommended: false,
            resolution: "Video",
            description: "Video generation model"
        },
        {
            id: "sd15-base",
            name: "SD 1.5 Base",
            file: "v1-5-pruned-emaonly-fp16.safetensors",
            size: "2.1 GB",
            recommended: false,
            resolution: "512x512",
            description: "Base SD1.5 (not for inpainting)"
        },
        {
            id: "flux-lustly",
            name: "Flux Lustly AI",
            file: "flux_lustly-ai_v1.safetensors",
            size: "344 MB",
            recommended: false,
            resolution: "Various",
            description: "Distilled/small Flux model"
        }
    ];

    // Get details of currently selected checkpoint
    const getSelectedCheckpointDetails = () => {
        return checkpoints.find(cp => cp.id === selectedCheckpoint);
    };
    const getFirstCheckpointDetails = () => {
        return checkpoints[3]
    }

    // Clear all prompts
    const clearPrompts = () => {
        setPositivePrompt("");
        setNegativePrompt("");
    };

    // Set a default checkpoint
    const setDefaultCheckpoint = () => {
        setSelectedCheckpoint("512-inpainting");
    };

    return {
        // State
        selectedCheckpoint,
        setSelectedCheckpoint,
        positivePrompt,
        setPositivePrompt,
        negativePrompt,
        setNegativePrompt,

        // Data
        checkpoints,

        // Helpers
        getSelectedCheckpointDetails,
        getFirstCheckpointDetails,
        clearPrompts,
        setDefaultCheckpoint,

        // Computed
        hasPrompts: positivePrompt.trim().length > 0 || negativePrompt.trim().length > 0,
        selectedCheckpointDetails: getSelectedCheckpointDetails(),
    };
};

export type InpaintingState = ReturnType<typeof useInpaintingState>;