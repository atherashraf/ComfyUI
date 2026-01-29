import * as React from "react";
import { Box, Typography, Stack, TextField, Divider, Button } from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import SaveAltIcon from "@mui/icons-material/SaveAlt";

/* ---------- types ---------- */

export type InpaintingStatusType = "idle" | "processing" | "success" | "error";

export interface InpaintingStatus {
    type: InpaintingStatusType;
    label: string;
    message: string;
}

export interface InpaintingPanelProps {
    apiUrl: string;
    setApiUrl: (url: string) => void;

    positivePrompt: string;
    setPositivePrompt: (value: string) => void;

    negativePrompt: string;
    setNegativePrompt: (value: string) => void;

    status: InpaintingStatus;

    // ✅ Option A: pass prompts into startInpaint
    startInpaint: (positivePrompt: string, negativePrompt: string) => void;

    openHint: () => void;
    saveHint: () => void;
}

/* ---------- component ---------- */

const InpaintingPanel: React.FC<InpaintingPanelProps> = ({
                                                             apiUrl,
                                                             setApiUrl,
                                                             positivePrompt,
                                                             setPositivePrompt,
                                                             negativePrompt,
                                                             setNegativePrompt,
                                                             status,
                                                             startInpaint,
                                                             openHint,
                                                             saveHint,
                                                         }) => {
    return (
        <Box sx={{ width: 360, p: 2 }}>
            <Typography variant="h6" sx={{ mb: 1 }}>
                Inpainting
            </Typography>

            <Stack spacing={1.5}>
                <TextField
                    label="Backend API URL"
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    helperText="Example: http://localhost:8000"
                    size="small"
                />

                <Divider />

                <TextField
                    label="Positive Prompt"
                    value={positivePrompt}
                    onChange={(e) => setPositivePrompt(e.target.value)}
                    placeholder="Describe what you want to appear..."
                    multiline
                    minRows={3}
                    size="small"
                />

                <TextField
                    label="Negative Prompt"
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                    placeholder="Describe what should NOT appear..."
                    multiline
                    minRows={2}
                    size="small"
                />

                <Divider />

                <Button
                    variant="contained"
                    startIcon={<PlayArrowIcon />}
                    onClick={() => startInpaint(positivePrompt, negativePrompt)}
                    disabled={status.type === "processing"}
                >
                    Start Inpainting
                </Button>

                <Button variant="outlined" startIcon={<OpenInNewIcon />} onClick={openHint}>
                    Open Image (Photopea)
                </Button>

                <Button variant="outlined" startIcon={<SaveAltIcon />} onClick={saveHint}>
                    Save (Photopea)
                </Button>

                <Divider />

                <Box
                    sx={{
                        p: 1,
                        borderRadius: 1,
                        bgcolor:
                            status.type === "idle"
                                ? "grey.100"
                                : status.type === "processing"
                                    ? "warning.light"
                                    : status.type === "success"
                                        ? "success.light"
                                        : "error.light",
                    }}
                >
                    <Typography variant="body2">
                        <b>Status:</b> {status.label}
                    </Typography>
                    <Typography variant="caption" sx={{ display: "block", mt: 0.5 }}>
                        {status.message}
                    </Typography>
                </Box>
            </Stack>
        </Box>
    );
};

export default InpaintingPanel;
