// components/modules/inpainting/InpaintingPanel.tsx
import * as React from "react";
import {
    Box,
    Typography,
    Stack,
    TextField,
    Divider,
    Button,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Chip,
    Tooltip
} from "@mui/material";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import SaveAltIcon from "@mui/icons-material/SaveAlt";
import ModelTrainingIcon from "@mui/icons-material/ModelTraining";
import ClearIcon from "@mui/icons-material/Clear";
import InfoIcon from "@mui/icons-material/Info";
import StarIcon from "@mui/icons-material/Star";
import {DEFAULT_COMFY_URL} from "@/hooks/usePhotopeaBridge.ts";

/* ---------- types ---------- */

export type InpaintingStatusType = "idle" | "processing" | "success" | "error";

export interface InpaintingStatus {
    type: InpaintingStatusType;
    label: string;
    message: string;
}

export interface CheckpointOption {
    id: string;
    name: string;
    file: string;
    size: string;
    recommended: boolean;
    resolution: string;
    description?: string;
}

export interface InpaintingPanelProps {
    // Checkpoint props
    selectedCheckpoint: string;
    setSelectedCheckpoint: (checkpoint: string) => void;
    checkpoints: CheckpointOption[];

    // Prompt props
    positivePrompt: string;
    setPositivePrompt: (value: string) => void;
    negativePrompt: string;
    setNegativePrompt: (value: string) => void;

    // Photopea bridge props
    apiUrl: string;
    setApiUrl: (url: string) => void;
    status: InpaintingStatus;

    // Actions
    startInpaint: (positivePrompt: string, negativePrompt: string) => void;
    openHint: () => void;
    saveHint: () => void;

    // Optional
    onClearPrompts?: () => void;
}

/* ---------- component ---------- */

const InpaintingPanel: React.FC<InpaintingPanelProps> = ({
                                                             // Checkpoint props
                                                             selectedCheckpoint,
                                                             setSelectedCheckpoint,
                                                             checkpoints,

                                                             // Prompt props
                                                             positivePrompt,
                                                             setPositivePrompt,
                                                             negativePrompt,
                                                             setNegativePrompt,

                                                             // Photopea bridge props
                                                             apiUrl,
                                                             setApiUrl,
                                                             status,

                                                             // Actions
                                                             startInpaint,
                                                             openHint,
                                                             saveHint,

                                                             // Optional
                                                             onClearPrompts,
                                                         }) => {
    // Get selected checkpoint details
    const selectedCheckpointDetails = checkpoints.find(cp => cp.id === selectedCheckpoint);

    // Sort checkpoints: recommended first, then by name
    const sortedCheckpoints = [...checkpoints].sort((a, b) => {
        if (a.recommended && !b.recommended) return -1;
        if (!a.recommended && b.recommended) return 1;
        return a.name.localeCompare(b.name);
    });

    return (
        <Box sx={{ width: 380, p: 2, height: '100vh', overflowY: 'auto' }}>
            <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                <ModelTrainingIcon sx={{ mr: 1 }} />
                Inpainting Studio
            </Typography>

            <Stack spacing={2}>
                {/* API URL */}
                <TextField
                    label="Backend API URL"
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    helperText={`Example: ${DEFAULT_COMFY_URL}`}
                    size="small"
                    fullWidth
                />

                {/* Checkpoint Dropdown */}
                <FormControl fullWidth size="small">
                    <InputLabel id="checkpoint-select-label">
                        Checkpoint Model
                    </InputLabel>
                    <Select
                        labelId="checkpoint-select-label"
                        value={selectedCheckpoint}
                        label="Checkpoint Model"
                        onChange={(e) => setSelectedCheckpoint(e.target.value)}
                    >
                        {sortedCheckpoints.map((checkpoint) => (
                            <MenuItem key={checkpoint.id} value={checkpoint.id}>
                                <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%', py: 0.5 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                            <ModelTrainingIcon sx={{ fontSize: 16, mr: 1, color: 'action.active' }} />
                                            <Typography variant="body2" sx={{ fontWeight: checkpoint.recommended ? 600 : 400 }}>
                                                {checkpoint.name}
                                            </Typography>
                                        </Box>
                                        {checkpoint.recommended && (
                                            <Chip
                                                icon={<StarIcon sx={{ fontSize: 14 }} />}
                                                label="Recommended"
                                                size="small"
                                                color="success"
                                                sx={{ height: 20, fontSize: '0.7rem', '& .MuiChip-icon': { ml: 0.5 } }}
                                            />
                                        )}
                                    </Box>
                                    <Typography variant="caption" color="text.secondary" sx={{ ml: 3 }}>
                                        {checkpoint.file} • {checkpoint.size} • {checkpoint.resolution}
                                    </Typography>
                                    {checkpoint.description && (
                                        <Typography variant="caption" color="text.secondary" sx={{ ml: 3, fontStyle: 'italic' }}>
                                            {checkpoint.description}
                                        </Typography>
                                    )}
                                </Box>
                            </MenuItem>
                        ))}
                    </Select>
                </FormControl>

                {/* Selected Model Info */}
                {selectedCheckpointDetails && (
                    <Box sx={{
                        p: 1.5,
                        borderRadius: 1,
                        bgcolor: 'info.light',
                        border: '1px solid',
                        borderColor: selectedCheckpointDetails.recommended ? 'success.main' : 'info.main',
                        position: 'relative'
                    }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                            <Box>
                                <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 0.5 }}>
                                    {selectedCheckpointDetails.recommended && (
                                        <StarIcon sx={{ fontSize: 12, mr: 0.5, color: 'success.main', verticalAlign: 'middle' }} />
                                    )}
                                    Selected Model: {selectedCheckpointDetails.name}
                                </Typography>
                                <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
                                    File: {selectedCheckpointDetails.file}
                                </Typography>
                                <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
                                    Size: {selectedCheckpointDetails.size} • Max Resolution: {selectedCheckpointDetails.resolution}
                                </Typography>
                                {selectedCheckpointDetails.description && (
                                    <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', mt: 0.5, fontStyle: 'italic' }}>
                                        {selectedCheckpointDetails.description}
                                    </Typography>
                                )}
                            </Box>
                            {selectedCheckpointDetails.recommended && (
                                <Tooltip title="Best for inpainting tasks">
                                    <InfoIcon sx={{ fontSize: 16, color: 'success.main' }} />
                                </Tooltip>
                            )}
                        </Box>
                    </Box>
                )}

                <Divider />

                {/* Prompts Section */}
                <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="subtitle2">Prompts</Typography>
                        <Button
                            size="small"
                            startIcon={<ClearIcon />}
                            onClick={onClearPrompts}
                            disabled={!positivePrompt && !negativePrompt}
                        >
                            Clear
                        </Button>
                    </Box>

                    <TextField
                        label="Positive Prompt"
                        value={positivePrompt}
                        onChange={(e) => setPositivePrompt(e.target.value)}
                        placeholder="Describe what you want to appear..."
                        multiline
                        minRows={3}
                        size="small"
                        fullWidth
                        sx={{ mb: 2 }}
                    />

                    <TextField
                        label="Negative Prompt"
                        value={negativePrompt}
                        onChange={(e) => setNegativePrompt(e.target.value)}
                        placeholder="Describe what should NOT appear..."
                        multiline
                        minRows={2}
                        size="small"
                        fullWidth
                    />
                </Box>

                <Divider />

                {/* Action Buttons */}
                <Stack spacing={1}>
                    <Button
                        variant="contained"
                        color="primary"
                        startIcon={<PlayArrowIcon />}
                        onClick={() => startInpaint(positivePrompt, negativePrompt)}
                        disabled={status.type === "processing" || !positivePrompt.trim()}
                        fullWidth
                        size="medium"
                    >
                        {status.type === "processing" ? "Processing..." : "Start Inpainting"}
                    </Button>

                    <Stack direction="row" spacing={1}>
                        <Button
                            variant="outlined"
                            startIcon={<OpenInNewIcon />}
                            onClick={openHint}
                            fullWidth
                            size="small"
                        >
                            Open in Photopea
                        </Button>
                        <Button
                            variant="outlined"
                            startIcon={<SaveAltIcon />}
                            onClick={saveHint}
                            fullWidth
                            size="small"
                        >
                            Save from Photopea
                        </Button>
                    </Stack>
                </Stack>

                <Divider />

                {/* Status Display */}
                <Box
                    sx={{
                        p: 1.5,
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
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                        <Typography variant="body2" sx={{ fontWeight: 'bold', flex: 1 }}>
                            Status: {status.label}
                        </Typography>
                        {status.type === "processing" && (
                            <Box sx={{
                                width: 8,
                                height: 8,
                                borderRadius: '50%',
                                bgcolor: 'warning.main',
                                animation: 'pulse 1.5s infinite',
                                '@keyframes pulse': {
                                    '0%': { opacity: 1 },
                                    '50%': { opacity: 0.5 },
                                    '100%': { opacity: 1 }
                                }
                            }} />
                        )}
                    </Box>
                    <Typography variant="caption" sx={{ display: 'block' }}>
                        {status.message}
                    </Typography>
                </Box>

                {/* Quick Tips */}
                <Box sx={{
                    p: 1.5,
                    borderRadius: 1,
                    bgcolor: 'grey.50',
                    border: '1px dashed',
                    borderColor: 'grey.300'
                }}>
                    <Typography variant="caption" sx={{ fontWeight: 'bold', display: 'block', mb: 0.5 }}>
                        💡 Quick Tips:
                    </Typography>
                    <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
                        • Use recommended checkpoints for best inpainting results
                    </Typography>
                    <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
                        • Be specific in your positive prompts
                    </Typography>
                    <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary' }}>
                        • Use negative prompts to remove unwanted elements
                    </Typography>
                </Box>
            </Stack>
        </Box>
    );
};

export default InpaintingPanel;