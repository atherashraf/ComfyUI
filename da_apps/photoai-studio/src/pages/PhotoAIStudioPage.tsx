// pages/PhotoStudioPage.tsx
import * as React from "react";
import {
    AppBar,
    Toolbar,
    IconButton,
    Typography,
    Drawer,
    Box,
    Button,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import ClearIcon from "@mui/icons-material/Clear";

import { usePhotopeaBridge } from "../hooks/usePhotopeaBridge";
import { useInpaintingState } from "../hooks/useInpaintingState";
import InpaintingPanel from "../components/modules/inpainting/InpaintingPanel";

const APPBAR_H = 60;
const DRAWER_W = 380;

export default function PhotoStudioPage() {
    const [drawerOpen, setDrawerOpen] = React.useState(false);

    const inpaintingState = useInpaintingState();

    const {
        iframeRef,
        status,
        apiUrl,
        setApiUrl,
        startInpaint,
        openHint,
        saveHint,
        isLoading,
    } = usePhotopeaBridge();

    const handleStartInpaint = () => {
        const res =
            inpaintingState.getSelectedCheckpointDetails() ||
            inpaintingState.getFirstCheckpointDetails();

        startInpaint(inpaintingState.positivePrompt, inpaintingState.negativePrompt, res);
    };

    const handleClearPrompts = () => {
        inpaintingState.clearPrompts();
    };

    return (
        <Box sx={{ width: "100vw", height: "100vh", overflow: "hidden" }}>
            <AppBar position="static" sx={{ height: `${APPBAR_H}px` }}>
                <Toolbar sx={{ minHeight: `${APPBAR_H}px` }}>
                    <IconButton
                        edge="start"
                        color="inherit"
                        onClick={() => setDrawerOpen(prev => !prev)}
                        sx={{ mr: 1 }}
                    >
                        <MenuIcon />
                    </IconButton>


                    <Typography variant="h6" sx={{ flex: 1 }}>
                        PhotoAIStudio
                    </Typography>

                    {inpaintingState.hasPrompts && (
                        <Button
                            color="inherit"
                            startIcon={<ClearIcon />}
                            onClick={handleClearPrompts}
                            size="small"
                            sx={{ mr: 2 }}
                        >
                            Clear Prompts
                        </Button>
                    )}

                    <Typography variant="body2" sx={{ opacity: 0.9 }}>
                        {status.label}
                    </Typography>
                </Toolbar>
            </AppBar>

            {/* Content row: Photopea + Drawer (drawer takes space, not overlay) */}
            <Box
                sx={{
                    display: "flex",
                    width: "100%",
                    height: `calc(100vh - ${APPBAR_H}px)`,
                    bgcolor: "#111",
                    overflow: "hidden",
                }}
            >
                {/* Photopea area (fills remaining space) */}
                <Box
                    sx={{
                        position: "relative",
                        flex: 1,
                        minWidth: 0, // ✅ crucial so the iframe can shrink when drawer opens
                        height: "100%",
                        overflow: "hidden",
                    }}
                >
                    {isLoading && (
                        <Box
                            sx={{
                                position: "absolute",
                                inset: 0,
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                bgcolor: "rgba(0,0,0,0.35)",
                                zIndex: 10,
                                color: "white",
                                fontSize: 14,
                            }}
                        >
                            Loading Photopea…
                        </Box>
                    )}

                    <iframe
                        ref={iframeRef}
                        title="Photopea"
                        src="https://www.photopea.com"
                        style={{
                            width: "100%",
                            height: "100%",
                            border: "none",
                            display: "block",
                        }}
                        allow="clipboard-read; clipboard-write"
                    />
                </Box>

                {/* Right drawer: width is 0 when closed, DRAWER_W when open */}
                <Drawer
                    anchor="right"
                    variant="persistent"
                    open={drawerOpen}
                    onClose={() => setDrawerOpen(false)}
                    sx={{
                        width: drawerOpen ? DRAWER_W : 0,
                        flexShrink: 0,
                        "& .MuiDrawer-paper": {
                            width: drawerOpen ? DRAWER_W : 0,
                            boxSizing: "border-box",
                            overflowX: "hidden",
                            position: "relative", // ✅ keeps it in the flex flow (prevents overlay feel)
                            transition: (theme) =>
                                theme.transitions.create("width", {
                                    easing: theme.transitions.easing.sharp,
                                    duration: theme.transitions.duration.enteringScreen,
                                }),
                        },
                    }}
                >
                    <InpaintingPanel
                        selectedCheckpoint={inpaintingState.selectedCheckpoint}
                        setSelectedCheckpoint={inpaintingState.setSelectedCheckpoint}
                        checkpoints={inpaintingState.checkpoints}
                        positivePrompt={inpaintingState.positivePrompt}
                        setPositivePrompt={inpaintingState.setPositivePrompt}
                        negativePrompt={inpaintingState.negativePrompt}
                        setNegativePrompt={inpaintingState.setNegativePrompt}
                        apiUrl={apiUrl}
                        setApiUrl={setApiUrl}
                        status={status}
                        startInpaint={handleStartInpaint}
                        openHint={openHint}
                        saveHint={saveHint}
                        onClearPrompts={handleClearPrompts}
                    />
                </Drawer>
            </Box>
        </Box>
    );
}
