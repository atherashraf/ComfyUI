import * as React from "react";
import { AppBar, Toolbar, IconButton, Typography, Drawer, Box } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";

import { usePhotopeaBridge } from "../hooks/usePhotopeaBridge";
import InpaintingPanel from "../components/modules/inpainting/InpaintingPanel";

const APPBAR_H = 60;

export default function PhotoStudioPage() {
    const [drawerOpen, setDrawerOpen] = React.useState(false);

    const [positivePrompt, setPositivePrompt] = React.useState<string>("");
    const [negativePrompt, setNegativePrompt] = React.useState<string>("");

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

    return (
        <Box sx={{ width: "100vw", height: "100vh", overflow: "hidden" }}>
            <AppBar position="static" sx={{ height: `${APPBAR_H}px` }}>
                <Toolbar sx={{ minHeight: `${APPBAR_H}px` }}>
                    <IconButton
                        edge="start"
                        color="inherit"
                        onClick={() => setDrawerOpen(true)}
                        sx={{ mr: 1 }}
                    >
                        <MenuIcon />
                    </IconButton>

                    <Typography variant="h6" sx={{ flex: 1 }}>
                        PhotoAIStudio
                    </Typography>

                    <Typography variant="body2" sx={{ opacity: 0.9 }}>
                        {status.label}
                    </Typography>
                </Toolbar>
            </AppBar>

            <Box
                sx={{
                    width: "100vw",
                    height: `calc(100vh - ${APPBAR_H}px)`,
                    position: "relative",
                    bgcolor: "#111",
                }}
            >
                {/* Loading overlay */}
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

            <Drawer anchor="right" open={drawerOpen} onClose={() => setDrawerOpen(false)}>
                <InpaintingPanel
                    apiUrl={apiUrl}
                    setApiUrl={setApiUrl}
                    positivePrompt={positivePrompt}
                    setPositivePrompt={setPositivePrompt}
                    negativePrompt={negativePrompt}
                    setNegativePrompt={setNegativePrompt}
                    status={status}
                    startInpaint={startInpaint}
                    openHint={openHint}
                    saveHint={saveHint}
                />
            </Drawer>
        </Box>
    );
}
