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
    Menu,
    MenuItem,
} from "@mui/material";

import MenuIcon from "@mui/icons-material/Menu";
import ClearIcon from "@mui/icons-material/Clear";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";
import CloseIcon from "@mui/icons-material/Close";


import {usePhotopeaBridge} from "../hooks/usePhotopeaBridge";
import {useInpaintingState} from "../hooks/useInpaintingState";
import InpaintingPanel from "../components/modules/inpainting/InpaintingPanel";


const APPBAR_H = 60;
const DRAWER_W = 380;

export default function PhotoStudioPage() {
    const [drawerOpen, setDrawerOpen] = React.useState(false);

    // AppBar dropdown menu state
    const [actionAnchor, setActionAnchor] = React.useState<null | HTMLElement>(null);
    const actionOpen = Boolean(actionAnchor);

    const openActionMenu = (e: React.MouseEvent<HTMLElement>) => setActionAnchor(e.currentTarget);
    const closeActionMenu = () => setActionAnchor(null);

    const inpaintingState = useInpaintingState();

    const {
        iframeRef,
        status,
        apiUrl,
        setApiUrl,
        startInpaint,
        copyActiveLayerToClipboard,
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
        <Box sx={{width: "100vw", height: "100vh", overflow: "hidden"}}>
            {/* AppBar ALWAYS ABOVE drawer */}
            <AppBar
                position="fixed"
                sx={(theme) => ({
                    height: `${APPBAR_H}px`,
                    zIndex: theme.zIndex.drawer + 2,
                })}
            >
                <Toolbar sx={{minHeight: `${APPBAR_H}px`}}>
                    {/* Dropdown button (Menu icon) */}
                    <IconButton
                        edge="start"
                        color="inherit"
                        onClick={openActionMenu}
                        sx={{mr: 1}}
                        aria-controls={actionOpen ? "appbar-actions" : undefined}
                        aria-haspopup="true"
                        aria-expanded={actionOpen ? "true" : undefined}
                    >
                        <MenuIcon/>
                    </IconButton>

                    {/* Dropdown menu */}
                    <Menu
                        id="appbar-actions"
                        anchorEl={actionAnchor}
                        open={actionOpen}
                        onClose={closeActionMenu}
                        anchorOrigin={{vertical: "bottom", horizontal: "left"}}
                        transformOrigin={{vertical: "top", horizontal: "left"}}
                    >
                        <MenuItem
                            onClick={() => {
                                closeActionMenu();
                                setDrawerOpen((p) => !p); // ✅ toggle works always
                            }}
                        >
                            <AutoFixHighIcon fontSize="small" style={{marginRight: 8}}/>
                            Inpainting Panel
                        </MenuItem>

                        <MenuItem
                            onClick={async () => {
                                closeActionMenu();
                                try {
                                    await copyActiveLayerToClipboard();
                                } catch (e) {
                                    console.error(e);
                                    alert(e instanceof Error ? e.message : String(e));
                                }
                            }}
                        >
                            <ContentCopyIcon fontSize="small" style={{marginRight: 8}}/>
                            Copy Active Layer
                        </MenuItem>
                    </Menu>

                    <Typography variant="h6" sx={{flex: 1}}>
                        PhotoAIStudio
                    </Typography>

                    {inpaintingState.hasPrompts && (
                        <Button
                            color="inherit"
                            startIcon={<ClearIcon/>}
                            onClick={handleClearPrompts}
                            size="small"
                            sx={{mr: 2}}
                        >
                            Clear Prompts
                        </Button>
                    )}

                    <Typography variant="body2" sx={{opacity: 0.9}}>
                        {status.label}
                    </Typography>
                </Toolbar>
            </AppBar>

            {/* push content below fixed AppBar */}
            <Box sx={{height: `${APPBAR_H}px`}}/>

            {/* Content row: Photopea + Drawer */}
            <Box
                sx={{
                    display: "flex",
                    width: "100%",
                    height: `calc(100vh - ${APPBAR_H}px)`,
                    bgcolor: "#111",
                    overflow: "hidden",
                }}
            >
                {/* Photopea area */}
                <Box
                    sx={{
                        position: "relative",
                        flex: 1,
                        minWidth: 0,
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

                {/* Right drawer - constrained below AppBar */}
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
                            position: "relative",
                            height: "100%",
                            display: "flex",
                            flexDirection: "column",
                        },
                    }}
                >
                    {/* Drawer header */}
                    <Box
                        sx={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            px: 1.5,
                            py: 1,
                            borderBottom: "1px solid rgba(255,255,255,0.12)",
                        }}
                    >
                        <Typography variant="subtitle1">Inpainting</Typography>
                        <IconButton
                            size="small"
                            onClick={() => setDrawerOpen(false)}
                            aria-label="Close panel"
                        >
                            <CloseIcon fontSize="small"/>
                        </IconButton>
                    </Box>

                    {/* Drawer content */}
                    <Box sx={{flex: 1, overflow: "auto"}}>
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
                    </Box>
                </Drawer>

            </Box>
        </Box>
    );
}
