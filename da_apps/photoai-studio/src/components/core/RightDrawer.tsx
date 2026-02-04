/**
 * RightDrawer
 * ---------------------------------------------------------------------
 * A resizable, slide-in drawer anchored to the **right** side of the screen.
 *
 * Features:
 * - Slide animation using MUI `Slide`
 * - Mounts/unmounts from DOM when hidden
 * - Mouse-driven horizontal resizing with min/max bounds
 * - Width persistence via `localStorage`
 * - Optional loading state with spinner
 * - Programmatic control through imperative methods
 * - Safe recovery if width becomes 0 (falls back to saved/default width)
 *
 * Resizing:
 * - Drag the left edge of the drawer to resize
 * - Width is clamped between `minWidth` and `maxWidth`
 * - Last width is restored from localStorage on mount
 *
 * Public API (via ref):
 * - openDrawer(width?)
 * - hideDrawer()
 * - closeDrawer()
 * - hideUnhideDrawer(durationMs, width?)
 * - setContent(heading, content, open?, width?)
 * - startLoading(heading?)
 * - stopLoading()
 * - setWidth(width)
 * - getWidth()
 * - isOpen()
 */

import * as React from "react";
import {
    AppBar, Box, IconButton, Toolbar, Typography, Paper, CircularProgress, Slide,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";

const LOCAL_STORAGE_KEY = "rightDrawerWidth";

interface RightDrawerProps {
    children?: React.ReactNode;
    appBarColor?: "primary" | "secondary" | "default" | "transparent" | "inherit";
    appBarHeight?: number;
    defaultWidth?: number;
    minWidth?: number;
    maxWidth?: number;
}

interface RightDrawerState {
    open: boolean;
    content: React.ReactNode;
    heading: string;
    isResizing: boolean;
    width: number;
    isLoading: boolean;
}

class RightDrawer extends React.PureComponent<RightDrawerProps, RightDrawerState> {
    static defaultProps: Partial<RightDrawerProps> = {
        appBarColor: "secondary", appBarHeight: 40, defaultWidth: 350, minWidth: 200, maxWidth: 1200,
    };

    private unhideTimer: number | null = null;

    constructor(props: RightDrawerProps) {
        super(props);

        const width = props.defaultWidth ?? 350;

        this.state = {
            open: false, content: null, heading: "", isResizing: false, width, isLoading: false,
        };
    }

    // ---------- persistence ----------
    private saveWidthToStorage = (width: number) => {
        try {
            localStorage.setItem(LOCAL_STORAGE_KEY, width.toString());
        } catch (err) {
            console.error("Error saving rightDrawer width:", err);
        }
    };

    private clampWidth = (w: number) => {
        const minW = this.props.minWidth ?? 200;
        const maxW = this.props.maxWidth ?? 1200;
        return Math.max(minW, Math.min(maxW, w));
    };

    private getFallbackWidth = () => {
        const defaultW = this.props.defaultWidth ?? 350;

        try {
            const saved = localStorage.getItem(LOCAL_STORAGE_KEY);
            const parsed = saved ? parseInt(saved, 10) : NaN;
            if (!Number.isNaN(parsed)) return this.clampWidth(parsed);
        } catch {
            /* ignore */
        }

        return this.clampWidth(defaultW);
    };

    private clearUnhideTimer = () => {
        if (this.unhideTimer != null) {
            window.clearTimeout(this.unhideTimer);
            this.unhideTimer = null;
        }
    };

    // ---------- public API ----------
    openDrawer = (width?: number) => {
    this.setState((prev) => {
    const baseWidth = prev.width > 0 ? prev.width : this.getFallbackWidth();
    return {
    open: true, width: width != null ? this.clampWidth(width) : baseWidth,
};
});
};

/** Hide but keep content & width */
hideDrawer = () => {
    this.clearUnhideTimer();
    this.setState({open: false});
};

/**
 * Hide for a duration (ms) and then unhide.
 * If called again, it resets the previous timer.
 * Optionally pass a width to apply on unhide.
 */
hideUnhideDrawer = (durationMs: number, width?: number) => {
    const d = Math.max(0, durationMs || 0);
    this.hideDrawer();

    this.clearUnhideTimer();
    this.unhideTimer = window.setTimeout(() => {
        this.openDrawer(width);
        this.unhideTimer = null;
    }, d);
};

/** Close and clear everything (keeps last width) */
closeDrawer = () => {
    this.clearUnhideTimer();
    this.setState({
        open: false, heading: "", content: null, isLoading: false,
    });
};

setContent = (heading: string, content: React.ReactNode, open: boolean = true, width?: number) => {
    this.setState((prev) => ({
        heading, content, isLoading: false, open, width: width != null ? this.clampWidth(width) : prev.width,
    }));
};

startLoading = (heading?: string) => {
    this.clearUnhideTimer();
    this.setState((prev) => ({
        isLoading: true,
        heading: heading ?? prev.heading,
        open: true,
        width: prev.width > 0 ? prev.width : this.getFallbackWidth(),
    }));
};

stopLoading = () => this.setState({isLoading: false});

setWidth = (width: number) => {
    const clamped = this.clampWidth(width);
    this.setState({width: clamped});
    this.saveWidthToStorage(clamped);
};

getWidth = () => this.state.width;

isOpen = () => this.state.open;

// ---------- resize handlers ----------
handleMouseDown = () => {
    document.body.style.cursor = "ew-resize";
    (document.body.style as any).userSelect = "none";
    this.setState({isResizing: true});
};

handleMouseMove = (e: MouseEvent) => {
    if (!this.state.isResizing) return;

    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth;

    // Right drawer: width is distance from cursor to right edge
    const next = this.clampWidth(viewportWidth - e.clientX);

    if (next !== this.state.width) {
        this.setState({width: next});
    }
};

handleMouseUp = () => {
    if (this.state.isResizing) {
        this.saveWidthToStorage(this.state.width);
    }
    document.body.style.cursor = "";
    (document.body.style as any).userSelect = "";
    this.setState({isResizing: false});
};

// ---------- lifecycle ----------
componentDidMount() {
    // Init width from localStorage; ensure we never start at 0.
    try {
        const saved = localStorage.getItem(LOCAL_STORAGE_KEY);
        if (saved) {
            const parsed = parseInt(saved, 10);
            if (!Number.isNaN(parsed)) {
                const clamped = this.clampWidth(parsed);
                this.setState({width: clamped > 0 ? clamped : this.getFallbackWidth()});
            }
        } else {
            this.setState((prev) => ({
                width: prev.width > 0 ? prev.width : this.getFallbackWidth(),
            }));
        }
    } catch {
        this.setState((prev) => ({
            width: prev.width > 0 ? prev.width : this.getFallbackWidth(),
        }));
    }

    document.addEventListener("mousemove", this.handleMouseMove);
    document.addEventListener("mouseup", this.handleMouseUp);
}

componentWillUnmount() {
    this.clearUnhideTimer();
    document.removeEventListener("mousemove", this.handleMouseMove);
    document.removeEventListener("mouseup", this.handleMouseUp);
}

private emitLayout = () => {
    window.dispatchEvent(new CustomEvent("drawerLayout", {
        detail: {
            side: "right", open: this.state.open, size: this.state.open ? this.state.width : 0,
        },
    }));
};

componentDidUpdate(_: any, prev: RightDrawerState) {
    if (prev.open !== this.state.open || prev.width !== this.state.width) {
        this.emitLayout();
    }
}


render() {
    const {open, heading, content, width, isLoading} = this.state;
    const {children, appBarColor = "secondary", appBarHeight = 40, minWidth} = this.props;

    const minW = minWidth ?? 200;

    return (<>
        {!open && content && (<Box
            onClick={() => this.openDrawer()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    this.openDrawer();
                }
            }}
            sx={{
                position: "fixed",
                top: "50%",
                height: "200px",
                right: 0,
                transform: "translateY(-50%)",
                bgcolor: "#1976d2",
                color: "white",
                px: 1,
                py: 0.5,
                borderRadius: "4px 0 0 4px",
                cursor: "pointer",
                zIndex: 1400,
                writingMode: "vertical-rl",
                textAlign: "center",
                userSelect: "none",
                fontSize: 12,
            }}
        >
            Tap to Expand
        </Box>)}

        {/*<Slide direction="left" in={open} mountOnEnter unmountOnExit>*/}
        <Slide direction="left" in={open}>

            <Box
                sx={{
                    position: "fixed",
                    top: 0,
                    right: 0,
                    width: `${width}px`,
                    minWidth: `${minW}px`,
                    height: "100%",
                    zIndex: 1300,
                    display: "flex",
                    flexDirection: "column",
                    bgcolor: "background.paper",
                    boxShadow: 4,
                    borderLeft: "1px solid #ccc",
                }}
            >
                {/* Left-edge internal resize handle */}
                <Box
                    onMouseDown={this.handleMouseDown}
                    sx={{
                        position: "absolute",
                        left: -3,
                        top: 0,
                        bottom: 0,
                        width: 8,
                        cursor: "ew-resize",
                        bgcolor: "#e0e0e0",
                        zIndex: 2,
                        pointerEvents: open ? "auto" : "none",
                        "&:hover": {bgcolor: "#d5d5d5"},
                        "&:active": {bgcolor: "#cfcfcf"},
                    }}
                />

                <AppBar position="static" color={appBarColor}
                        sx={{height: appBarHeight, justifyContent: "center"}}>
                    <Toolbar variant="dense" sx={{minHeight: `${appBarHeight}px !important`, px: 1}}>
                        <Typography variant="h6" sx={{flexGrow: 1, fontSize: 16}}>
                            {heading}
                        </Typography>

                        <IconButton size="small" onClick={this.hideDrawer} sx={{color: "white"}}>
                            <VisibilityOffIcon fontSize="small"/>
                        </IconButton>

                        <IconButton size="small" onClick={this.closeDrawer} sx={{color: "white"}}>
                            <CloseIcon fontSize="small"/>
                        </IconButton>
                    </Toolbar>
                </AppBar>

                <Paper
                    sx={{
                        display: "flex",
                        flexDirection: "column",
                        width: "100%",
                        height: "100%",
                        overflowY: "auto",
                        boxShadow: "-4px 0px 12px rgba(0,0,0,0.2)",
                        borderTopLeftRadius: 8,
                        borderBottomLeftRadius: 8,
                        position: "relative",
                        zIndex: 1,
                    }}
                    elevation={0}
                >
                    <Box sx={{flexGrow: 1, p: 0.5, display: "flex", justifyContent: "center"}}>
                        {isLoading ? (
                            <CircularProgress size={40} thickness={4}/>) : content ? (content) : (children)}
                    </Box>
                </Paper>
            </Box>
        </Slide>
    </>);
}
}

export default RightDrawer;
export type RightDrawerHandle = typeof RightDrawer.prototype;
