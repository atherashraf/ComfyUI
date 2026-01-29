/**
 * AppSnackbar
 * ----------
 *
 * A reusable, programmatically-controlled Snackbar + Alert component based on MUI.
 *
 * Features:
 * - Show success, error, info, warning messages
 * - Set message, severity (color), and duration dynamically
 * - Controlled via exposed `show` method through `ref`
 *
 * Usage Example:
 *
 * import AppSnackbar, { DASnackbarHandle } from "@/components/base/AppSnackbar";
 *
 * const snackbarRef = useRef<DASnackbarHandle>(null);
 *
 * // Render
 * <AppSnackbar ref={snackbarRef} />
 *
 * // To show a message
 * snackbarRef.current?.show("Successfully saved!", "success");
 *
 * Parameters for `show`:
 * - message: string
 * - severity?: "success" | "info" | "warning" | "error" (default: "info")
 * - duration?: number in milliseconds (default: 3000 ms)
 */

import { forwardRef, useImperativeHandle, useState } from "react";
import { Snackbar, Alert, type AlertColor } from "@mui/material";

// Interface for exposing methods to parent
export interface AppSnackbarHandle {
    show: (message: string, severity?: AlertColor, duration?: number) => void;
}

// Main Component
const AppSnackbar = forwardRef<AppSnackbarHandle>((_, ref) => {
    const [open, setOpen] = useState(false);
    const [message, setMessage] = useState("");
    const [severity, setSeverity] = useState<AlertColor>("info");
    const [duration, setDuration] = useState(3000);

    useImperativeHandle(ref, () => ({
        show(msg: string, sev: AlertColor = "info", dur = 3000) {
            setMessage(msg);
            setSeverity(sev);
            setDuration(dur);
            setOpen(true);
        },
    }));

    const handleClose = () => setOpen(false);

    return (
        <Snackbar
            open={open}
            autoHideDuration={duration}
            onClose={handleClose}
            anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
        >
            <Alert onClose={handleClose} severity={severity} variant="filled">
                {message}
            </Alert>
        </Snackbar>
    );
});

export default AppSnackbar;
