import { BrowserRouter } from "react-router-dom";
import { CssBaseline } from "@mui/material";
import AppRoutes from "./AppRoutes";

export default function App() {
    return (
        <BrowserRouter>
            <CssBaseline />
            <AppRoutes />
        </BrowserRouter>
    );
}
