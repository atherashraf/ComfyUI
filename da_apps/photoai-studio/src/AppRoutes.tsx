import { Routes, Route } from "react-router-dom";
import PhotoAIStudioPage from "./pages/PhotoAIStudioPage";

export default function AppRoutes() {
    return (
        <Routes>
            <Route path="/" element={<PhotoAIStudioPage />} />
        </Routes>
    );
}
