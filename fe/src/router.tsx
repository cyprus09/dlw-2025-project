import { createBrowserRouter } from "react-router-dom";
import App from "./App";
import Ocr from "./pages/Ocr";
import StructuredOutput from "./pages/StructuredOutput";
import CarbonInsights from "./pages/CarbonInsights";

export const router = createBrowserRouter([
    {
        path: "/",
        element: <App />,
    },
    {
        path: "/ocr",
        element: <Ocr />,
    },
    {
        path: "/structured-output",
        element: <StructuredOutput />,
    },
    {
        path: "/insights",
        element: <CarbonInsights />,
    },
]);
