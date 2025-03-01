import { createBrowserRouter } from "react-router-dom";
import App from "./App";
import Ocr from "./pages/Ocr";
import StructuredOutput from "./pages/StructuredOutput";

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
]);
