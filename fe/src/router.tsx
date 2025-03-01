import { createBrowserRouter } from "react-router-dom";
import App from "./App";
import Ocr from "./pages/Ocr";

export const router = createBrowserRouter([
    {
        path: "/",
        element: <App />,
    },
    {
        path: "/ocr",
        element: <Ocr />,
    },
]);
