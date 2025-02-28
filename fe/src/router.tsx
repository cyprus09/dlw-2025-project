import { createBrowserRouter } from "react-router-dom";
import App from "./App";
import HelloWorld from "./pages/HelloWorld";

export const router = createBrowserRouter([
    {
        path: "/",
        element: <App />,
    },
    {
        path: "/hello",
        element: <HelloWorld />,
    },
]);
