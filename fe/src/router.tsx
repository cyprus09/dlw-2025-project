import { createBrowserRouter } from "react-router-dom";
import App from "./App";
import Process from "./pages/Process";
import Loading from "./pages/Loading";

export const router = createBrowserRouter([
    {
        path: "/",
        element: <App />,
    },
    {
        path: "/process",
        element: <Process />,
    },
    {
        path: "/loading",
        element: <Loading />,
    },
]);
