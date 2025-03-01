import { createBrowserRouter } from "react-router-dom";
import App from "./App";
import CarbonInsights from "./pages/CarbonInsights";
import Loading from "./pages/Loading";
import Process from "./pages/Process";

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
        path: "/loading-output",
        element: <Loading />,
    },
    {
        path: "/insights",
        element: <CarbonInsights />,
    },
]);
