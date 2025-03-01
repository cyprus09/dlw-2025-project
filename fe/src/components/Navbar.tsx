import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { Link, useLocation } from "react-router-dom";

export function Navbar() {
    const [activeTab, setActiveTab] = useState("home");
    const location = useLocation();

    // Update active tab based on current route
    useState(() => {
        const path = location.pathname;
        if (path === "/") {
            setActiveTab("home");
        } else if (path === "/ocr") {
            setActiveTab("ocr");
        } else if (path === "/structured-output") {
            setActiveTab("structured-output");
        } else if (path.includes("about")) {
            setActiveTab("about");
        } else if (path.includes("how")) {
            setActiveTab("how");
        } else if (path.includes("contact")) {
            setActiveTab("contact");
        }
    });

    return (
        <nav className="sticky top-0 z-50 backdrop-blur-lg bg-[#1E3A2F] border-b border-[#7aa56a]/20">
            <div className="container mx-auto px-4 py-4 flex justify-between items-center">
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5 }}
                    className="flex items-center space-x-2"
                >
                    <div className="w-10 h-10 rounded-full bg-[#7aa56a] flex items-center justify-center">
                        <span className="text-[#1E3A2F] font-bold text-xl">
                            TBD
                        </span>
                    </div>
                    <span className="font-bold text-xl text-white">
                        TrueCarbon
                    </span>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.1 }}
                    className="hidden md:flex space-x-8"
                >
                    <Link
                        to="/"
                        onClick={() => setActiveTab("home")}
                        className={`${
                            activeTab === "home"
                                ? "text-[#7aa56a] font-medium"
                                : "text-gray-300"
                        } hover:text-[#7aa56a] transition-colors`}
                    >
                        Home
                    </Link>
                    <Link
                        to="/"
                        onClick={() => setActiveTab("about")}
                        className={`${
                            activeTab === "about"
                                ? "text-[#7aa56a] font-medium"
                                : "text-gray-300"
                        } hover:text-[#7aa56a] transition-colors`}
                    >
                        About
                    </Link>
                    <Link
                        to="/"
                        onClick={() => setActiveTab("how")}
                        className={`${
                            activeTab === "how"
                                ? "text-[#7aa56a] font-medium"
                                : "text-gray-300"
                        } hover:text-[#7aa56a] transition-colors`}
                    >
                        How It Works
                    </Link>
                    <Link
                        to="/"
                        onClick={() => setActiveTab("contact")}
                        className={`${
                            activeTab === "contact"
                                ? "text-[#7aa56a] font-medium"
                                : "text-gray-300"
                        } hover:text-[#7aa56a] transition-colors`}
                    >
                        Contact
                    </Link>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5 }}
                >
                    <Button
                        variant="default"
                        size="default"
                        className="bg-[#7aa56a] hover:bg-[#7aa56a]/90 text-[#1E3A2F]"
                        onClick={() => (window.location.href = "/process")}
                    >
                        Get Started
                    </Button>
                </motion.div>
            </div>
        </nav>
    );
}
