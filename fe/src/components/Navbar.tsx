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
        <nav className="sticky top-0 z-50 backdrop-blur-lg bg-background/80 border-b border-border">
            <div className="container mx-auto px-4 py-4 flex justify-between items-center">
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5 }}
                    className="flex items-center space-x-2"
                >
                    <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center">
                        <span className="text-primary-foreground font-bold text-xl">
                            CV
                        </span>
                    </div>
                    <span className="font-bold text-xl">CarbonVerify</span>
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
                                ? "text-primary font-medium"
                                : "text-muted-foreground"
                        } hover:text-primary transition-colors`}
                    >
                        Home
                    </Link>
                    <Link
                        to="/about"
                        onClick={() => setActiveTab("about")}
                        className={`${
                            activeTab === "about"
                                ? "text-primary font-medium"
                                : "text-muted-foreground"
                        } hover:text-primary transition-colors`}
                    >
                        About
                    </Link>
                    <Link
                        to="/ocr"
                        onClick={() => setActiveTab("ocr")}
                        className={`${
                            activeTab === "ocr"
                                ? "text-primary font-medium"
                                : "text-muted-foreground"
                        } hover:text-primary transition-colors`}
                    >
                        OCR
                    </Link>
                    <Link
                        to="/structured-output"
                        onClick={() => setActiveTab("structured-output")}
                        className={`${
                            activeTab === "structured-output"
                                ? "text-primary font-medium"
                                : "text-muted-foreground"
                        } hover:text-primary transition-colors`}
                    >
                        Structured Output
                    </Link>
                    <Link
                        to="/how-it-works"
                        onClick={() => setActiveTab("how")}
                        className={`${
                            activeTab === "how"
                                ? "text-primary font-medium"
                                : "text-muted-foreground"
                        } hover:text-primary transition-colors`}
                    >
                        How It Works
                    </Link>
                    <Link
                        to="/contact"
                        onClick={() => setActiveTab("contact")}
                        className={`${
                            activeTab === "contact"
                                ? "text-primary font-medium"
                                : "text-muted-foreground"
                        } hover:text-primary transition-colors`}
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
                        className="bg-primary hover:bg-primary/90 text-primary-foreground"
                    >
                        Get Started
                    </Button>
                </motion.div>
            </div>
        </nav>
    );
}
