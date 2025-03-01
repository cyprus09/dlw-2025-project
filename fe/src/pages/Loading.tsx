import { Navbar } from "@/components/Navbar";
import { DotLottieReact } from "@lottiefiles/dotlottie-react";
import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState } from "react";

const carbonFraudFacts = [
    "Up to 40% of carbon offset projects may have questionable environmental benefits.",
    "Some carbon credit schemes have been found to overestimate their impact by up to 400%.",
    "Ghost credits: Some projects claim reductions for forests that were never at risk.",
    "Double counting remains a major issue in carbon credit markets.",
    "Many offset projects fail to meet the 'additionality' requirement.",
];

const Loading = () => {
    const [currentFactIndex, setCurrentFactIndex] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentFactIndex((prev) => (prev + 1) % carbonFraudFacts.length);
        }, 5000);

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="h-screen overflow-hidden">
            <Navbar />
            <div className="h-[calc(100vh-64px)] flex flex-col items-center justify-center">
                <DotLottieReact
                    src="https://lottie.host/69d02061-45c1-4bcb-a8d7-6ead33357758/gesZ3ByQ5m.lottie"
                    loop
                    autoplay
                    speed={0.8}
                    className="h-96 -mt-10"
                />

                <motion.div
                    initial={{ y: 20 }}
                    animate={{ y: 0 }}
                    className="w-full max-w-lg"
                >
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={currentFactIndex}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -20 }}
                            className="text-center text-gray-700 text-md font-medium"
                        >
                            {carbonFraudFacts[currentFactIndex]}
                        </motion.div>
                    </AnimatePresence>
                </motion.div>
            </div>
        </div>
    );
};

export default Loading;
