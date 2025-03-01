import { motion, useScroll, useTransform } from "framer-motion";
import { Button } from "@/components/ui/button";
import "./App.css";
import { Navbar } from "./components/Navbar";

function App() {
    const { scrollYProgress } = useScroll();
    const opacity = useTransform(scrollYProgress, [0, 0.2], [1, 0]);
    const scale = useTransform(scrollYProgress, [0, 0.2], [1, 0.9]);

    return (
        <div className="min-h-screen bg-[#1E3A2F] text-white">
            <div className="absolute inset-0 bg-gradient-to-b from-[#7aa56a]/10 to-transparent" />

            {/* Navigation Bar */}
            <Navbar />

            {/* Hero Section */}
            <section className="py-20 md:py-32 container mx-auto px-4 relative overflow-hidden">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.7 }}
                        className="space-y-6 relative"
                    >
                        <motion.div
                            initial={{ scale: 0.9 }}
                            animate={{ scale: 1 }}
                            transition={{ duration: 0.5 }}
                            className="inline-flex items-center rounded-full px-4 py-1 text-sm bg-[#7aa56a]/20 text-white mb-4"
                        >
                            <span className="mr-2">🌍</span>
                            Sustainable Future Starts Here
                        </motion.div>
                        <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold leading-tight">
                            Verify Carbon Credits with{" "}
                            <motion.span
                                className="text-[#7aa56a] inline-block"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.3, duration: 0.5 }}
                            >
                                Transparency
                            </motion.span>
                        </h1>
                        <p className="text-lg text-gray-300">
                            Our AI-powered platform detects discrepancies
                            between claimed and actual waste management,
                            ensuring companies only earn carbon credits for
                            genuine environmental impact.
                        </p>
                        <div className="flex flex-col sm:flex-row gap-4">
                            <motion.div
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                <Button
                                    size="lg"
                                    className="px-8 bg-[#7aa56a] hover:bg-[#7aa56a]/90 text-[#1E3A2F]"
                                    onClick={() =>
                                        (window.location.href = "/process")
                                    }
                                >
                                    <span className="mr-2">🌟</span>
                                    Verify Claims
                                </Button>
                            </motion.div>
                            <motion.div
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                            >
                                <Button
                                    size="lg"
                                    variant="outline"
                                    className="px-8 border-[#7aa56a] text-[#7aa56a] hover:bg-[#7aa56a]/10 bg-transparent hover:text-[#7aa56a]"
                                    onClick={() =>
                                        (window.location.href = "/process")
                                    }
                                >
                                    <span className="mr-2">📚</span>
                                    Learn More
                                </Button>
                            </motion.div>
                        </div>
                    </motion.div>

                    <motion.div
                        style={{ scale, opacity }}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.7, delay: 0.2 }}
                        className="relative"
                    >
                        <div className="aspect-square scale-75 rounded-2xl bg-gradient-to-br from-[#7aa56a]/20 to-[#7aa56a]/30 flex items-center justify-center overflow-hidden backdrop-blur-lg border border-[#7aa56a]/20">
                            <motion.div
                                className="absolute w-full h-full bg-[radial-gradient(circle_at_center,rgba(157,200,141,0.1),transparent_70%)]"
                                animate={{
                                    rotate: 360,
                                    scale: [1, 1.1, 1],
                                }}
                                transition={{
                                    duration: 20,
                                    repeat: Infinity,
                                    repeatType: "reverse",
                                }}
                            />
                            <div className="relative w-3/5 h-3/5 rounded-xl bg-[#1E3A2F]/60 p-6 shadow-lg backdrop-blur-sm border border-[#7aa56a]/20">
                                <div className="space-y-4">
                                    <motion.div
                                        className="w-full h-40 rounded-lg bg-gradient-to-r from-[#7aa56a]/20 to-[#7aa56a]/30"
                                        animate={{
                                            opacity: [0.5, 0.8, 0.5],
                                        }}
                                        transition={{
                                            duration: 2,
                                            repeat: Infinity,
                                        }}
                                    />
                                    <div className="space-y-2">
                                        <motion.div
                                            className="w-3/4 h-6 rounded bg-[#7aa56a]/20"
                                            animate={{
                                                opacity: [0.5, 0.8, 0.5],
                                            }}
                                            transition={{
                                                duration: 2,
                                                delay: 0.2,
                                                repeat: Infinity,
                                            }}
                                        />
                                        <motion.div
                                            className="w-1/3 h-6 rounded bg-[#7aa56a]/20"
                                            animate={{
                                                opacity: [0.5, 0.8, 0.5],
                                            }}
                                            transition={{
                                                duration: 2,
                                                delay: 0.2,
                                                repeat: Infinity,
                                            }}
                                        />
                                        <motion.div
                                            className="w-1/2 h-6 rounded bg-[#7aa56a]/20"
                                            animate={{
                                                opacity: [0.5, 0.8, 0.5],
                                            }}
                                            transition={{
                                                duration: 2,
                                                delay: 0.4,
                                                repeat: Infinity,
                                            }}
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* Features Section */}
            <section className="py-20 bg-[#1E3A2F]/80 relative overflow-hidden">
                <motion.div
                    className="absolute -left-20 top-0 w-40 h-40 bg-[#7aa56a]/10 rounded-full blur-3xl"
                    animate={{
                        scale: [1, 1.2, 1],
                        opacity: [0.3, 0.5, 0.3],
                    }}
                    transition={{
                        duration: 5,
                        repeat: Infinity,
                    }}
                />
                <motion.div
                    className="absolute -right-20 bottom-0 w-40 h-40 bg-[#7aa56a]/10 rounded-full blur-3xl"
                    animate={{
                        scale: [1.2, 1, 1.2],
                        opacity: [0.3, 0.5, 0.3],
                    }}
                    transition={{
                        duration: 5,
                        repeat: Infinity,
                    }}
                />

                <div className="container mx-auto px-4">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.7 }}
                        viewport={{ once: true }}
                        className="text-center mb-16 max-w-3xl mx-auto"
                    >
                        <h2 className="text-3xl md:text-4xl font-bold mb-4 text-[#7aa56a]">
                            How Our Platform Works
                        </h2>
                        <p className="text-gray-300 text-lg">
                            Our solution addresses the global challenge of waste
                            management inefficiencies by promoting
                            accountability and transparency among companies.
                        </p>
                    </motion.div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        {[
                            {
                                title: "Input Claims",
                                description:
                                    "Companies input their waste management claims and coordinates into our platform.",
                                icon: "📁",
                                delay: 0,
                            },
                            {
                                title: "AI Analysis",
                                description:
                                    "Our AI models analyze satellite imagery and data to verify the claims.",
                                icon: "🤖",
                                delay: 0.2,
                            },
                            {
                                title: "Verification Report",
                                description:
                                    "Get detailed reports on the accuracy of waste management claims.",
                                icon: "📊",
                                delay: 0.4,
                            },
                        ].map((feature) => (
                            <motion.div
                                key={feature.title}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                transition={{
                                    duration: 0.5,
                                    delay: feature.delay,
                                }}
                                viewport={{ once: true }}
                                className="relative p-6 rounded-xl bg-gradient-to-br from-[#7aa56a]/10 to-[#7aa56a]/5 backdrop-blur-sm border border-[#7aa56a]/20"
                            >
                                <div className="text-4xl mb-4">
                                    {feature.icon}
                                </div>
                                <h3 className="text-xl font-semibold mb-2 text-[#7aa56a]">
                                    {feature.title}
                                </h3>
                                <p className="text-gray-300">
                                    {feature.description}
                                </p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Process Flow Section */}
            <section className="py-20 container mx-auto px-4">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.7 }}
                    viewport={{ once: true }}
                    className="text-center mb-16 max-w-3xl mx-auto"
                >
                    <h2 className="text-3xl md:text-4xl font-bold mb-4 text-[#7aa56a]">
                        Verification Process
                    </h2>
                    <p className="text-gray-300 text-lg">
                        Our platform uses a systematic approach to verify carbon
                        credit claims and ensure transparency.
                    </p>
                </motion.div>

                <div className="flex flex-col gap-4 max-w-3xl mx-auto">
                    {[
                        {
                            title: "1. Submit Verification Request",
                            description:
                                "Companies submit the claims they wish to verify, including waste amount produced or plants planted to offset waste.",
                        },
                        {
                            title: "2. Quantify Carbon Credits",
                            description:
                                "Our platform converts inputs into a quantifiable measure using standardized carbon credit metrics.",
                        },
                        {
                            title: "3. Satellite Imagery Analysis",
                            description:
                                "Using the provided coordinates, we analyze satellite imagery to verify the actual waste management activities.",
                        },
                        {
                            title: "4. Compare & Rate",
                            description:
                                "We compare the claimed carbon credits with our calculated values to determine a fraud rating on a normalized scale.",
                        },
                    ].map((step, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 10 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5 }}
                            viewport={{ once: true }}
                            className="bg-[#1E3A2F]/60 rounded-lg p-4 shadow-sm border border-[#7aa56a]/20"
                        >
                            <div className="flex items-center gap-8">
                                <div>
                                    <h3 className="text-lg font-semibold text-[#7aa56a]">
                                        {step.title}
                                    </h3>
                                    <p className="text-gray-300">
                                        {step.description}
                                    </p>
                                </div>
                            </div>
                        </motion.div>
                    ))}
                </div>
            </section>

            {/* CTA Section */}
            <section className="py-20 bg-[#1E3A2F]/80">
                <div className="container mx-auto px-4">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.7 }}
                        viewport={{ once: true }}
                        className="max-w-4xl mx-auto text-center bg-[#1E3A2F]/60 rounded-2xl p-8 md:p-12 shadow-lg border border-[#7aa56a]/20"
                    >
                        <h2 className="text-3xl md:text-4xl font-bold mb-4 text-[#7aa56a]">
                            Ready to Verify Carbon Claims?
                        </h2>
                        <p className="text-lg text-gray-300 mb-8 max-w-2xl mx-auto">
                            Join the movement towards transparency and
                            accountability in waste management and carbon credit
                            claims.
                        </p>
                        <div className="flex flex-col sm:flex-row gap-4 justify-center">
                            <Button
                                size="lg"
                                className="px-8 bg-[#7aa56a] hover:bg-[#7aa56a]/90 text-[#1E3A2F]"
                            >
                                Get Started Now
                            </Button>
                            <Button
                                size="lg"
                                variant="outline"
                                className="px-8 border-[#7aa56a] text-[#7aa56a] hover:bg-[#7aa56a]/10"
                            >
                                Request Demo
                            </Button>
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* Footer */}
            <footer className="py-12 bg-[#1E3A2F]/80 border-t border-[#7aa56a]/20">
                <div className="container mx-auto px-4">
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
                        <div>
                            <div className="flex items-center space-x-2 mb-4">
                                <div className="w-8 h-8 rounded-full bg-[#7aa56a] flex items-center justify-center">
                                    <span className="text-[#1E3A2F] font-bold text-sm">
                                        T
                                    </span>
                                </div>
                                <span className="font-bold text-[#7aa56a]">
                                    TBD.
                                </span>
                            </div>
                            <p className="text-gray-300 text-sm">
                                Promoting accountability and transparency in
                                waste management and carbon credit claims.
                            </p>
                        </div>

                        <div>
                            <h4 className="font-semibold mb-4 text-[#7aa56a]">
                                Platform
                            </h4>
                            <ul className="space-y-2 text-sm text-gray-300">
                                <li>
                                    <a
                                        href="#"
                                        className="hover:text-[#7aa56a] transition-colors"
                                    >
                                        How it Works
                                    </a>
                                </li>
                                <li>
                                    <a
                                        href="#"
                                        className="hover:text-[#7aa56a] transition-colors"
                                    >
                                        Features
                                    </a>
                                </li>
                                <li>
                                    <a
                                        href="#"
                                        className="hover:text-[#7aa56a] transition-colors"
                                    >
                                        Pricing
                                    </a>
                                </li>
                                <li>
                                    <a
                                        href="#"
                                        className="hover:text-[#7aa56a] transition-colors"
                                    >
                                        FAQ
                                    </a>
                                </li>
                            </ul>
                        </div>

                        <div>
                            <h4 className="font-semibold mb-4 text-[#7aa56a]">
                                Company
                            </h4>
                            <ul className="space-y-2 text-sm text-gray-300">
                                <li>
                                    <a
                                        href="#"
                                        className="hover:text-[#7aa56a] transition-colors"
                                    >
                                        About Us
                                    </a>
                                </li>
                                <li>
                                    <a
                                        href="#"
                                        className="hover:text-[#7aa56a] transition-colors"
                                    >
                                        Blog
                                    </a>
                                </li>
                                <li>
                                    <a
                                        href="#"
                                        className="hover:text-[#7aa56a] transition-colors"
                                    >
                                        Careers
                                    </a>
                                </li>
                                <li>
                                    <a
                                        href="#"
                                        className="hover:text-[#7aa56a] transition-colors"
                                    >
                                        Contact
                                    </a>
                                </li>
                            </ul>
                        </div>

                        <div>
                            <h4 className="font-semibold mb-4 text-[#7aa56a]">
                                Legal
                            </h4>
                            <ul className="space-y-2 text-sm text-gray-300">
                                <li>
                                    <a
                                        href="#"
                                        className="hover:text-[#7aa56a] transition-colors"
                                    >
                                        Terms of Service
                                    </a>
                                </li>
                                <li>
                                    <a
                                        href="#"
                                        className="hover:text-[#7aa56a] transition-colors"
                                    >
                                        Privacy Policy
                                    </a>
                                </li>
                                <li>
                                    <a
                                        href="#"
                                        className="hover:text-[#7aa56a] transition-colors"
                                    >
                                        Cookie Policy
                                    </a>
                                </li>
                            </ul>
                        </div>
                    </div>

                    <div className="mt-12 pt-8 border-t border-[#7aa56a]/20 text-center text-sm text-gray-300">
                        <p>
                            &copy; {new Date().getFullYear()} TBD.. All rights
                            reserved.
                        </p>
                    </div>
                </div>
            </footer>
        </div>
    );
}

export default App;
