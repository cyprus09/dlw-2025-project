import {
    Calculator,
    CheckCircle,
    Lightbulb,
    MapPin,
    Notebook,
    StarIcon,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Navbar } from "../components/Navbar";
import { Button } from "../components/ui/button";
import { API_ENDPOINTS } from "../config/api";
import { FileUpload } from "@/components/FileUpload";
import { FilePreview } from "@/components/FilePreview";
import Loading from "./Loading";
import CarbonInsights from "./CarbonInsights";

interface StructuredAnalysisResponse {
    filename: string;
    ocr_text: string;
    structured_response: {
        project_title: string;
        doc_version_or_issue_date: string;
        summary_description: string;
        location: {
            country: string;
            provinces: string[];
        };
        claimed_reductions: {
            total_claimed_tCO2e: number;
            average_annual_tCO2e: number;
        };
        key_activities: {
            name: string;
            description: string;
        }[];
        thinkingSpace: {
            observations: string[];
            possible_inconsistencies: string[];
            suggestions_for_further_verification: string[];
            final_thoughts: string;
        };
    };
    usage: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
    };
}

export default function Process() {
    const [file, setFile] = useState<File | null>(null);
    const [structuredData, setStructuredData] = useState<
        StructuredAnalysisResponse["structured_response"] | null
    >(null);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadError, setUploadError] = useState<string | null>(null);
    const [uploadSuccess, setUploadSuccess] = useState(false);

    const [isVerifying, setIsVerifying] = useState(false);
    const [verificationStatus, setVerificationStatus] = useState<
        "idle" | "success" | "error"
    >("idle");

    const handleRemoveFile = () => {
        setFile(null);
        setStructuredData(null);
        setUploadSuccess(false);
        setUploadError(null);
    };

    const uploadFile = async (fileToUpload: File) => {
        setIsUploading(true);
        setUploadError("");
        setUploadSuccess(false);
        setStructuredData(null);

        try {
            // Create a FormData instance
            const formData = new FormData();
            formData.append("file", fileToUpload);

            // Determine which endpoint to use based on file type
            let endpoint;

            if (fileToUpload.type === "application/pdf") {
                // Use PDF-specific endpoint
                endpoint = API_ENDPOINTS.ocr.analyze;
                formData.append("file_type", "pdf");
            } else if (fileToUpload.type.startsWith("image/")) {
                // Use image-specific endpoint
                endpoint = API_ENDPOINTS.ocr.analyze;
                formData.append("file_type", "image");
            } else {
                throw new Error(
                    "Unsupported file type. Please upload a PDF or image file."
                );
            }

            const response = await fetch(endpoint, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(
                    errorData.detail || "Failed to process document"
                );
            }

            const data: StructuredAnalysisResponse = await response.json();

            setStructuredData(data.structured_response);
            setUploadSuccess(true);
        } catch (error) {
            console.error("Error processing document:", error);
            setUploadError(
                error instanceof Error
                    ? error.message
                    : "An unknown error occurred"
            );
        } finally {
            setIsUploading(false);
        }
    };

    const verifyFile = async () => {
        setIsVerifying(true);
        setVerificationStatus("idle");

        try {
            // Mock verification delay
            await new Promise((resolve) => setTimeout(resolve, 5000));
            setVerificationStatus("success");
        } catch (error) {
            setVerificationStatus("error");
            console.error("Verification error:", error);
        } finally {
            setIsVerifying(false);
        }
    };

    // Reset states when changing files
    useEffect(() => {
        if (file) {
            setUploadSuccess(false);
        }
    }, [file]);

    if (isVerifying) return <Loading />;
    if (verificationStatus === "success") return <CarbonInsights />;

    return (
        <div className="flex flex-col h-screen">
            <Navbar />
            <div className="h-[calc(100vh-50px)] p-5 overflow-hidden">
                <div className="h-full w-full">
                    {!file ? (
                        <FileUpload
                            onFileSelect={setFile}
                            onUploadFile={uploadFile}
                        />
                    ) : (
                        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 h-full">
                            {/* Left Column - File Display */}
                            <div className="rounded-lg border bg-white p-6 shadow-sm flex flex-col h-full">
                                <FilePreview
                                    file={file}
                                    onFileRemove={handleRemoveFile}
                                />
                            </div>

                            {/* Right Column - Processing Status */}
                            <div className="rounded-lg flex flex-col h-full overflow-y-scroll scrollbar-hide">
                                {isUploading ? (
                                    <div className="flex flex-1 flex-col items-center justify-center">
                                        <div className="mb-4 h-12 w-12 animate-spin rounded-full border-4 border-gray-200 border-t-emerald-500"></div>
                                        <p className="mb-1 text-center text-lg font-medium text-gray-700">
                                            Processing...
                                        </p>
                                        <p className="text-center text-sm text-gray-500">
                                            This may take a moment...
                                        </p>
                                    </div>
                                ) : uploadError ? (
                                    <div className="flex flex-1 flex-col items-center justify-center rounded-lg bg-red-50 p-6 text-center">
                                        <h3 className="mb-2 text-lg font-medium text-red-800">
                                            Processing Error
                                        </h3>
                                        <p className="text-sm text-red-600">
                                            {uploadError}
                                        </p>
                                        <Button
                                            onClick={() =>
                                                uploadFile(file as File)
                                            }
                                            className="mt-4"
                                            variant="outline"
                                        >
                                            Try Again
                                        </Button>
                                    </div>
                                ) : uploadSuccess && structuredData ? (
                                    <>
                                        <div className="flex flex-1 flex-col h-[calc(100%-3rem)] overflow-y-scroll scrollbar-hide">
                                            {structuredData && (
                                                <div className="mb-4 rounded-lg">
                                                    <div className="space-y-6">
                                                        {/* Project Header */}
                                                        <div className="bg-white rounded-lg p-5 shadow border border-gray-100">
                                                            <h2 className="text-2xl font-bold mb-2">
                                                                {
                                                                    structuredData.project_title
                                                                }
                                                            </h2>
                                                            <div className="flex items-center">
                                                                <span className="text-sm text-gray-600">
                                                                    Document
                                                                    Version:{" "}
                                                                    {
                                                                        structuredData.doc_version_or_issue_date
                                                                    }
                                                                </span>
                                                            </div>
                                                        </div>

                                                        {/* Summary Section */}
                                                        <div className="bg-white rounded-lg p-5 shadow border border-gray-100">
                                                            <h3 className="text-lg font-semibold text-gray-800 mb-2 flex items-center">
                                                                <Notebook className="h-5 w-5 mr-2 text-green-600" />
                                                                Summary
                                                            </h3>
                                                            <p className="text-gray-600">
                                                                {
                                                                    structuredData.summary_description
                                                                }
                                                            </p>
                                                        </div>

                                                        {/* Two Column Layout for Location and Emissions */}
                                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                                            {/* Location Section */}
                                                            <div className="bg-white rounded-lg p-5 shadow border border-gray-100">
                                                                <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                                                                    <MapPin className="h-5 w-5 mr-2 text-green-600" />
                                                                    Location
                                                                </h3>
                                                                <div className="space-y-2">
                                                                    <div className="flex items-center">
                                                                        <span className="font-medium text-gray-700 w-24">
                                                                            Country:
                                                                        </span>
                                                                        <span className="text-gray-600">
                                                                            {
                                                                                structuredData
                                                                                    .location
                                                                                    .country
                                                                            }
                                                                        </span>
                                                                    </div>
                                                                    <div>
                                                                        <span className="font-medium text-gray-700">
                                                                            Provinces:
                                                                        </span>
                                                                        <div className="mt-1 flex flex-wrap gap-1">
                                                                            {structuredData.location.provinces.map(
                                                                                (
                                                                                    province,
                                                                                    index
                                                                                ) => (
                                                                                    <span
                                                                                        key={
                                                                                            index
                                                                                        }
                                                                                        className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full"
                                                                                    >
                                                                                        {
                                                                                            province
                                                                                        }
                                                                                    </span>
                                                                                )
                                                                            )}
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                            </div>

                                                            {/* Emissions Reduction Section */}
                                                            <div className="bg-white rounded-lg p-5 shadow border border-gray-100">
                                                                <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                                                                    <Calculator className="h-5 w-5 mr-2 text-green-600" />
                                                                    Claimed
                                                                    Reductions
                                                                </h3>
                                                                <div className="space-y-3">
                                                                    <div className="flex flex-col">
                                                                        <span className="font-medium text-gray-700">
                                                                            Total
                                                                            Claimed
                                                                            CO₂
                                                                            Equivalent:
                                                                        </span>
                                                                        <div className="mt-1">
                                                                            <span className="text-2xl font-bold text-green-700">
                                                                                {structuredData.claimed_reductions.total_claimed_tCO2e.toLocaleString()}
                                                                            </span>
                                                                            <span className="ml-1 text-sm text-gray-500">
                                                                                tCO₂e
                                                                            </span>
                                                                        </div>
                                                                    </div>
                                                                    <div className="flex flex-col">
                                                                        <span className="font-medium text-gray-700">
                                                                            Average
                                                                            Annual
                                                                            Reduction:
                                                                        </span>
                                                                        <div className="mt-1">
                                                                            <span className="text-xl font-bold text-green-700">
                                                                                {structuredData.claimed_reductions.average_annual_tCO2e.toLocaleString()}
                                                                            </span>
                                                                            <span className="ml-1 text-sm text-gray-500">
                                                                                tCO₂e/year
                                                                            </span>
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </div>

                                                        {/* Key Activities Section */}
                                                        <div className="bg-white rounded-lg p-5 shadow border border-gray-100">
                                                            <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                                                                <StarIcon className="h-5 w-5 mr-2 text-green-600" />
                                                                Key Activities
                                                            </h3>
                                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                                                {structuredData.key_activities.map(
                                                                    (
                                                                        activity,
                                                                        index
                                                                    ) => (
                                                                        <div
                                                                            key={
                                                                                index
                                                                            }
                                                                            className="bg-green-50 rounded-lg p-3 border border-green-100"
                                                                        >
                                                                            <h4 className="font-semibold text-green-800">
                                                                                {
                                                                                    activity.name
                                                                                }
                                                                            </h4>
                                                                            <p className="text-sm text-gray-600 mt-1">
                                                                                {
                                                                                    activity.description
                                                                                }
                                                                            </p>
                                                                        </div>
                                                                    )
                                                                )}
                                                            </div>
                                                        </div>

                                                        {/* Thinking Space Section */}
                                                        <div className="bg-white rounded-lg p-5 shadow border border-gray-100">
                                                            <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                                                                <Lightbulb className="h-5 w-5 mr-2 text-green-600" />
                                                                Analysis &
                                                                Observations
                                                            </h3>
                                                            <div className="space-y-3">
                                                                <div>
                                                                    <h4 className="font-medium text-gray-700">
                                                                        Observations:
                                                                    </h4>
                                                                    <ul className="list-disc pl-5 mt-1 space-y-1 text-gray-600">
                                                                        {structuredData.thinkingSpace.observations.map(
                                                                            (
                                                                                obs,
                                                                                index
                                                                            ) => (
                                                                                <li
                                                                                    key={
                                                                                        index
                                                                                    }
                                                                                >
                                                                                    {
                                                                                        obs
                                                                                    }
                                                                                </li>
                                                                            )
                                                                        )}
                                                                    </ul>
                                                                </div>
                                                                <div>
                                                                    <h4 className="font-medium text-gray-700">
                                                                        Suggestions
                                                                        for
                                                                        Further
                                                                        Verification:
                                                                    </h4>
                                                                    <ul className="list-disc pl-5 mt-1 space-y-1 text-gray-600">
                                                                        {structuredData.thinkingSpace.suggestions_for_further_verification.map(
                                                                            (
                                                                                sug,
                                                                                index
                                                                            ) => (
                                                                                <li
                                                                                    key={
                                                                                        index
                                                                                    }
                                                                                >
                                                                                    {
                                                                                        sug
                                                                                    }
                                                                                </li>
                                                                            )
                                                                        )}
                                                                    </ul>
                                                                </div>
                                                                <div>
                                                                    <h4 className="font-medium text-gray-700">
                                                                        Final
                                                                        Thoughts:
                                                                    </h4>
                                                                    <p className="text-gray-600 mt-1">
                                                                        {
                                                                            structuredData
                                                                                .thinkingSpace
                                                                                .final_thoughts
                                                                        }
                                                                    </p>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                        <Button
                                            className="w-full text-md"
                                            size={"lg"}
                                            onClick={verifyFile}
                                        >
                                            <CheckCircle className="h-5 w-5 text-white" />
                                            <span className="ml-2">
                                                Verify Report
                                            </span>
                                        </Button>
                                    </>
                                ) : null}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
