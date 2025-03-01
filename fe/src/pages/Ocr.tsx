import { useState, useEffect } from "react";
import { Button } from "../components/ui/button";
import { API_ENDPOINTS } from "../config/api";
import { Navbar } from "../components/Navbar";

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
            total_area_ha: number;
            forest_area_ha: number;
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

export default function Ocr() {
    const [file, setFile] = useState<File | null>(null);
    const [extractedText, setExtractedText] = useState<string>("");
    const [structuredData, setStructuredData] = useState<
        StructuredAnalysisResponse["structured_response"] | null
    >(null);
    const [usageStats, setUsageStats] = useState<
        StructuredAnalysisResponse["usage"] | null
    >(null);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadError, setUploadError] = useState<string | null>(null);
    const [dragActive, setDragActive] = useState(false);
    const [uploadSuccess, setUploadSuccess] = useState(false);
    const [processingStage, setProcessingStage] = useState<string>("");

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const selectedFile = e.target.files[0];
            setFile(selectedFile);
            uploadFile(selectedFile);
        }
    };

    const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(true);
    };

    const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
    };

    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const droppedFile = e.dataTransfer.files[0];
            setFile(droppedFile);
            uploadFile(droppedFile);
        }
    };

    const uploadFile = async (fileToUpload: File) => {
        setIsUploading(true);
        setUploadError("");
        setUploadSuccess(false);
        setProcessingStage("Uploading file...");
        setStructuredData(null);
        setUsageStats(null);

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
                setProcessingStage("Processing PDF document...");
            } else if (fileToUpload.type.startsWith("image/")) {
                // Use image-specific endpoint
                endpoint = API_ENDPOINTS.ocr.analyze;
                formData.append("file_type", "image");
                setProcessingStage("Processing image...");
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

            setProcessingStage("Extracting text and analyzing content...");
            setExtractedText(data.ocr_text);
            setStructuredData(data.structured_response);
            setUsageStats(data.usage);
            setUploadSuccess(true);
            setProcessingStage("Analysis complete");
        } catch (error) {
            console.error("Error processing document:", error);
            setUploadError(
                error instanceof Error
                    ? error.message
                    : "An unknown error occurred"
            );
            setProcessingStage("Error occurred");
        } finally {
            setIsUploading(false);
        }
    };

    const handleRemoveFile = () => {
        setFile(null);
        setExtractedText("");
        setStructuredData(null);
        setUsageStats(null);
        setUploadSuccess(false);
        setUploadError(null);
        setProcessingStage("");
    };

    // Reset states when changing files
    useEffect(() => {
        if (file) {
            setUploadSuccess(false);
            setExtractedText("");
        }
    }, [file]);

    return (
        <div className="flex flex-col min-h-screen">
            <Navbar />
            <div className="flex-grow p-5">
                <div className="h-full w-full">
                    {!file ? (
                        <div className="flex flex-col items-center justify-center h-full">
                            <div className="mb-8 text-center">
                                <h1 className="mb-2 text-3xl font-bold text-gray-800">
                                    Upload a Project Report
                                </h1>
                                <p className="text-gray-600">
                                    Upload an image or PDF to analyze it
                                </p>
                            </div>
                            <div
                                className={`relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-12 transition-all w-7/12 ${
                                    dragActive
                                        ? "border-emerald-500 bg-emerald-50"
                                        : "border-gray-300 bg-gray-50 hover:bg-gray-100"
                                }`}
                                onDragOver={handleDragOver}
                                onDragLeave={handleDragLeave}
                                onDrop={handleDrop}
                            >
                                <div className="mb-4 text-center">
                                    <svg
                                        xmlns="http://www.w3.org/2000/svg"
                                        className="mx-auto h-12 w-12 text-gray-400"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                        stroke="currentColor"
                                    >
                                        <path
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            strokeWidth={2}
                                            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                                        />
                                    </svg>
                                    <p className="mt-2 text-sm font-medium text-gray-700">
                                        Drag & drop your file here, or
                                    </p>
                                </div>
                                <input
                                    id="file-upload"
                                    type="file"
                                    accept="image/*,application/pdf"
                                    onChange={handleFileChange}
                                    className="hidden"
                                />
                                <label htmlFor="file-upload">
                                    <Button
                                        type="button"
                                        variant="default"
                                        size="lg"
                                        onClick={() =>
                                            document
                                                .getElementById("file-upload")
                                                ?.click()
                                        }
                                        className="cursor-pointer"
                                    >
                                        Browse Files
                                    </Button>
                                </label>
                                <p className="mt-2 text-xs text-gray-500">
                                    Supported formats: JPG, PNG, PDF
                                </p>
                            </div>
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 h-full">
                            {/* Left Column - File Display */}
                            <div className="rounded-lg border bg-white p-6 shadow-sm flex flex-col h-full">
                                <div className="mb-4 flex items-center justify-between">
                                    <div className="flex items-center space-x-3">
                                        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-emerald-100">
                                            {file.type.startsWith("image/") ? (
                                                <svg
                                                    xmlns="http://www.w3.org/2000/svg"
                                                    className="h-6 w-6 text-emerald-600"
                                                    fill="none"
                                                    viewBox="0 0 24 24"
                                                    stroke="currentColor"
                                                >
                                                    <path
                                                        strokeLinecap="round"
                                                        strokeLinejoin="round"
                                                        strokeWidth={2}
                                                        d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                                                    />
                                                </svg>
                                            ) : (
                                                <svg
                                                    xmlns="http://www.w3.org/2000/svg"
                                                    className="h-6 w-6 text-emerald-600"
                                                    fill="none"
                                                    viewBox="0 0 24 24"
                                                    stroke="currentColor"
                                                >
                                                    <path
                                                        strokeLinecap="round"
                                                        strokeLinejoin="round"
                                                        strokeWidth={2}
                                                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                                                    />
                                                </svg>
                                            )}
                                        </div>
                                        <div>
                                            <p className="font-medium text-gray-800">
                                                {file.name}
                                            </p>
                                            <p className="text-xs text-gray-500">
                                                {(file.size / 1024).toFixed(2)}{" "}
                                                KB •{" "}
                                                {file.type || "Unknown type"}
                                            </p>
                                        </div>
                                    </div>
                                    <button
                                        onClick={handleRemoveFile}
                                        className="rounded-full p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-500"
                                    >
                                        <svg
                                            xmlns="http://www.w3.org/2000/svg"
                                            className="h-5 w-5"
                                            viewBox="0 0 20 20"
                                            fill="currentColor"
                                        >
                                            <path
                                                fillRule="evenodd"
                                                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                                            />
                                        </svg>
                                    </button>
                                </div>

                                <div className="mb-4 rounded-lg bg-gray-50 p-4 flex-1">
                                    {file.type.startsWith("image/") ? (
                                        <div className="flex justify-center">
                                            <img
                                                src={URL.createObjectURL(file)}
                                                alt="Preview"
                                                className="rounded-md object-contain"
                                            />
                                        </div>
                                    ) : file.type === "application/pdf" ? (
                                        <div className={"flex flex-col h-full"}>
                                            <iframe
                                                src={URL.createObjectURL(file)}
                                                title="PDF Viewer"
                                                className={`w-full rounded-t-md border-0 bg-white shadow-sm flex-1`}
                                            />
                                        </div>
                                    ) : (
                                        <div className="flex flex-col items-center justify-center rounded-md border-2 border-dashed border-gray-300 bg-gray-100 p-4">
                                            <svg
                                                xmlns="http://www.w3.org/2000/svg"
                                                className="mb-2 h-10 w-10 text-gray-400"
                                                fill="none"
                                                viewBox="0 0 24 24"
                                                stroke="currentColor"
                                            >
                                                <path
                                                    strokeLinecap="round"
                                                    strokeLinejoin="round"
                                                    strokeWidth={2}
                                                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                                                />
                                            </svg>
                                            <p className="text-center text-sm text-gray-600">
                                                File preview not available
                                            </p>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Right Column - Processing Status */}
                            <div className="rounded-lg border bg-white p-6 shadow-sm flex flex-col h-full overflow-y-scroll">
                                <div className="mb-4 flex items-center justify-between">
                                    <h2 className="text-xl font-semibold text-gray-800">
                                        {uploadSuccess
                                            ? "Project Analysis"
                                            : "Processing Status"}
                                    </h2>
                                    {uploadSuccess && (
                                        <div className="flex items-center">
                                            <span className="mr-2 inline-flex h-2 w-2 rounded-full bg-green-500"></span>
                                            <span className="text-sm font-medium text-green-600">
                                                Completed
                                            </span>
                                        </div>
                                    )}
                                </div>

                                {isUploading ? (
                                    <div className="flex flex-1 flex-col items-center justify-center">
                                        <div className="mb-4 h-12 w-12 animate-spin rounded-full border-4 border-gray-200 border-t-emerald-500"></div>
                                        <p className="mb-1 text-center text-lg font-medium text-gray-700">
                                            {processingStage}
                                        </p>
                                        <p className="text-center text-sm text-gray-500">
                                            This may take a moment...
                                        </p>
                                    </div>
                                ) : uploadError ? (
                                    <div className="flex flex-1 flex-col items-center justify-center rounded-lg bg-red-50 p-6 text-center">
                                        <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-red-100">
                                            <svg
                                                className="h-8 w-8 text-red-600"
                                                xmlns="http://www.w3.org/2000/svg"
                                                viewBox="0 0 24 24"
                                                fill="none"
                                                stroke="currentColor"
                                                strokeWidth="2"
                                                strokeLinecap="round"
                                                strokeLinejoin="round"
                                            >
                                                <circle
                                                    cx="12"
                                                    cy="12"
                                                    r="10"
                                                ></circle>
                                                <line
                                                    x1="15"
                                                    y1="9"
                                                    x2="9"
                                                    y2="15"
                                                ></line>
                                                <line
                                                    x1="9"
                                                    y1="9"
                                                    x2="15"
                                                    y2="15"
                                                ></line>
                                            </svg>
                                        </div>
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
                                ) : uploadSuccess ? (
                                    <div className="flex flex-1 flex-col">
                                        {structuredData && (
                                            <div className="mb-4 rounded-lg bg-whitep-4">
                                                <h3 className="text-lg font-semibold text-gray-800 mb-3">
                                                    Project Details
                                                </h3>
                                                <div className="overflow-auto">
                                                    <div className="space-y-6">
                                                        {/* Project Header */}
                                                        <div className="bg-gradient-to-r from-green-600 to-green-800 rounded-lg p-5 text-white shadow-md">
                                                            <h2 className="text-2xl font-bold mb-2">
                                                                {
                                                                    structuredData.project_title
                                                                }
                                                            </h2>
                                                            <div className="flex items-center text-green-100">
                                                                <span className="text-sm">
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
                                                                <svg
                                                                    xmlns="http://www.w3.org/2000/svg"
                                                                    className="h-5 w-5 mr-2 text-green-600"
                                                                    viewBox="0 0 20 20"
                                                                    fill="currentColor"
                                                                >
                                                                    <path
                                                                        fillRule="evenodd"
                                                                        d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z"
                                                                    />
                                                                </svg>
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
                                                                    <svg
                                                                        xmlns="http://www.w3.org/2000/svg"
                                                                        className="h-5 w-5 mr-2 text-green-600"
                                                                        viewBox="0 0 20 20"
                                                                        fill="currentColor"
                                                                    >
                                                                        <path
                                                                            fillRule="evenodd"
                                                                            d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z"
                                                                        />
                                                                    </svg>
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
                                                                    <div className="flex items-center">
                                                                        <span className="font-medium text-gray-700 w-24">
                                                                            Total
                                                                            Area:
                                                                        </span>
                                                                        <span className="text-gray-600">
                                                                            {structuredData.location.total_area_ha.toLocaleString()}{" "}
                                                                            hectares
                                                                        </span>
                                                                    </div>
                                                                    <div className="flex items-center">
                                                                        <span className="font-medium text-gray-700 w-24">
                                                                            Forest
                                                                            Area:
                                                                        </span>
                                                                        <span className="text-gray-600">
                                                                            {structuredData.location.forest_area_ha.toLocaleString()}{" "}
                                                                            hectares
                                                                        </span>
                                                                    </div>
                                                                </div>
                                                            </div>

                                                            {/* Emissions Reduction Section */}
                                                            <div className="bg-white rounded-lg p-5 shadow border border-gray-100">
                                                                <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                                                                    <svg
                                                                        xmlns="http://www.w3.org/2000/svg"
                                                                        className="h-5 w-5 mr-2 text-green-600"
                                                                        viewBox="0 0 20 20"
                                                                        fill="currentColor"
                                                                    >
                                                                        <path
                                                                            fillRule="evenodd"
                                                                            d="M5 2a1 1 0 011 1v1h1a1 1 0 010 2H6v1a1 1 0 01-2 0V6H3a1 1 0 010-2h1V3a1 1 0 011-1zm0 10a1 1 0 011 1v1h1a1 1 0 110 2H6v1a1 1 0 11-2 0v-1H3a1 1 0 110-2h1v-1a1 1 0 011-1zm7-10a1 1 0 01.707.293l.707.707L15.414 5a1 1 0 11-1.414 1.414L13 5.414l-.707.707a1 1 0 01-1.414-1.414l.707-.707 1.414-1.414A1 1 0 0112 2zm-1 10a1 1 0 01.707.293l.707.707 1.414-1.414a1 1 0 111.414 1.414l-1.414 1.414-.707.707a1 1 0 01-1.414-1.414l.707-.707 1.414-1.414A1 1 0 0111 12z"
                                                                        />
                                                                    </svg>
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
                                                                <svg
                                                                    xmlns="http://www.w3.org/2000/svg"
                                                                    className="h-5 w-5 mr-2 text-green-600"
                                                                    viewBox="0 0 20 20"
                                                                    fill="currentColor"
                                                                >
                                                                    <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
                                                                    <path
                                                                        fillRule="evenodd"
                                                                        d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z"
                                                                    />
                                                                </svg>
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
                                                                <svg
                                                                    xmlns="http://www.w3.org/2000/svg"
                                                                    className="h-5 w-5 mr-2 text-green-600"
                                                                    viewBox="0 0 20 20"
                                                                    fill="currentColor"
                                                                >
                                                                    <path
                                                                        fillRule="evenodd"
                                                                        d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z"
                                                                    />
                                                                </svg>
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
                                                                        Possible
                                                                        Inconsistencies:
                                                                    </h4>
                                                                    <ul className="list-disc pl-5 mt-1 space-y-1 text-gray-600">
                                                                        {structuredData.thinkingSpace.possible_inconsistencies.map(
                                                                            (
                                                                                inc,
                                                                                index
                                                                            ) => (
                                                                                <li
                                                                                    key={
                                                                                        index
                                                                                    }
                                                                                >
                                                                                    {
                                                                                        inc
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
                                            </div>
                                        )}

                                        <div className="mt-4 flex justify-end">
                                            <Button
                                                onClick={() => {
                                                    navigator.clipboard.writeText(
                                                        extractedText
                                                    );
                                                }}
                                                variant="outline"
                                                className="flex items-center"
                                            >
                                                <svg
                                                    className="mr-2 h-4 w-4"
                                                    xmlns="http://www.w3.org/2000/svg"
                                                    viewBox="0 0 24 24"
                                                    fill="none"
                                                    stroke="currentColor"
                                                    strokeWidth="2"
                                                    strokeLinecap="round"
                                                    strokeLinejoin="round"
                                                >
                                                    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
                                                    <rect
                                                        x="8"
                                                        y="2"
                                                        width="8"
                                                        height="4"
                                                        rx="1"
                                                        ry="1"
                                                    ></rect>
                                                </svg>
                                                Copy Text
                                            </Button>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="flex flex-1 flex-col items-center justify-center text-center">
                                        <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-emerald-100">
                                            <svg
                                                className="h-8 w-8 text-emerald-600"
                                                xmlns="http://www.w3.org/2000/svg"
                                                viewBox="0 0 24 24"
                                                fill="none"
                                                stroke="currentColor"
                                                strokeWidth="2"
                                                strokeLinecap="round"
                                                strokeLinejoin="round"
                                            >
                                                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                                <polyline points="14 2 14 8 20 8"></polyline>
                                                <line
                                                    x1="16"
                                                    y1="13"
                                                    x2="8"
                                                    y2="13"
                                                ></line>
                                                <line
                                                    x1="16"
                                                    y1="17"
                                                    x2="8"
                                                    y2="17"
                                                ></line>
                                                <polyline points="10 9 9 9 8 9"></polyline>
                                            </svg>
                                        </div>
                                        <h3 className="mb-2 text-lg font-medium text-gray-800">
                                            Ready to Process
                                        </h3>
                                        <p className="mb-4 text-sm text-gray-600">
                                            Click the button below to extract
                                            text from your document.
                                        </p>
                                        <Button
                                            onClick={() =>
                                                uploadFile(file as File)
                                            }
                                            disabled={isUploading}
                                            className="w-full"
                                            size="lg"
                                        >
                                            {isUploading
                                                ? "Processing..."
                                                : "Process File"}
                                        </Button>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
