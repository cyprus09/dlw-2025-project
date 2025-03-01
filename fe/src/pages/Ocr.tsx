import { useState, useEffect } from "react";
import { Button } from "../components/ui/button";
import { API_ENDPOINTS } from "../config/api";

interface StructuredAnalysisResponse {
    filename: string;
    ocr_text: string;
    structured_response: {
        summary_of_invoice: string;
        invoice_number: string;
        date: string;
        total_amount: number;
        vendor: string;
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
        <div className="flex h-screen flex-col items-center justify-center p-5">
            <div className="relative h-full w-full">
                {!file ? (
                    <div className="flex h-full flex-col items-center justify-center">
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
                                    ? "border-blue-500 bg-blue-50"
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
                        <div className="rounded-lg border bg-white p-6 shadow-sm flex flex-col">
                            <div className="mb-4 flex items-center justify-between">
                                <div className="flex items-center space-x-3">
                                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
                                        {file.type.startsWith("image/") ? (
                                            <svg
                                                xmlns="http://www.w3.org/2000/svg"
                                                className="h-6 w-6 text-blue-600"
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
                                                className="h-6 w-6 text-blue-600"
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
                                            {(file.size / 1024).toFixed(2)} KB •{" "}
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
                                            clipRule="evenodd"
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
                        <div className="rounded-lg border bg-white p-6 shadow-sm flex flex-col">
                            <div className="mb-4 flex items-center justify-between">
                                <h2 className="text-xl font-semibold text-gray-800">
                                    {uploadSuccess
                                        ? "Extracted Text"
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
                                    <div className="mb-4 h-12 w-12 animate-spin rounded-full border-4 border-gray-200 border-t-blue-500"></div>
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
                                        onClick={() => uploadFile(file as File)}
                                        className="mt-4"
                                        variant="outline"
                                    >
                                        Try Again
                                    </Button>
                                </div>
                            ) : uploadSuccess ? (
                                <div className="flex flex-1 flex-col">
                                    {structuredData && (
                                        <div className="mb-4 rounded-lg bg-white border border-gray-200 p-4 shadow-sm">
                                            <h3 className="text-lg font-semibold text-gray-800 mb-3">
                                                Invoice Details
                                            </h3>
                                            <div className="flex flex-col mb-2">
                                                <span className="text-sm font-medium text-gray-500">
                                                    Summary
                                                </span>
                                                <span className="text-base font-medium text-gray-900">
                                                    {
                                                        structuredData.summary_of_invoice
                                                    }
                                                </span>
                                            </div>
                                            <div className="grid grid-cols-2 gap-4">
                                                <div className="flex flex-col">
                                                    <span className="text-sm font-medium text-gray-500">
                                                        Invoice Number
                                                    </span>
                                                    <span className="text-base font-medium text-gray-900">
                                                        {
                                                            structuredData.invoice_number
                                                        }
                                                    </span>
                                                </div>
                                                <div className="flex flex-col">
                                                    <span className="text-sm font-medium text-gray-500">
                                                        Date
                                                    </span>
                                                    <span className="text-base font-medium text-gray-900">
                                                        {structuredData.date}
                                                    </span>
                                                </div>
                                                <div className="flex flex-col">
                                                    <span className="text-sm font-medium text-gray-500">
                                                        Vendor
                                                    </span>
                                                    <span className="text-base font-medium text-gray-900">
                                                        {structuredData.vendor}
                                                    </span>
                                                </div>
                                                <div className="flex flex-col">
                                                    <span className="text-sm font-medium text-gray-500">
                                                        Total Amount
                                                    </span>
                                                    <span className="text-base font-medium text-gray-900">
                                                        $
                                                        {structuredData.total_amount.toFixed(
                                                            2
                                                        )}
                                                    </span>
                                                </div>
                                            </div>
                                            {usageStats && (
                                                <div className="mt-3 pt-3 border-t border-gray-200">
                                                    <details className="text-xs text-gray-500">
                                                        <summary className="cursor-pointer font-medium">
                                                            API Usage Stats
                                                        </summary>
                                                        <div className="mt-2 pl-2">
                                                            <p>
                                                                Prompt Tokens:{" "}
                                                                {
                                                                    usageStats.prompt_tokens
                                                                }
                                                            </p>
                                                            <p>
                                                                Completion
                                                                Tokens:{" "}
                                                                {
                                                                    usageStats.completion_tokens
                                                                }
                                                            </p>
                                                            <p>
                                                                Total Tokens:{" "}
                                                                {
                                                                    usageStats.total_tokens
                                                                }
                                                            </p>
                                                        </div>
                                                    </details>
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    <div className="mb-2 flex items-center">
                                        <svg
                                            className="mr-2 h-5 w-5 text-gray-500"
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
                                        <span className="text-sm font-medium text-gray-700">
                                            Raw Extracted Text from{" "}
                                            <span className="font-semibold">
                                                {file?.name}
                                            </span>
                                        </span>
                                    </div>

                                    <div className="flex-1 overflow-auto rounded-md border border-gray-200 bg-gray-50 p-4">
                                        {extractedText ? (
                                            <pre className="whitespace-pre-wrap break-words text-sm text-gray-700">
                                                {extractedText}
                                            </pre>
                                        ) : (
                                            <p className="text-center text-gray-500">
                                                No text was extracted from this
                                                document.
                                            </p>
                                        )}
                                    </div>

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
                                    <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-blue-100">
                                        <svg
                                            className="h-8 w-8 text-blue-600"
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
                                        Click the button below to extract text
                                        from your document.
                                    </p>
                                    <Button
                                        onClick={() => uploadFile(file as File)}
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
    );
}
