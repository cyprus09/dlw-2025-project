import React, { useState } from "react";
import { API_ENDPOINTS } from "../config/api";

const StructuredOutputDemo: React.FC = () => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [result, setResult] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files.length > 0) {
            setSelectedFile(event.target.files[0]);
        }
    };

    const handleSubmit = async () => {
        if (!selectedFile) {
            setError("Please select both a file and a schema type");
            return;
        }

        setIsLoading(true);
        setError(null);
        setResult(null);

        try {
            const formData = new FormData();
            formData.append("file", selectedFile);
            formData.append("file_type", "pdf");

            const response = await fetch(API_ENDPOINTS.ocr.analyze, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(
                    errorData.detail || "Failed to process document"
                );
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            setError((err as Error).message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto">
            <div className="my-8">
                <h1 className="text-3xl font-bold mb-4">
                    Structured Output Demo
                </h1>

                <div className="mb-6">
                    <p className="text-gray-600 mb-4">
                        Upload a document and extract structured data using
                        predefined schemas.
                    </p>
                </div>

                <div className="mb-6">
                    <input
                        accept="image/*,application/pdf"
                        className="hidden"
                        id="contained-button-file"
                        type="file"
                        onChange={handleFileChange}
                    />
                    <label htmlFor="contained-button-file">
                        <button
                            className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition-colors"
                            onClick={() =>
                                document
                                    .getElementById("contained-button-file")
                                    ?.click()
                            }
                        >
                            Upload File
                        </button>
                    </label>
                    {selectedFile && (
                        <p className="mt-2 text-sm text-gray-600">
                            Selected: {selectedFile.name}
                        </p>
                    )}
                </div>

                <div className="mb-6">
                    <button
                        className={`${
                            isLoading || !selectedFile
                                ? "bg-blue-400 cursor-not-allowed"
                                : "bg-blue-600 hover:bg-blue-700"
                        } text-white font-medium py-2 px-4 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition-colors`}
                        onClick={handleSubmit}
                        disabled={isLoading || !selectedFile}
                    >
                        {isLoading ? (
                            <div className="flex items-center">
                                <svg
                                    className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                                    xmlns="http://www.w3.org/2000/svg"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                >
                                    <circle
                                        className="opacity-25"
                                        cx="12"
                                        cy="12"
                                        r="10"
                                        stroke="currentColor"
                                        strokeWidth="4"
                                    ></circle>
                                    <path
                                        className="opacity-75"
                                        fill="currentColor"
                                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                    ></path>
                                </svg>
                                Processing...
                            </div>
                        ) : (
                            "Process Document"
                        )}
                    </button>
                </div>

                {error && (
                    <div className="mb-6">
                        <p className="text-red-600">{error}</p>
                    </div>
                )}

                {result && (
                    <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
                        <h2 className="text-xl font-semibold mb-4">
                            Extracted Structured Data:
                        </h2>
                        <div className="bg-gray-100 p-4 rounded-md overflow-auto max-h-96">
                            <pre className="text-sm">
                                {(() => {
                                    try {
                                        if (
                                            typeof result.structured_response ===
                                            "string"
                                        ) {
                                            return JSON.stringify(
                                                JSON.parse(
                                                    result.structured_response
                                                ),
                                                null,
                                                2
                                            );
                                        } else {
                                            return JSON.stringify(
                                                result.structured_response,
                                                null,
                                                2
                                            );
                                        }
                                    } catch (error) {
                                        return `Error parsing JSON: ${result.structured_response}`;
                                    }
                                })()}
                            </pre>
                        </div>

                        <div className="mt-6">
                            <h3 className="text-lg font-medium mb-2">
                                Usage Statistics:
                            </h3>
                            <p className="text-sm text-gray-600">
                                Prompt Tokens: {result.usage.prompt_tokens}
                            </p>
                            <p className="text-sm text-gray-600">
                                Completion Tokens:{" "}
                                {result.usage.completion_tokens}
                            </p>
                            <p className="text-sm text-gray-600">
                                Total Tokens: {result.usage.total_tokens}
                            </p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default StructuredOutputDemo;
