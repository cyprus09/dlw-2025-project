import { useState } from "react";
import { API_ENDPOINTS } from "../config/api";

export default function HelloWorld() {
    const [file, setFile] = useState<File | null>(null);
    const [extractedText, setExtractedText] = useState<string>("");
    const [isUploading, setIsUploading] = useState(false);
    const [uploadError, setUploadError] = useState<string | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setUploadError(null);
        }
    };

    const handleUpload = async () => {
        if (!file) {
            setUploadError("Please select a file first");
            return;
        }

        setIsUploading(true);
        setUploadError(null);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(API_ENDPOINTS.ocr.extract, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Failed to upload image");
            }

            const data = await response.json();
            setExtractedText(data.text);
        } catch (err) {
            setUploadError(
                err instanceof Error ? err.message : "An error occurred"
            );
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="flex min-h-screen flex-col items-center justify-center gap-8 p-4">
            <div className="w-full max-w-md space-y-4">
                <div className="flex flex-col gap-2">
                    <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                    />
                    <button
                        onClick={handleUpload}
                        disabled={isUploading || !file}
                        className="rounded bg-blue-500 px-4 py-2 font-semibold text-white hover:bg-blue-600 disabled:bg-gray-400"
                    >
                        {isUploading ? "Processing..." : "Extract Text"}
                    </button>
                </div>

                {uploadError && (
                    <div className="rounded-md bg-red-50 p-4 text-red-700">
                        {uploadError}
                    </div>
                )}

                {extractedText && (
                    <div className="rounded-lg border p-4">
                        <h2 className="mb-2 text-lg font-semibold">
                            Extracted Text:
                        </h2>
                        <p className="whitespace-pre-wrap">{extractedText}</p>
                    </div>
                )}
            </div>
        </div>
    );
}
