import { useState } from "react";
import { Button } from "./ui/button";

interface FileUploadProps {
    onFileSelect: (file: File) => void;
    onUploadFile: (file: File) => void;
}

export function FileUpload({ onFileSelect, onUploadFile }: FileUploadProps) {
    const [dragActive, setDragActive] = useState(false);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const selectedFile = e.target.files[0];
            onFileSelect(selectedFile);
            onUploadFile(selectedFile);
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
            onFileSelect(droppedFile);
            onUploadFile(droppedFile);
        }
    };

    return (
        <div className="flex flex-col items-center justify-center h-full">
            <div className="mb-8 text-center">
                <h1 className="mb-2 text-3xl font-bold text-gray-800">
                    Project Report Verification
                </h1>
                <p className="text-gray-600">
                    Upload an image or PDF to verify it
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
                            document.getElementById("file-upload")?.click()
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
    );
}
