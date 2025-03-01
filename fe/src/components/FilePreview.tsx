export function FilePreview({
    file,
    onFileRemove,
}: {
    file: File;
    onFileRemove: () => void;
}) {
    return (
        <>
            {" "}
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
                        <p className="font-medium text-gray-800">{file.name}</p>
                        <p className="text-xs text-gray-500">
                            {(file.size / 1024).toFixed(2)} KB •{" "}
                            {file.type || "Unknown type"}
                        </p>
                    </div>
                </div>
                <button
                    onClick={onFileRemove}
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
        </>
    );
}
