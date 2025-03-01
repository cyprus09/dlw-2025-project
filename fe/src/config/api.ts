export const API_BASE_URL = "http://localhost:8000";

export const API_ENDPOINTS = {
    base: API_BASE_URL,
    hello: `${API_BASE_URL}/`,
    ocr: {
        upload: `${API_BASE_URL}/api/ocr/upload`,
        analyze: `${API_BASE_URL}/api/ocr/upload-and-analyze`,
    },
} as const;
