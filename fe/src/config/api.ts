export const API_BASE_URL = "http://localhost:8000";

export const API_ENDPOINTS = {
    base: API_BASE_URL,
    hello: `${API_BASE_URL}/`,
    ocr: {
        pdf: `${API_BASE_URL}/api/ocr/upload-pdf`,
        image: `${API_BASE_URL}/api/ocr/upload`,
    },
} as const;
