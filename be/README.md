# 🚀 DLW 2025 Backend

Backend service for DLW 2025 project built with FastAPI. This service handles all the API endpoints and business logic for the DLW 2025 web application. ⚡

## 📖 Description

This FastAPI backend provides the necessary APIs for the DLW 2025 web application. It includes authentication, user management, and other core functionalities required for the project. The service is built with scalability and maintainability in mind, following best practices in API development. 🏗️

## 🛠️ Getting Started

### ✅ Prerequisites

-   Python 3.8 or higher 🐍
-   pip (Python package installer) 📦
-   virtualenv or venv 🔒

### 🔧 Development Setup

1. **Create and activate a virtual environment:**

    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On macOS/Linux 🍏🐧
    .venv\Scripts\activate     # On Windows 🖥️
    ```

2. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```

3. **Set up pre-commit hooks:**
    ```sh
    pre-commit install
    ```

### 🎯 Working with Pre-Commit Hooks

-   **Bypass hooks during commit:** 🚨

    ```sh
    git commit --no-verify
    ```

-   **Run hooks manually on all files:** 🔄

    ```sh
    pre-commit run --all-files
    ```

-   **Update hooks:** 🔃
    ```sh
    pre-commit autoupdate
    ```

### 📂 Recommended Folder Structure

```
be/
├── src/
│   ├── main.py          # 🚀 FastAPI application entry point
│   ├── api/             # 🌍 API routes and endpoints
│   ├── core/            # ⚙️ Core configurations
│   ├── models/          # 🗄️ Database models
│   ├── schemas/         # 📜 Pydantic models for request/response
│   └── services/        # 🏗️ Business logic
├── .env                 # 🔑 Environment variables
├── .flake8
├── mypy.ini
├── pyproject.toml
├── .gitignore
├── .pre-commit-config.yaml
├── requirements.txt     # 📦 Project dependencies
├── requirements-dev.txt # 🛠️ Project dev dependencies
└── README.md
```

### 📌 Folder Descriptions

**src/**: Main application package 📁

-   **main.py**: FastAPI application initialization and configuration 🚀
-   **api/**: Contains all API routes organized by versions 🌍
-   **core/**: Core configurations and utilities ⚙️
-   **models/**: SQLAlchemy models for database tables 🗄️
-   **schemas/**: Pydantic models for request/response validation 📜
-   **services/**: Business logic and external service integrations 🏗️
