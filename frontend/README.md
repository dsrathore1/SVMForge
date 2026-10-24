# SVMForge Frontend

The SVMForge frontend is a separate Next.js application built with React and JavaScript.

Its purpose is to provide a clean, responsive interface for exploring model predictions, evaluation metrics, and operational status through the SVMForge backend.

**Status:** The Next.js application foundation is in place. Model Studio, Operations Console, and full backend integration are being developed incrementally.

## Technology Stack

- Next.js
- React
- JavaScript
- CSS and responsive UI styling
- FastAPI backend integration — planned
- Light/dark theme toggle — planned as part of the interface experience

## Planned Interface

### Model Studio

Model Studio will provide the user-facing machine learning experience:

- Submit valid feature inputs for prediction.
- Display the model's predicted class.
- Present supported model evaluation metrics.
- Compare Linear and RBF SVM results.
- Display clear validation, loading, success, and error states.

Prediction confidence will only be shown if the selected model and API provide a properly defined, meaningful value.

### Operations Console

Operations Console will focus on service health and operational visibility:

- Backend health status.
- Request counts, latency, and error information.
- Prometheus metrics and Grafana dashboard access.
- Clear indicators when the backend or monitoring service is unavailable.

These features will be documented as complete after implementation and verification.

## Prerequisites

- Node.js compatible with the project's Next.js version
- npm
- Git
- The SVMForge backend environment for integration testing

## Local Development

From the repository root:

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

Keep the development server running while testing the frontend.

## Build and Verification

Run the production build:

```powershell
npm run build
```

Start the production server after a successful build:

```powershell
npm run start
```

The exact available scripts can be checked in `package.json`.

## Backend Integration

The frontend will communicate with the FastAPI backend over HTTP. The backend URL should be configured through an environment variable rather than hardcoded into components.

For local development, a future configuration may use a variable such as:

```text
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Use the actual variable name adopted by the implementation when configuring the application. Do not put credentials or private secrets in variables prefixed with `NEXT_PUBLIC_`, because those values can be exposed to the browser.

CORS must be configured on the backend to allow the intended frontend origin.

## Design Principles

- Lavender and violet visual accents.
- Responsive layouts for desktop and smaller screens.
- A working light/dark theme toggle.
- Reusable UI components and consistent spacing.
- Accessible labels and readable contrast.
- Helpful loading, empty, and error states.
- No unnecessary UI complexity or excessive animations.

## Planned Work

- Finish the shared application layout and navigation.
- Implement Model Studio.
- Implement Operations Console.
- Connect prediction forms to the FastAPI backend.
- Present model metrics from actual experiment results.
- Add health and monitoring information.
- Verify frontend behavior and production builds.
- Complete end-to-end documentation.

## Development Practices

- Use JavaScript, not TypeScript, for this project.
- Keep frontend code separate from backend Python code.
- Avoid hardcoding deployment URLs in components.
- Verify the production build after significant UI changes.
- Use small Conventional Commits, for example:
  - `feat(frontend): build model studio layout`
  - `feat(frontend): integrate prediction API`
  - `fix(frontend): handle backend errors gracefully`

## Project Scope

SVMForge is a portfolio-oriented MLOps project. The frontend is designed to demonstrate model interaction and operational visibility rather than to serve as a general-purpose medical application.
