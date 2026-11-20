# Chest Xpert web interface

React and Vite frontend for the FastAPI chest radiograph research prototype.

The interface provides Home, New Analysis, Report Review, History, Templates, and About pages. New Analysis accepts chest radiographs and shows validation errors for unsuitable uploads. Report Review displays an editable draft and separate CNN evidence, supports a reviewer acknowledgement, and exports a PDF. The acknowledgement records entered details; it is not identity verification or a clinical signature.

## Start

Run the backend first from the repository root, then:

```bash
cd frontend-react
npm ci
npm run dev
```

Vite serves the app at `http://localhost:5173`. Configure `VITE_API_URL` if the backend is at a different origin.

Build with `npm run build`. The generated `dist/` directory is excluded from Git.
