# 🖼️ PhotoAIStudio

**PhotoAIStudio** is a modular, browser-based AI image editing platform built with **React**, **TypeScript**, and **Material UI**, integrating advanced tools like **Photopea** with custom AI workflows such as inpainting, upscaling, and other future AI-assisted image operations.

> **Inpainting is just one feature** — the platform is designed to grow into a full AI-powered photo studio.

---

## ✨ Key Features

* 🔌 Embedded **Photopea** editor (via iframe)
* 🧠 AI-powered image processing workflows
* 🧩 Modular feature architecture
* 🎛️ Side-panel driven UI (Drawers)
* ⚡ React + TypeScript (mixed JS/TS supported)
* 🎨 Material UI (MUI)
* 🧭 React Router–based navigation
* 📦 Clean separation of concerns (pages, modules, hooks, API)

---

## 🗂️ Project Structure

```
src/
├── api/
│   └── AppApi.ts                # Central API helpers / backend calls
│
├── components/
│   ├── core/
│   │   ├── AppSnackbar.tsx      # Global notifications
│   │   └── RightDrawer.tsx      # Reusable resizable drawer
│   │
│   └── modules/
│       └── inpainting/
│           └── InpaintingPanel.tsx   # Inpainting UI module
│
├── hooks/
│   └── usePhotopeaBridge.js     # Photopea ↔ App communication logic
│
├── pages/
│   └── PhotoAIStudioPage.tsx    # Main studio page (iframe + feature panels)
│
├── App.tsx                      # App shell (providers only)
├── AppRoutes.tsx                # Route definitions
├── index.tsx                     # Entry point
├── index.css
├── App.css
│
└── temp/                        # Experimental / scratch files
```

---

## 🌐 Public Assets

```
public/
├── index.html
└── favicon.ico
```

---

## 🧠 Architecture Philosophy

### 1️⃣ App.tsx

* Manages **application-level providers only**
* Examples:

    * Router
    * ThemeProvider
    * Global contexts (future)
* Contains **no feature UI logic**

### 2️⃣ AppRoutes.tsx

* Centralized routing using React Router v6
* Each route maps to a full feature page

### 3️⃣ Pages

* High-level UI composition
* Example: `PhotoAIStudioPage`
* Responsible for:

    * AppBar
    * Photopea iframe
    * Feature panels (via Drawer / RightDrawer)

### 4️⃣ Modules

* Each AI feature is self-contained
* Example modules:

    * `inpainting`
    * `upscale` (future)
    * `segmentation` (future)

### 5️⃣ Hooks

* Business logic and integrations
* Keeps UI components clean and declarative
* Example: `usePhotopeaBridge`

---

## 🎨 Current Feature: Inpainting

The **Inpainting module** provides:

* Backend API configuration
* Positive and Negative prompt input
* Mask handling via Photopea
* Start / Open / Save actions
* Processing state feedback

> This module serves as a reference implementation for future AI tools.

---

## 🛣️ Routing

```tsx
<Route path="/" element={<PhotoAIStudioPage />} />
```

Planned future routes:

* `/gallery`
* `/settings`
* `/experiments`
* `/batch`

---

## 🧩 Adding a New AI Module (Quick Guide)

1. Create a new module directory:

```
components/modules/<feature-name>/
```

2. Implement the feature panel UI
3. Add backend logic via `api/` or a custom hook
4. Plug the panel into a page (Drawer or new route)

No core refactor required.

---

## 🛠️ Tech Stack

* React 18
* TypeScript (mixed JS/TS)
* Material UI (MUI)
* React Router v6
* Photopea API
* REST / Fetch-based backend

---

## 📌 Project Status

* 🟢 Active development
* 🧪 Architecture stable
* 🚧 More AI modules planned

---

## 📄 License

Internal / Experimental (define later)
