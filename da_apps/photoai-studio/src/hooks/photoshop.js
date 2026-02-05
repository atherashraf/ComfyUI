import photoshop from "photoshop";
import { storage } from "uxp";

const { app, action, core } = photoshop;

// ---------- Helpers ----------
function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let bin = "";
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
        bin += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
    }
    return btoa(bin);
}

function base64ToUint8Array(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
}

function normalizeDataUrlOrBase64(imgField) {
    if (!imgField) throw new Error("No image returned by backend");
    if (imgField.startsWith("data:")) return imgField;
    return "data:image/png;base64," + imgField;
}

function dataUrlToUint8Array(dataUrl) {
    const idx = dataUrl.indexOf("base64,");
    if (idx < 0) throw new Error("Expected base64 data URL");
    const b64 = dataUrl.slice(idx + "base64,".length);
    return base64ToUint8Array(b64);
}

// Export current document as PNG (composite)
async function exportCompositePngDataUrl() {
    const doc = app.activeDocument;
    if (!doc) throw new Error("No active document");

    // Export to temp file, then read as bytes
    const temp = await storage.localFileSystem.getTemporaryFolder();
    const file = await temp.createFile(`export_${Date.now()}.png`, { overwrite: true });

    await core.executeAsModal(async () => {
        await doc.saveAs.png(file, { compression: 6, interlaced: false });
    }, { commandName: "Export Composite PNG" });

    const buf = await file.read({ format: storage.formats.binary });
    const b64 = arrayBufferToBase64(buf);
    return "data:image/png;base64," + b64;
}

// Export ONLY the active layer as PNG (by hiding others temporarily)
async function exportActiveLayerPngDataUrl() {
    const doc = app.activeDocument;
    if (!doc) throw new Error("No active document");
    const layers = doc.layers;
    const active = doc.activeLayers?.[0] || doc.activeLayer;
    if (!active) throw new Error("No active layer");

    const vis = layers.map(l => l.visible);

    const temp = await storage.localFileSystem.getTemporaryFolder();
    const file = await temp.createFile(`layer_${Date.now()}.png`, { overwrite: true });

    await core.executeAsModal(async () => {
        // hide all, show active
        layers.forEach(l => (l.visible = false));
        active.visible = true;

        await doc.saveAs.png(file, { compression: 6, interlaced: false });

        // restore
        layers.forEach((l, i) => (l.visible = vis[i]));
    }, { commandName: "Export Active Layer PNG" });

    const buf = await file.read({ format: storage.formats.binary });
    const b64 = arrayBufferToBase64(buf);
    return "data:image/png;base64," + b64;
}

// Convert alpha PNG to mask PNG (invert alpha -> grayscale)
async function alphaToMaskDataUrl(alphaPngDataUrl) {
    // UXP can use a canvas in panel UI. Here’s a pure browser-canvas approach:
    // This function expects it runs in a UXP panel context (has DOM).
    const img = await new Promise((res, rej) => {
        const im = new Image();
        im.onload = () => res(im);
        im.onerror = () => rej(new Error("Failed to load alpha png"));
        im.src = alphaPngDataUrl;
    });

    const c = document.createElement("canvas");
    c.width = img.width;
    c.height = img.height;

    const ctx = c.getContext("2d");
    if (!ctx) throw new Error("Canvas context not available");

    ctx.drawImage(img, 0, 0);
    const imd = ctx.getImageData(0, 0, c.width, c.height);
    const d = imd.data;

    for (let i = 0; i < d.length; i += 4) {
        const alpha = d[i + 3];
        const inv = 255 - alpha; // matches your Photopea logic
        d[i] = inv;
        d[i + 1] = inv;
        d[i + 2] = inv;
        d[i + 3] = 255;
    }

    ctx.putImageData(imd, 0, 0);
    return c.toDataURL("image/png");
}

// Place result image (PNG) as a new layer in current doc
async function placePngAsNewLayerFromDataUrl(pngDataUrl, layerName = "AI Result") {
    const doc = app.activeDocument;
    if (!doc) throw new Error("No active document");

    const bytes = dataUrlToUint8Array(pngDataUrl);

    const temp = await storage.localFileSystem.getTemporaryFolder();
    const file = await temp.createFile(`ai_${Date.now()}.png`, { overwrite: true });
    await file.write(bytes, { format: storage.formats.binary });

    await core.executeAsModal(async () => {
        // Open the PNG, copy merged, paste into original doc
        const opened = await app.open(file);

        // Select all + copy merged
        await action.batchPlay(
            [
                { _obj: "selectAll", _target: [{ _ref: "document", _id: opened.id }] },
                { _obj: "copyMerged" },
            ],
            { synchronousExecution: true }
        );

        // Switch back and paste
        app.activeDocument = doc;
        await action.batchPlay([{ _obj: "paste" }], { synchronousExecution: true });

        // Rename pasted layer
        const pasted = doc.activeLayers?.[0] || doc.activeLayer;
        if (pasted) pasted.name = layerName;

        // Close temp doc without save
        app.activeDocument = opened;
        await opened.closeWithoutSaving();
        app.activeDocument = doc;
    }, { commandName: "Place AI Result as Layer" });
}

// ---------- Main: startInpaint ----------
export async function startInpaint({
                                       positivePrompt,
                                       negativePrompt,
                                       checkpoint,      // { name, file }
                                       apiUrl,          // e.g. "http://127.0.0.1:8188/api/image-mask" or your proxy
                                   }) {
    const doc = app.activeDocument;
    if (!doc) throw new Error("No active document");

    // 1) Export full image
    const image = await exportCompositePngDataUrl();

    // 2) Export alpha layer and convert to mask
    const alphaImage = await exportActiveLayerPngDataUrl();
    const mask = await alphaToMaskDataUrl(alphaImage);

    // 3) Call backend
    const payload = {
        image,
        mask,
        positive_prompt: positivePrompt,
        negative_prompt: negativePrompt,
        checkpoint_file: checkpoint.file,
        checkpoint_name: checkpoint.name,
    };

    const r = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    const text = await r.text();
    let data = {};
    try { data = JSON.parse(text); } catch {}
    if (!r.ok) throw new Error(data.detail || text || "Backend error");

    // 4) Place result
    const resultImage = normalizeDataUrlOrBase64(data.image);
    await placePngAsNewLayerFromDataUrl(resultImage, `AI Result (${checkpoint.name})`);
}
