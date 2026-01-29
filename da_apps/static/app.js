class PhotopeaBridge {
    constructor() {
        // Configuration
        this.apiUrl = localStorage.getItem("api_url") || "http://localhost:8000";

        // DOM Elements
        this.iframe = document.getElementById("photopea-frame");
        this.statusEl = document.getElementById("status");
        this.loadingOverlay = document.getElementById("loading-overlay");

        this.init();
    }

    /* -------------------- 1. INITIALIZATION -------------------- */

    init() {
        // Setup Photopea Iframe
        if (this.iframe) {
            this.iframe.onload = () => {
                if (this.loadingOverlay) this.loadingOverlay.style.display = "none";
                this.updateStatus("idle", `Photopea ready • API ${this.apiUrl}`);
            };
        }

        // Bind Buttons
        document.getElementById("btn-start")?.addEventListener("click", () => this.start());
        document.getElementById("btn-open")?.addEventListener("click", () =>
            alert("Use Photopea → File → Open")
        );
        document.getElementById("btn-save")?.addEventListener("click", () =>
            alert("Use Photopea → File → Save")
        );

        this.updateStatus("idle", "Ready");
    }

    updateStatus(type, msg) {
        if (!this.statusEl) return;
        this.statusEl.className = `status-indicator status-${type}`;
        this.statusEl.innerHTML = `<span class="status-dot dot-${type}"></span>${msg}`;
    }


    /* -------------------- 3. PHOTOPEA COMMUNICATION -------------------- */

    runPhotopea(script) {
        this.iframe.contentWindow.postMessage(script, "https://www.photopea.com");
    }

    async exportPng(script, timeout = 20000) {
        return new Promise((resolve, reject) => {
            let buffer = null;

            const handler = (e) => {
                if (e.origin !== "https://www.photopea.com") return;

                if (e.data instanceof ArrayBuffer) {
                    buffer = e.data;
                    return;
                }

                if (e.data === "done") {
                    window.removeEventListener("message", handler);
                    if (!buffer) return reject("No data returned from Photopea");

                    // Convert ArrayBuffer to Base64
                    const bytes = new Uint8Array(buffer);
                    let bin = "";
                    for (let b of bytes) bin += String.fromCharCode(b);
                    resolve("data:image/png;base64," + btoa(bin));
                }
            };

            window.addEventListener("message", handler);
            this.runPhotopea(script);

            // Timeout safety
            setTimeout(() => {
                window.removeEventListener("message", handler);
                reject("Photopea export timeout (20s)");
            }, timeout);
        });
    }

    /* -------------------- 4. IMAGE OPERATIONS -------------------- */

    async getImage() {
        return this.exportPng(`app.activeDocument.saveToOE("png");`);
    }

    async getLayerOnly() {
        console.log("Isolating and exporting active layer...");

        const script = `
        var doc = app.activeDocument;
        var active = doc.activeLayer;
        console.log("Active layer", active)
        console.log("buffer data", active.buffer)
        var state = []; // To remember original visibility

        // 1. Hide all layers except the active one
        // We loop through top-level layers only (groups are treated as single layers)
        for (var i = 0; i < doc.layers.length; i++) {
            var layer = doc.layers[i];
            state.push(layer.visible); // Remember state
            
            if(layer.selected){
                layer.visible = true
               
            }else{
                layer.visible = false
            }
        }

        // 2. Export
        doc.saveToOE("png");

        // 3. Restore visibility
        for (var i = 0; i < doc.layers.length; i++) {
             doc.layers[i].visible = state[i];
        }
    `;

        return this.exportPng(script);
    }

    async getTrimmedLayer() {
        const script = `
            var doc = app.activeDocument;
            var d = doc.duplicate(); // We MUST duplicate here to avoid cropping original
            var layer = d.activeLayer;
            
            // Remove others
            for(var i=d.layers.length-1; i>=0; i--) {
                if(d.layers[i] !== layer) d.layers[i].remove();
            }
    
            // Trim transparent pixels (Top, Left, Bottom, Right)
            d.trim(TrimType.TRANSPARENT);
    
            d.saveToOE("png");
            d.close();
        `;
        return this.exportPng(script);
    }

    async addImageAsNewLayer(pngDataUrl, layerName = "AI Result") {
        console.log("Adding image as new layer...");

        const script = `
            var resource = "${pngDataUrl}";
            
            // open(URL, asSmartObject, asLayer) -> true = open as layer in current doc
            app.open(resource, null, true);
            
            // The new layer is automatically active, rename it
            app.activeDocument.activeLayer.name = "${layerName}";
            
            // Rasterize if it loaded as a Smart Object
            if(app.activeDocument.activeLayer.kind == LayerKind.SMARTOBJECT) {
                app.activeDocument.activeLayer.rasterize(RasterizeType.ENTIRELAYER);
            }
        `;

        this.runPhotopea(script);
    }

    /* -------------------- 5. MASK EXTRACTION -------------------- */

    /* -------------------- OPTIMIZED MASK EXTRACTION -------------------- */

    async getMaskFromActiveLayer() {
        console.log("Extracting mask (Fast Mode)...");

        const script = `
        var orig = app.activeDocument;
        var layer = orig.activeLayer;
        
        // 1. Create a new empty document with the same dimensions
        // arguments: width, height, resolution, name, mode, fill
        var newDoc = app.documents.add(orig.width, orig.height, 72, "TempMask", NewDocumentMode.RGB, DocumentFill.TRANSPARENT);
        
        // 2. Duplicate ONLY the active layer to the new document
        // This avoids cloning the entire heavy project history
        layer.duplicate(newDoc, ElementPlacement.PLACEATBEGINNING);
        
        // 3. Switch focus to the new document
        app.activeDocument = newDoc;
        
        // 4. Force Rasterize to bake the mask into pixels
        try {
            newDoc.activeLayer.rasterize(RasterizeType.ENTIRELAYER);
        } catch(e) {
            // Ignore if already rasterized
        }
        
        // 5. Export
        newDoc.saveToOE("png");
        
        // 6. Cleanup: Close without saving
        newDoc.close();
        
        // 7. Restore focus to original
        app.activeDocument = orig;
    `;

        // Increased timeout to 40 seconds just in case
        const pngData = await this.exportPng(script, 40000);

        return this.alphaToMask(pngData);
    }

    async removeAlpha(pngDataUrl) {
        const img = await this.loadImage(pngDataUrl);

        const c = document.createElement("canvas");
        c.width = img.width;
        c.height = img.height;
        const ctx = c.getContext("2d");

        // 1. Draw the image initially
        ctx.drawImage(img, 0, 0);

        // 2. GET RAW PIXEL DATA
        // This gives us an array of [R, G, B, A, R, G, B, A...]
        const imageData = ctx.getImageData(0, 0, c.width, c.height);
        const data = imageData.data;
        console.log("removing alpha data", data.length)
        // 3. FORCE OPACITY
        // Loop through every pixel and set Alpha (the 4th value) to 255.
        for (let i = 0; i < data.length; i += 4) {
            data[i + 3] = 255; // Alpha = 255 (Fully Opaque)
        }

        // 4. Put the modified pixels back
        ctx.putImageData(imageData, 0, 0);

        // 5. Export as JPEG (now 3-band compatible)
        return c.toDataURL("image/jpeg", 0.9);
    }

    async alphaToMask(pngDataUrl) {
        const img = await this.loadImage(pngDataUrl);
        const c = document.createElement("canvas");
        c.width = img.width;
        c.height = img.height;

        const ctx = c.getContext("2d");
        ctx.drawImage(img, 0, 0);

        const im = ctx.getImageData(0, 0, c.width, c.height);
        const d = im.data;

        // Loop through pixels
        for (let i = 0; i < d.length; i += 4) {
            const alpha = d[i + 3]; // Read transparency

            // Write to RGB (Grayscale)
            d[i] = alpha;     // R
            d[i + 1] = alpha; // G
            d[i + 2] = alpha; // B
            d[i + 3] = 255;   // Set full opacity
        }

        ctx.putImageData(im, 0, 0);
        return c.toDataURL("image/png");
    }

    /* -------------------- 6. UTILITIES -------------------- */

    loadImage(src) {
        return new Promise((res, rej) => {
            const img = new Image();
            img.onload = () => res(img);
            img.onerror = rej;
            img.src = src;
        });
    }

    async testApiPing() {
        // Optional debug method
        const tiny = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6n2mUAAAAASUVORK5CYII=";
        const r = await fetch(`${this.apiUrl}/api/image-mask`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({image: tiny, mask: tiny}),
        });
        if (!r.ok) throw new Error("Backend not reachable");
    }

    /* -------------------- 2. MAIN WORKFLOW -------------------- */

    async start() {
        try {
            // Step 1: Get the full image (Context)
            this.updateStatus("processing", "Exporting image...");
            const image = await this.getImage();


            // Step 2: Get the mask from the active layer (Target area)
            this.updateStatus("processing", "Extracting mask...");
            // const mask = await this.getMaskFromActiveLayer();
            const alphaImage = await this.getLayerOnly();
            const mask = await this.alphaToMask(alphaImage)
            // console.log("mask", mask)

            // Step 3: Send payload to backend
            this.updateStatus("processing", "Sending to AI...");
            const r = await fetch(`${this.apiUrl}/api/image-mask`, {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({image, mask}),
            });

            const data = await r.json();
            if (!r.ok) throw new Error(data.detail || "Backend error");

            // Step 4: Add result back to Photopea
            this.updateStatus("success", "Result received");

            const resultImage = data.image.startsWith("data:")
                ? data.image
                : "data:image/png;base64," + data.image;

            await this.addImageAsNewLayer(resultImage, "AI Result");

        } catch (e) {
            console.error(e);
            this.updateStatus("error", e.toString());
            alert("Error: " + e.message);
        }
    }
}

// Start
document.addEventListener("DOMContentLoaded", () => {
    new PhotopeaBridge();
});