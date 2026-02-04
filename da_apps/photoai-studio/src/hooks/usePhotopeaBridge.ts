import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { CheckpointOption } from "@/hooks/useInpaintingState.ts";

export const DEFAULT_COMFY_URL =
    import.meta.env.VITE_COMFY_URL || "http://127.0.0.1:8188";

export type InpaintingStatusType = "idle" | "processing" | "success" | "error";

export type InpaintingStatus = {
    type: InpaintingStatusType;
    label: string;
    message: string;
};

type BackendInpaintResponse = {
    image?: string; // may be "data:..." or base64
    detail?: string;
};

function statusLabel(type: InpaintingStatusType): string {
    switch (type) {
        case "idle":
            return "Idle";
        case "processing":
            return "Processing";
        case "success":
            return "Success";
        case "error":
            return "Error";
    }
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer);
    let bin = "";
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
        bin += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
    }
    return btoa(bin);
}

export function usePhotopeaBridge() {
    const iframeRef = useRef<HTMLIFrameElement | null>(null);

    const [isLoading, setIsLoading] = useState<boolean>(true);

    const [status, setStatus] = useState<InpaintingStatus>({
        type: "idle",
        label: "Ready",
        message: "Ready",
    });

    const updateStatus = useCallback((type: InpaintingStatusType, message: string) => {
        setStatus({
            type,
            label: statusLabel(type),
            message,
        });
    }, []);

    useEffect(() => {
        const iframe = iframeRef.current;
        if (!iframe) return;

        const onLoad = () => {
            setIsLoading(false);
            updateStatus(
                "idle",
                import.meta.env.DEV
                    ? "Photopea ready • Dev proxy: /api → ComfyUI"
                    : "Photopea ready • Prod: Tauri proxy → ComfyUI"
            );
        };

        iframe.addEventListener("load", onLoad);
        return () => iframe.removeEventListener("load", onLoad);
    }, [updateStatus]);

    const runPhotopea = useCallback((script: string) => {
        const iframe = iframeRef.current;
        if (!iframe?.contentWindow) throw new Error("Photopea iframe not ready");
        iframe.contentWindow.postMessage(script, "https://www.photopea.com");
    }, []);

    const exportPng = useCallback(
        async (script: string, timeout = 20000): Promise<string> => {
            return new Promise<string>((resolve, reject) => {
                let buffer: ArrayBuffer | null = null;

                const handler = (e: MessageEvent) => {
                    if (e.origin !== "https://www.photopea.com") return;

                    if (e.data instanceof ArrayBuffer) {
                        buffer = e.data;
                        return;
                    }

                    if (e.data === "done") {
                        window.removeEventListener("message", handler);

                        if (!buffer) {
                            reject(new Error("No data returned from Photopea"));
                            return;
                        }

                        const b64 = arrayBufferToBase64(buffer);
                        resolve("data:image/png;base64," + b64);
                    }
                };

                window.addEventListener("message", handler);
                runPhotopea(script);

                window.setTimeout(() => {
                    window.removeEventListener("message", handler);
                    reject(new Error("Photopea export timeout (20s)"));
                }, timeout);
            });
        },
        [runPhotopea]
    );

    const loadImage = useCallback((src: string): Promise<HTMLImageElement> => {
        return new Promise((res, rej) => {
            const img = new Image();
            img.onload = () => res(img);
            img.onerror = () => rej(new Error("Failed to load image"));
            img.src = src;
        });
    }, []);

    const alphaToMask = useCallback(
        async (pngDataUrl: string): Promise<string> => {
            const img = await loadImage(pngDataUrl);

            const c = document.createElement("canvas");
            c.width = img.width;
            c.height = img.height;

            const ctx = c.getContext("2d");
            if (!ctx) throw new Error("Canvas context not available");

            ctx.drawImage(img, 0, 0);

            const im = ctx.getImageData(0, 0, c.width, c.height);
            const d = im.data;

            for (let i = 0; i < d.length; i += 4) {
                const alpha = d[i + 3];
                const inv = 255 - alpha;

                d[i] = inv;
                d[i + 1] = inv;
                d[i + 2] = inv;
                d[i + 3] = 255;
            }

            ctx.putImageData(im, 0, 0);
            return c.toDataURL("image/png");
        },
        [loadImage]
    );

    const getImage = useCallback(async (): Promise<string> => {
        return exportPng(`app.activeDocument.saveToOE("png");`);
    }, [exportPng]);

    const getLayerOnly = useCallback(async (): Promise<string> => {
        const script = `
      var doc = app.activeDocument;
      var state = [];
      for (var i = 0; i < doc.layers.length; i++) {
          var layer = doc.layers[i];
          state.push(layer.visible);
          if (layer.selected) layer.visible = true;
          else layer.visible = false;
      }
      doc.saveToOE("png");
      for (var i = 0; i < doc.layers.length; i++) {
          doc.layers[i].visible = state[i];
      }
    `;
        return exportPng(script);
    }, [exportPng]);

    const addImageAsNewLayer = useCallback(
        async (pngDataUrl: string, layerName = "AI Result"): Promise<void> => {
            const safeName = String(layerName).replace(/"/g, '\\"');

            const script = `
        var resource = "${pngDataUrl}";
        app.open(resource, null, true);
        app.activeDocument.activeLayer.name = "${safeName}";
        if (app.activeDocument.activeLayer.kind == LayerKind.SMARTOBJECT) {
          app.activeDocument.activeLayer.rasterize(RasterizeType.ENTIRELAYER);
        }
      `;
            runPhotopea(script);
        },
        [runPhotopea]
    );

    // Helper: call ComfyUI via Vite proxy in dev, via Tauri in prod
    const callInpaintApi = useCallback(async (payload: unknown): Promise<BackendInpaintResponse> => {
        if (import.meta.env.DEV) {
            const r = await fetch(`/api/image-mask`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data = (await r.json()) as BackendInpaintResponse;
            if (!r.ok) throw new Error(data.detail || "Backend error");
            return data;
        }

        // Production: go through Rust (prevents Origin mismatch 403)
        const data = (await invoke("comfy_post_json", {
            path: "/api/image-mask",
            body: payload,
        })) as BackendInpaintResponse;

        return data;
    }, []);

    const startInpaint = useCallback(
        async (
            positivePrompt: string,
            negativePrompt: string,
            checkpoint: CheckpointOption
        ): Promise<void> => {
            try {
                updateStatus("processing", `Exporting image with ${checkpoint.name}…`);
                const image = await getImage();

                updateStatus("processing", "Extracting mask…");
                const alphaImage = await getLayerOnly();
                const mask = await alphaToMask(alphaImage);

                updateStatus("processing", `Sending to AI (${checkpoint.name})…`);

                const payload = {
                    image,
                    mask,
                    positive_prompt: positivePrompt,
                    negative_prompt: negativePrompt,
                    checkpoint_file: checkpoint.file,
                    checkpoint_name: checkpoint.name,
                };

                const data = await callInpaintApi(payload);

                updateStatus("success", `Result received from ${checkpoint.name}`);
                const imgField = data.image || "";

                const resultImage = imgField.startsWith("data:")
                    ? imgField
                    : "data:image/png;base64," + imgField;

                await addImageAsNewLayer(resultImage, `AI Result (${checkpoint.name})`);
            } catch (e: unknown) {
                console.error(e);
                const msg = e instanceof Error ? e.message : String(e);
                updateStatus("error", msg);
                alert("Error: " + msg);
            }
        },
        [addImageAsNewLayer, alphaToMask, callInpaintApi, getImage, getLayerOnly, updateStatus]
    );

    const openHint = useCallback(() => {
        alert("Use Photopea → File → Open");
    }, []);

    const saveHint = useCallback(() => {
        alert("Use Photopea → File → Save");
    }, []);

    return {
        iframeRef,
        status,
        startInpaint,
        openHint,
        saveHint,
        isLoading,
        // optional debug info:
        comfyUrl: DEFAULT_COMFY_URL,
    };
}
