// src/editor/useEditorVM.ts
import { useEffect, useMemo, useReducer, useRef } from "react";
import { editorReducer, initialState, makeId } from "./reducer";
import { isModKey } from "./shortcuts";
import type { Layer } from "./types";
import { buildMergedLayer, mergeTwoLayersToDataUrl } from "./merge";

async function fileToDataUrl(file: File): Promise<{ dataUrl: string; w: number; h: number }> {
    const dataUrl = await new Promise<string>((resolve, reject) => {
        const fr = new FileReader();
        fr.onload = () => resolve(String(fr.result));
        fr.onerror = reject;
        fr.readAsDataURL(file);
    });

    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
        const i = new Image();
        i.onload = () => resolve(i);
        i.onerror = reject;
        i.src = dataUrl;
    });

    return { dataUrl, w: img.width, h: img.height };
}

function fitLayerToCanvas(canvasW: number, canvasH: number, imgW: number, imgH: number) {
    const scale = Math.min(canvasW / imgW, canvasH / imgH);
    const drawW = imgW * scale;
    const drawH = imgH * scale;
    const x = (canvasW - drawW) / 2;
    const y = (canvasH - drawH) / 2;
    return { x, y, scaleX: scale, scaleY: scale };
}

export function useEditorVM() {
    const [state, dispatch] = useReducer(editorReducer, initialState);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const canvasWRef = useRef(state.canvasW);
    const canvasHRef = useRef(state.canvasH);

    useEffect(() => {
        canvasWRef.current = state.canvasW;
        canvasHRef.current = state.canvasH;
    }, [state.canvasW, state.canvasH]);

    const activeIndex = useMemo(() => {
        if (!state.activeLayerId) return -1;
        return state.layers.findIndex((l) => l.id === state.activeLayerId);
    }, [state.layers, state.activeLayerId]);

    function openImageDialog() {
        fileInputRef.current?.click();
    }

    async function onPickFile(file?: File) {
        if (!file) return;
        const { dataUrl, w, h } = await fileToDataUrl(file);

        dispatch({ type: "SET_CANVAS", w, h });

        const fit = fitLayerToCanvas(w, h, w, h);

        const layer: Layer = {
            id: makeId("img"),
            name: file.name,
            type: "image",
            visible: true,
            opacity: 1,
            x: fit.x,
            y: fit.y,
            scaleX: 1,
            scaleY: 1,
            rotation: 0,
            src: dataUrl,
            width: w,
            height: h,
            filters: {},
        };

        dispatch({ type: "ADD_IMAGE_LAYER", layer });
    }

    function copy() {
        if (!state.activeLayerId) return;
        dispatch({ type: "COPY_LAYER", id: state.activeLayerId });
    }

    function pasteLayer() {
        dispatch({ type: "PASTE_LAYER" });
    }

    function duplicate() {
        if (!state.activeLayerId) return;
        dispatch({ type: "DUPLICATE_LAYER", id: state.activeLayerId });
    }

    async function mergeDown() {
        if (activeIndex < 0) return;
        const top = state.layers[activeIndex];
        const bottom = state.layers[activeIndex + 1];
        if (!top || !bottom) return;

        const { mergedSrc } = await mergeTwoLayersToDataUrl(state.canvasW, state.canvasH, top, bottom);
        const merged = buildMergedLayer(bottom, mergedSrc, state.canvasW, state.canvasH);

        dispatch({
            type: "MERGE_DOWN_RESULT",
            topId: top.id,
            bottomId: bottom.id,
            merged,
        });
    }

    useEffect(() => {
        const onKeyDown = (e: KeyboardEvent) => {
            if (!isModKey(e)) return;
            const k = e.key.toLowerCase();

            if (k === "c") {
                e.preventDefault();
                copy();
            }
            // do NOT intercept Ctrl+V here (let paste event fire)
            if (k === "j") {
                e.preventDefault();
                duplicate();
            }
            if (k === "e") {
                e.preventDefault();
                mergeDown();
            }
        };

        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [state.activeLayerId, state.layers, activeIndex, state.canvasW, state.canvasH]);

    useEffect(() => {
        const onPaste = async (e: ClipboardEvent) => {
            const el = document.activeElement;
            const isTyping =
                el &&
                (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || (el as any).isContentEditable);
            if (isTyping) return;

            const items = e.clipboardData?.items;
            if (!items) return;

            const imgItem = Array.from(items).find((it) => it.type.startsWith("image/"));
            if (imgItem) {
                const file = imgItem.getAsFile();
                if (!file) return;

                e.preventDefault();

                const dataUrl = await new Promise<string>((resolve, reject) => {
                    const r = new FileReader();
                    r.onload = () => resolve(String(r.result));
                    r.onerror = reject;
                    r.readAsDataURL(file);
                });

                const img = await new Promise<HTMLImageElement>((resolve, reject) => {
                    const i = new Image();
                    i.onload = () => resolve(i);
                    i.onerror = reject;
                    i.src = dataUrl;
                });

                const cw = canvasWRef.current;
                const ch = canvasHRef.current;
                const fit = fitLayerToCanvas(cw, ch, img.width, img.height);

                dispatch({
                    type: "ADD_IMAGE_LAYER",
                    layer: {
                        id: makeId("clip"),
                        name: `Pasted ${new Date().toLocaleTimeString()}`,
                        type: "image",
                        visible: true,
                        opacity: 1,
                        x: fit.x,
                        y: fit.y,
                        scaleX: fit.scaleX,
                        scaleY: fit.scaleY,
                        rotation: 0,
                        src: dataUrl,
                        width: img.width,
                        height: img.height,
                        filters: {},
                    },
                });

                return;
            }

            dispatch({ type: "PASTE_LAYER" });
        };

        window.addEventListener("paste", onPaste as any, true);
        return () => window.removeEventListener("paste", onPaste as any, true);
    }, []);

    return {
        state,
        dispatch,
        fileInputRef,
        actions: {
            openImageDialog,
            onPickFile,
            copy,
            pasteLayer,
            duplicate,
            mergeDown,
            setActive: (id: string | null) => dispatch({ type: "SET_ACTIVE_LAYER", id }),
            transformPatch: (
                id: string,
                patch: Partial<Pick<Layer, "x" | "y" | "scaleX" | "scaleY" | "rotation">>
            ) => dispatch({ type: "UPDATE_LAYER_TRANSFORM", id, patch }),
            toggleVis: (id: string) => dispatch({ type: "TOGGLE_LAYER_VIS", id }),
            rename: (id: string, name: string) => dispatch({ type: "RENAME_LAYER", id, name }),
            opacity: (id: string, opacity: number) => dispatch({ type: "SET_LAYER_OPACITY", id, opacity }),
        },
    };
}
