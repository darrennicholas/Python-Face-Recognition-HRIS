"use strict";


// =========================================================
// CONFIG
// =========================================================

const CAMERA_CONFIG = {

    // Webcam
    cameraWidth: 640,
    cameraHeight: 480,
    cameraFPS: 30,

    // AI
    aiWidth: 640,
    jpegQuality: 0.58,

    // YuNet target FPS
    maxAiFPS: 25,

    // Visual box smoothing
    trackingSmoothing: 0.50,

    // When detection disappears,
    // keep box visually for a very short time.
    maxTrackingHoldMs: 120,

    // WebSocket
    websocketURL:
        "ws://127.0.0.1:8765",

    reconnectDelay: 1500
};


// =========================================================
// DOM
// =========================================================

const video =
    document.getElementById(
        "camera"
    );

const overlay =
    document.getElementById(
        "overlay"
    );

const loading =
    document.getElementById(
        "loading"
    );

const status =
    document.getElementById(
        "status"
    );

const identity =
    document.getElementById(
        "identity"
    );

const similarity =
    document.getElementById(
        "similarity"
    );


if (!video) {
    throw new Error(
        "#camera tidak ditemukan."
    );
}


if (!overlay) {
    throw new Error(
        "#overlay tidak ditemukan."
    );
}


// =========================================================
// CANVAS
// =========================================================

const captureCanvas =
    document.createElement(
        "canvas"
    );

const captureContext =
    captureCanvas.getContext(
        "2d",
        {
            alpha: false
        }
    );

const overlayContext =
    overlay.getContext(
        "2d"
    );


// =========================================================
// STATE
// =========================================================

let cameraStream = null;

let socket = null;

let cameraReady = false;

let websocketReady = false;

let destroyed = false;

let reconnectTimer = null;

let lastFrameSentAt = 0;


// =========================================================
// TRACKING STATE
// =========================================================

let targetBox = null;

let currentBox = null;

let lastTargetTime = 0;

let aiFrameWidth = 640;

let aiFrameHeight = 480;


// =========================================================
// OVERLAY SIZE
// =========================================================

let overlayCssWidth = 0;

let overlayCssHeight = 0;

let overlayDpr = 1;


// =========================================================
// AI FRAME TIMER
// =========================================================

let frameTimer = null;


// =========================================================
// START CAMERA
// =========================================================

async function startCamera() {

    if (cameraReady) {
        return;
    }


    setStatus(
        "Requesting camera..."
    );


    try {

        cameraStream =
            await navigator
                .mediaDevices
                .getUserMedia({

                    video: {

                        width: {
                            ideal: 640,
                            max: 640
                        },

                        height: {
                            ideal: 480,
                            max: 480
                        },

                        frameRate: {
                            ideal: 30,
                            max: 30
                        },

                        facingMode:
                            "user"
                    },

                    audio: false
                });


        video.srcObject =
            cameraStream;


        await video.play();


        cameraReady = true;


        syncCameraGeometry();

        resizeOverlay();


        hideLoading();


        setStatus(
            websocketReady
                ? "Camera + recognition ready"
                : "Camera ready"
        );


        startFrameLoop();


    } catch (error) {

        console.error(
            "[CAMERA]",
            error
        );


        cameraReady = false;


        setStatus(
            "Camera access failed"
        );


        if (loading) {

            loading.textContent =
                "Camera access denied.";
        }
    }
}


// =========================================================
// CAMERA GEOMETRY
// =========================================================

function syncCameraGeometry() {

    const wrapper =
        document.querySelector(
            ".camera-wrapper"
        );


    if (
        !wrapper ||
        !video.videoWidth ||
        !video.videoHeight
    ) {

        return;
    }


    wrapper.style.aspectRatio =
        `${video.videoWidth}/${video.videoHeight}`;


    video.style.objectFit =
        "fill";
}


// =========================================================
// OVERLAY RESIZE
// =========================================================

function resizeOverlay() {

    const rect =
        video.getBoundingClientRect();


    if (
        rect.width <= 0 ||
        rect.height <= 0
    ) {

        return null;
    }


    const dpr =
        window.devicePixelRatio ||
        1;


    const pixelWidth =
        Math.max(
            1,
            Math.round(
                rect.width * dpr
            )
        );


    const pixelHeight =
        Math.max(
            1,
            Math.round(
                rect.height * dpr
            )
        );


    if (
        overlay.width !== pixelWidth ||
        overlay.height !== pixelHeight
    ) {

        overlay.width =
            pixelWidth;

        overlay.height =
            pixelHeight;
    }


    overlay.style.width =
        `${rect.width}px`;

    overlay.style.height =
        `${rect.height}px`;


    overlayCssWidth =
        rect.width;

    overlayCssHeight =
        rect.height;

    overlayDpr =
        dpr;


    overlayContext.setTransform(
        dpr,
        0,
        0,
        dpr,
        0,
        0
    );


    return rect;
}


// =========================================================
// CLEAR
// =========================================================

function clearOverlay() {

    const rect =
        resizeOverlay();


    if (!rect) {
        return;
    }


    overlayContext.clearRect(
        0,
        0,
        rect.width,
        rect.height
    );
}


// =========================================================
// RESET TRACKING
// =========================================================

function resetTracking() {

    targetBox = null;

    currentBox = null;

    lastTargetTime = 0;

    clearOverlay();
}


// =========================================================
// WEBSOCKET
// =========================================================

function connectWebSocket() {

    if (destroyed) {
        return;
    }


    if (
        socket &&
        (
            socket.readyState ===
                WebSocket.OPEN ||

            socket.readyState ===
                WebSocket.CONNECTING
        )
    ) {

        return;
    }


    console.log(
        "🔌 [WS] Connecting..."
    );


    try {

        socket =
            new WebSocket(
                CAMERA_CONFIG
                    .websocketURL
            );

    } catch (error) {

        console.error(
            error
        );


        scheduleReconnect();

        return;
    }


    socket.binaryType =
        "arraybuffer";


    socket.onopen = () => {

        console.log(
            "✅ [WS] Connected."
        );


        websocketReady = true;


        setStatus(
            cameraReady
                ? "Camera + recognition ready"
                : "Recognition connected"
        );


        startFrameLoop();
    };


    socket.onmessage =
        (event) => {

            try {

                const result =
                    JSON.parse(
                        event.data
                    );


                handleRecognitionResult(
                    result
                );


            } catch (error) {

                console.error(
                    "[WS] JSON error:",
                    error
                );
            }
        };


    socket.onerror =
        (error) => {

            console.error(
                "❌ [WS]",
                error
            );


            websocketReady =
                false;
        };


    socket.onclose =
        () => {

            console.log(
                "ℹ️ [WS] disconnected"
            );


            websocketReady =
                false;


            resetTracking();


            if (!destroyed) {

                scheduleReconnect();
            }
        };
}


// =========================================================
// RECONNECT
// =========================================================

function scheduleReconnect() {

    if (destroyed) {
        return;
    }


    if (
        reconnectTimer !==
        null
    ) {

        return;
    }


    reconnectTimer =
        window.setTimeout(

            () => {

                reconnectTimer =
                    null;


                connectWebSocket();

            },

            CAMERA_CONFIG
                .reconnectDelay
        );
}


// =========================================================
// CONTINUOUS FRAME LOOP
// =========================================================

function startFrameLoop() {

    if (frameTimer !== null) {
        return;
    }


    const loop = () => {

        frameTimer = null;


        if (destroyed) {
            return;
        }


        if (
            cameraReady &&
            websocketReady
        ) {

            const now =
                performance.now();


            const interval =
                1000 /
                CAMERA_CONFIG
                    .maxAiFPS;


            if (
                now -
                lastFrameSentAt
                >=
                interval
            ) {

                captureAndSendFrame();
            }
        }


        frameTimer =
            window.setTimeout(
                loop,
                4
            );
    };


    loop();
}


// =========================================================
// CAPTURE + SEND
//
// IMPORTANT:
//
// NO processingFrame LOCK.
//
// Browser can continue sending newest
// frames while Python processes previous frame.
// =========================================================

function captureAndSendFrame() {

    if (
        !cameraReady ||
        !websocketReady
    ) {

        return;
    }


    if (
        !socket ||
        socket.readyState !==
            WebSocket.OPEN
    ) {

        return;
    }


    if (
        video.readyState <
        HTMLMediaElement
            .HAVE_CURRENT_DATA
    ) {

        return;
    }


    /*
     * Prevent WebSocket memory buildup.
     *
     * If previous data is still buffered heavily,
     * skip this cycle.
     */

    if (
        socket.bufferedAmount
        >
        512 * 1024
    ) {

        return;
    }


    const sourceWidth =
        video.videoWidth ||
        640;


    const sourceHeight =
        video.videoHeight ||
        480;


    if (
        sourceWidth <= 0 ||
        sourceHeight <= 0
    ) {

        return;
    }


    const targetWidth =
        Math.min(
            640,
            sourceWidth
        );


    const targetHeight =
        Math.round(
            sourceHeight *
            (
                targetWidth /
                sourceWidth
            )
        );


    captureCanvas.width =
        targetWidth;


    captureCanvas.height =
        targetHeight;


    // ------------------------------------------------------
    // MIRROR FRAME SENT TO PYTHON
    // ------------------------------------------------------

    /*
     * Webcam displayed to user is mirrored.
     *
     * Therefore AI receives the same mirrored frame.
     */

    captureContext.setTransform(
        -1,
        0,
        0,
        1,
        targetWidth,
        0
    );


    captureContext.drawImage(
        video,
        0,
        0,
        targetWidth,
        targetHeight
    );


    lastFrameSentAt =
        performance.now();


    captureCanvas.toBlob(

        (blob) => {

            if (!blob) {
                return;
            }


            if (
                !socket ||
                socket.readyState !==
                    WebSocket.OPEN
            ) {

                return;
            }


            /*
             * Latest-frame system:
             * send immediately.
             */

            try {

                socket.send(
                    blob
                );

            } catch (error) {

                console.error(
                    "[WS] Send error:",
                    error
                );
            }
        },

        "image/jpeg",

        CAMERA_CONFIG
            .jpegQuality
    );
}


// =========================================================
// HANDLE AI RESULT
// =========================================================

function handleRecognitionResult(
    result
) {

    if (!result) {
        return;
    }


    // ------------------------------------------------------
    // Dimensions
    // ------------------------------------------------------

    if (
        Number.isFinite(
            Number(
                result.width
            )
        )
    ) {

        aiFrameWidth =
            Number(
                result.width
            );
    }


    if (
        Number.isFinite(
            Number(
                result.height
            )
        )
    ) {

        aiFrameHeight =
            Number(
                result.height
            );
    }


    // ------------------------------------------------------
    // Loading
    // ------------------------------------------------------

    if (
        result.status ===
        "loading_yunet"
    ) {

        setStatus(
            "Loading YuNet..."
        );


        return;
    }


    // ------------------------------------------------------
    // Current frame HAS FACE
    // ------------------------------------------------------

    if (result.box) {

        targetBox = {

            x:
                Number(
                    result.box.x
                ),

            y:
                Number(
                    result.box.y
                ),

            w:
                Number(
                    result.box.w
                ),

            h:
                Number(
                    result.box.h
                )
        };


        lastTargetTime =
            performance.now();


        /*
         * First frame:
         * initialize exactly at detection.
         */

        if (!currentBox) {

            currentBox = {
                ...targetBox
            };
        }


        setStatus(
            result.status ===
                "face_detected_preparing"

                ? "Face detected • SFace preparing..."

                : "Face detected"
        );


    } else {

        /*
         * Do NOT instantly stop the visual box.
         *
         * Give detector a very short grace period.
         */

        const elapsed =
            performance.now() -
            lastTargetTime;


        if (
            elapsed >
            CAMERA_CONFIG
                .maxTrackingHoldMs
        ) {

            targetBox = null;

            currentBox = null;

            clearOverlay();
        }


        setStatus(
            "Searching for face..."
        );
    }


    // ------------------------------------------------------
    // Identity
    // ------------------------------------------------------

    if (identity) {

        identity.textContent =
            result.name ||
            "Unknown";
    }


    // ------------------------------------------------------
    // Similarity
    // ------------------------------------------------------

    const similarityValue =
        Number(
            result.similarity
        );


    if (
        similarity &&
        Number.isFinite(
            similarityValue
        )
    ) {

        similarity.textContent =
            (
                similarityValue *
                100
            ).toFixed(1)
            + "%";
    }
}


// =========================================================
// VISUAL TRACKING LOOP
//
// Runs independently from AI.
//
// Browser can render at ~60 FPS.
// YuNet can return at ~25 FPS.
//
// This is what makes the box feel continuous.
// =========================================================

function renderTrackingBox() {

    if (destroyed) {
        return;
    }


    const rect =
        resizeOverlay();


    if (!rect) {

        requestAnimationFrame(
            renderTrackingBox
        );

        return;
    }


    overlayContext.clearRect(
        0,
        0,
        rect.width,
        rect.height
    );


    if (
        !targetBox ||
        !currentBox
    ) {

        requestAnimationFrame(
            renderTrackingBox
        );

        return;
    }


    // ------------------------------------------------------
    // Smooth target
    // ------------------------------------------------------

    const alpha =
        CAMERA_CONFIG
            .trackingSmoothing;


    currentBox.x +=
        (
            targetBox.x -
            currentBox.x
        ) * alpha;


    currentBox.y +=
        (
            targetBox.y -
            currentBox.y
        ) * alpha;


    currentBox.w +=
        (
            targetBox.w -
            currentBox.w
        ) * alpha;


    currentBox.h +=
        (
            targetBox.h -
            currentBox.h
        ) * alpha;


    // ------------------------------------------------------
    // Draw
    // ------------------------------------------------------

    drawTrackingBox(
        currentBox,
        rect
    );


    requestAnimationFrame(
        renderTrackingBox
    );
}


// =========================================================
// DRAW TRACKING BOX
// =========================================================

function drawTrackingBox(
    box,
    rect
) {

    if (
        !box ||
        !rect
    ) {

        return;
    }


    const scaleX =
        rect.width /
        aiFrameWidth;


    const scaleY =
        rect.height /
        aiFrameHeight;


    /*
     * NO manual X flip.
     *
     * Frame sent to Python is already mirrored.
     */

    const x =
        box.x *
        scaleX;


    const y =
        box.y *
        scaleY;


    const width =
        box.w *
        scaleX;


    const height =
        box.h *
        scaleY;


    if (
        !Number.isFinite(x) ||
        !Number.isFinite(y) ||
        !Number.isFinite(width) ||
        !Number.isFinite(height)
    ) {

        return;
    }


    const safeX =
        Math.max(
            0,
            Math.min(
                x,
                rect.width
            )
        );


    const safeY =
        Math.max(
            0,
            Math.min(
                y,
                rect.height
            )
        );


    const safeWidth =
        Math.max(
            1,
            Math.min(
                width,
                rect.width -
                safeX
            )
        );


    const safeHeight =
        Math.max(
            1,
            Math.min(
                height,
                rect.height -
                safeY
            )
        );


    // ------------------------------------------------------
    // COLOR
    // ------------------------------------------------------

    const name =
        identity
            ? (
                identity.textContent
                || "Unknown"
            )
            : "Unknown";


    const known =
        name !== "Unknown" &&
        name !== "Preparing";


    const color =
        known
            ? "#16a34a"
            : "#ef3340";


    // ------------------------------------------------------
    // BOX
    // ------------------------------------------------------

    overlayContext.strokeStyle =
        color;


    overlayContext.lineWidth =
        3;


    overlayContext.strokeRect(
        safeX,
        safeY,
        safeWidth,
        safeHeight
    );


    // ------------------------------------------------------
    // LABEL
    // ------------------------------------------------------

    overlayContext.font =
        "600 14px Arial";


    const padding = 8;

    const labelHeight = 26;


    const textWidth =
        overlayContext
            .measureText(
                name
            )
            .width;


    const labelWidth =
        textWidth +
        padding * 2;


    let labelX =
        safeX;


    let labelY =
        safeY -
        labelHeight;


    if (
        labelX +
        labelWidth >
        rect.width
    ) {

        labelX =
            rect.width -
            labelWidth;
    }


    labelX =
        Math.max(
            0,
            labelX
        );


    labelY =
        Math.max(
            0,
            labelY
        );


    overlayContext.fillStyle =
        color;


    overlayContext.fillRect(
        labelX,
        labelY,
        labelWidth,
        labelHeight
    );


    overlayContext.fillStyle =
        "#ffffff";


    overlayContext.fillText(
        name,
        labelX +
            padding,
        labelY +
            18
    );
}


// =========================================================
// LOADING
// =========================================================

function hideLoading() {

    if (!loading) {
        return;
    }


    loading.classList.add(
        "hidden"
    );
}


// =========================================================
// STATUS
// =========================================================

function setStatus(
    message
) {

    if (!status) {
        return;
    }


    status.textContent =
        message;
}


// =========================================================
// IDENTITY RESET
// =========================================================

function resetIdentityDisplay() {

    if (identity) {

        identity.textContent =
            "Unknown";
    }


    if (similarity) {

        similarity.textContent =
            "0.0%";
    }
}


// =========================================================
// EVENTS
// =========================================================

video.addEventListener(
    "loadedmetadata",
    () => {

        syncCameraGeometry();

        resizeOverlay();

        resetTracking();
    }
);


video.addEventListener(
    "resize",
    () => {

        syncCameraGeometry();

        resizeOverlay();
    }
);


window.addEventListener(
    "resize",
    () => {

        syncCameraGeometry();

        resizeOverlay();
    }
);


// =========================================================
// START
// =========================================================

connectWebSocket();

startCamera();


// =========================================================
// START VISUAL TRACKING
// =========================================================

requestAnimationFrame(
    renderTrackingBox
);


// =========================================================
// CLEANUP
// =========================================================

window.addEventListener(
    "beforeunload",
    () => {

        destroyed = true;


        if (
            frameTimer !==
            null
        ) {

            clearTimeout(
                frameTimer
            );

            frameTimer = null;
        }


        if (
            reconnectTimer !==
            null
        ) {

            clearTimeout(
                reconnectTimer
            );

            reconnectTimer = null;
        }


        if (socket) {

            try {

                socket.close();

            } catch (_) {}

            socket = null;
        }


        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(
                    (
                        track
                    ) => {

                        track.stop();
                    }
                );

            cameraStream = null;
        }


        cameraReady = false;

        websocketReady = false;

        resetTracking();
    }
);