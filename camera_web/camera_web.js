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
    ["localhost", "127.0.0.1"].includes(window.location.hostname)
        ? `ws://${window.location.hostname}:8765`
        : `wss://${window.location.host}/ws`,
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
const backToHrisButton =
document.getElementById(
"backToHrisButton"
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
// ---------------------------------------------------------
// ATTENDANCE / VISUAL RESULT STATE
// ---------------------------------------------------------
// Green box = server-confirmed face + HRIS match.
// Attendance is saved only after Laravel returns result_saved=true.
let faceMatchConfirmed = false;
let attendanceProcessing = false;
let attendanceSaved = false;
// Laravel one-time authorization code from /camera?code=...
const hrisAuthCode =
new URLSearchParams(
window.location.search
).get("code");
// Fixed HRIS redirect destination.
const HRIS_RETURN_URL = "https://hris-medialys.test/";
// Laravel may return the same destination, but the fixed URL above is
// always available as the fallback and is the required redirect target.
let hrisReturnUrl = HRIS_RETURN_URL;
let attendanceRedirected = false;
let hrisAuthenticated = false;
// SSO authorization lifetime returned by Python after Laravel exchange.
let ssoExpiresAt = 0;
let ssoExpiryTimer = null;
// =========================================================
// SSO EXPIRY
// =========================================================
function clearSsoExpiryTimer() {
if (ssoExpiryTimer !== null) {
    window.clearTimeout(
        ssoExpiryTimer
    );
    ssoExpiryTimer = null;
}
ssoExpiresAt = 0;
}
function ssoStillValid() {
return (
    hrisAuthenticated &&
    ssoExpiresAt > Date.now()
);
}
function buildHrisCancelUrl() {
    try {
        const url = new URL(HRIS_RETURN_URL);
        url.searchParams.set('faceauth_cancelled', '1');
        return url.toString();
    } catch (_) {
        return HRIS_RETURN_URL;
    }
}
function stopCameraForSsoExpiry() {
if (attendanceRedirected) {
    return;
}
attendanceRedirected = true;
hrisAuthenticated = false;
destroyed = true;
if (frameTimer !== null) {
    window.clearTimeout(
        frameTimer
    );
    frameTimer = null;
}
if (reconnectTimer !== null) {
    window.clearTimeout(
        reconnectTimer
    );
    reconnectTimer = null;
}
if (socket) {
    try {
        if (socket.readyState === WebSocket.OPEN) {
            socket.send(
                JSON.stringify({
                    type: "logout"
                })
            );
        }
    } catch (_) {}
    try {
        socket.close();
    } catch (_) {}
    socket = null;
}
websocketReady = false;
if (cameraStream) {
    cameraStream
        .getTracks()
        .forEach(
            (track) => track.stop()
        );
    cameraStream = null;
}
resetTracking();
clearSsoExpiryTimer();
setStatus(
    "SSO expired. Kembali ke HRIS..."
);
// The attendance upload, if any, is already handled by Python in the
// background. Browser navigation is independent of that upload.
window.setTimeout(() => {
    window.location.replace(buildHrisCancelUrl());
}, 150);
}
function startSsoExpiryTimer(
remainingSeconds
) {
clearSsoExpiryTimer();
const seconds = Number(
    remainingSeconds
);
if (
    !Number.isFinite(seconds) ||
    seconds <= 0
) {
    stopCameraForSsoExpiry();
    return;
}
ssoExpiresAt =
    Date.now() +
    (seconds * 1000);
ssoExpiryTimer =
    window.setTimeout(
        () => {
            stopCameraForSsoExpiry();
        },
        Math.max(
            0,
            seconds * 1000
        )
    );
}
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
// Fresh state for a new attendance attempt.
faceMatchConfirmed = false;
attendanceProcessing = false;
attendanceSaved = false;
attendanceRedirected = false;
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
        hrisAuthenticated
            ? "Camera + recognition ready"
            : (
                hrisAuthCode
                    ? "Waiting for HRIS authentication..."
                    : "HRIS authorization code missing"
            )
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
function backToHris() {
    // Manual Back is a cancelled FaceAuth attempt.
    attendanceSaved = false;
    attendanceProcessing = false;
    faceMatchConfirmed = false;
    destroyed = true;
    hrisAuthenticated = false;
    clearSsoExpiryTimer();
    if (frameTimer !== null) {
        window.clearTimeout(
            frameTimer
        );
        frameTimer = null;
    }
    if (reconnectTimer !== null) {
        window.clearTimeout(
            reconnectTimer
        );
        reconnectTimer = null;
    }
    if (socket) {
        try {
            if (socket.readyState === WebSocket.OPEN) {
                socket.send(
                    JSON.stringify({
                        type: "logout"
                    })
                );
            }
        } catch (_) {}
        try {
            socket.close();
        } catch (_) {}
        socket = null;
    }
    websocketReady = false;
    if (cameraStream) {
        cameraStream
            .getTracks()
            .forEach(
                (track) => track.stop()
            );
        cameraStream = null;
    }
    cameraReady = false;
    setStatus(
        "Kembali ke HRIS..."
    );
    window.setTimeout(() => {
        window.location.replace(buildHrisCancelUrl());
    }, 100);
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
faceMatchConfirmed = false;
attendanceProcessing = false;
attendanceSaved = false;
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
    if (!hrisAuthCode) {
        console.error(
            "❌ [SSO] Authorization code missing from URL."
        );
        hrisAuthenticated = false;
        setStatus(
            "HRIS authorization code missing"
        );
        return;
    }
    try {
        socket.send(
            JSON.stringify({
                type: "auth",
                code: hrisAuthCode
            })
        );
        setStatus(
            "Authenticating with HRIS..."
        );
    } catch (error) {
        console.error(
            "❌ [SSO] Failed to send auth code:",
            error
        );
        hrisAuthenticated = false;
        setStatus(
            "HRIS authentication failed"
        );
    }
};
socket.onmessage =
    (event) => {
        try {
            const result =
                JSON.parse(
                    event.data
                );
            if (
                result &&
                result.type === "auth_result"
            ) {
                if (!result.success) {
                    hrisAuthenticated = false;
                    clearSsoExpiryTimer();
                    setStatus(
                        result.message ||
                        "HRIS authentication failed"
                    );
                    console.error(
                        "❌ [SSO]",
                        result.message ||
                        "Authentication failed"
                    );
                    return;
                }
                hrisAuthenticated = true;
                hrisReturnUrl =
                    HRIS_RETURN_URL;
                startSsoExpiryTimer(
                    result.sso_expires_in
                );
                console.log(
                    "✅ [SSO] HRIS authentication successful."
                );
                // Remove the one-time code from the address bar
                // after Laravel exchange succeeds.
                try {
                    window.history.replaceState(
                        {},
                        document.title,
                        window.location.pathname
                    );
                } catch (_) {
                    // Non-critical.
                }
                setStatus(
                    cameraReady
                        ? "Camera + recognition ready"
                        : "HRIS authenticated"
                );
                startFrameLoop();
                return;
            }
            if (!hrisAuthenticated) {
                return;
            }
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
        websocketReady = false;
        const wasAuthenticated =
            hrisAuthenticated;
        hrisAuthenticated = false;
        clearSsoExpiryTimer();
        resetTracking();
        if (!destroyed) {
            if (wasAuthenticated) {
                // The Laravel one-time code has already been
                // consumed. Do not reconnect automatically.
                setStatus(
                    "Connection lost. Please return to HRIS."
                );
            } else {
                scheduleReconnect();
            }
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
        websocketReady &&
        hrisAuthenticated &&
        ssoStillValid()
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
    !websocketReady ||
    !hrisAuthenticated ||
    !ssoStillValid()
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
function redirectBackToHrisOnce() {
if (attendanceRedirected) {
    return;
}
// Called only after:
// result_saved=true + attendance_confirmed=true + hris_match=true.
hrisReturnUrl = HRIS_RETURN_URL;
attendanceRedirected = true;
attendanceSaved = true;
attendanceProcessing = false;
faceMatchConfirmed = true;
clearSsoExpiryTimer();
// Stop sending new frames.
if (frameTimer !== null) {
    clearTimeout(frameTimer);
    frameTimer = null;
}
// Close SSO/WebSocket, but keep the visual state on screen.
hrisAuthenticated = false;
if (socket) {
    try {
        if (socket.readyState === WebSocket.OPEN) {
            socket.send(
                JSON.stringify({
                    type: "logout"
                })
            );
        }
    } catch (_) {}
    try {
        socket.close();
    } catch (_) {}
    socket = null;
}
websocketReady = false;
// Stop camera capture, but do NOT clear targetBox/currentBox.
// This leaves the final green box visible until redirect.
if (cameraStream) {
    cameraStream
        .getTracks()
        .forEach(
            (track) => track.stop()
        );
    cameraStream = null;
}
setStatus(
    "Attendance tersimpan ✓ Kembali ke HRIS..."
);
// Allow the browser to paint the green/saved state first.
window.requestAnimationFrame(() => {
    window.setTimeout(() => {
        destroyed = true;
        window.location.assign(
            hrisReturnUrl
        );
    }, 350);
});
}
function handleRecognitionResult(
result
) {
if (!result) {
    return;
}
if (result.status === "sso_expired") {
    stopCameraForSsoExpiry();
    return;
}
// ------------------------------------------------------
// 1. FRAME DIMENSIONS
// ------------------------------------------------------
// Use the same dimensions as the AI image that produced result.box.
if (
    Number.isFinite(Number(result.width)) &&
    Number(result.width) > 0
) {
    aiFrameWidth = Number(result.width);
}
if (
    Number.isFinite(Number(result.height)) &&
    Number(result.height) > 0
) {
    aiFrameHeight = Number(result.height);
}
// ------------------------------------------------------
// 2. SERVER MATCH STATE
// ------------------------------------------------------
// Identity text does NOT control the box color.
// Only backend confirmation does.
if (
    result.hris_match === true &&
    result.attendance_confirmed === true
) {
    faceMatchConfirmed = true;
}
if (result.status === "attendance_processing") {
    // Python has already locked the one-shot recognition.
    faceMatchConfirmed = true;
    attendanceProcessing = true;
}
if (result.result_saved === true) {
    faceMatchConfirmed = true;
    attendanceProcessing = false;
    attendanceSaved = true;
}
// ------------------------------------------------------
// 3. IDENTITY + SIMILARITY
// ------------------------------------------------------
// Update these BEFORE processing/saved returns.
if (identity) {
    identity.textContent =
        result.name ||
        result.candidate ||
        identity.textContent ||
        "Unknown";
}
const similarityValue =
    Number(result.similarity);
if (
    similarity &&
    Number.isFinite(similarityValue)
) {
    similarity.textContent =
        (
            similarityValue * 100
        ).toFixed(1) + "%";
}
// ------------------------------------------------------
// 4. BOUNDING BOX
// ------------------------------------------------------
// Python receives a mirrored frame from the browser.
// Therefore the returned box is already in the same coordinate
// system. Do NOT flip X again.
if (result.box) {
    const x = Number(result.box.x);
    const y = Number(result.box.y);
    const w = Number(result.box.w);
    const h = Number(result.box.h);
    if (
        Number.isFinite(x) &&
        Number.isFinite(y) &&
        Number.isFinite(w) &&
        Number.isFinite(h) &&
        w > 0 &&
        h > 0
    ) {
        targetBox = {
            x,
            y,
            w,
            h
        };
        lastTargetTime =
            performance.now();
        // On match, snap directly to latest server box.
        // This removes smoothing lag at confirmation time.
        if (faceMatchConfirmed) {
            currentBox = {
                ...targetBox
            };
        } else if (!currentBox) {
            currentBox = {
                ...targetBox
            };
        }
    }
}
// ------------------------------------------------------
// 5. FINAL SAVED
// ------------------------------------------------------
// This is the ONLY state that can redirect.
if (
    result.result_saved === true &&
    result.attendance_confirmed === true &&
    result.hris_match === true
) {
    attendanceSaved = true;
    attendanceProcessing = false;
    faceMatchConfirmed = true;
    setStatus(
        "Attendance tersimpan ✓"
    );
    redirectBackToHrisOnce();
    return;
}
// ------------------------------------------------------
// 6. SAVING / PROCESSING
// ------------------------------------------------------
// Keep the last accurate green box while Laravel saves.
if (result.status === "attendance_processing") {
    setStatus(
        "Face cocok ✓ Menyimpan attendance..."
    );
    return;
}
// ------------------------------------------------------
// 7. LOADING
// ------------------------------------------------------
if (result.status === "loading_yunet") {
    setStatus(
        "Loading YuNet..."
    );
    return;
}
// ------------------------------------------------------
// 8. NORMAL DETECTION
// ------------------------------------------------------
if (result.box) {
    setStatus(
        result.status ===
            "face_detected_preparing"
            ? "Face detected • SFace preparing..."
            : faceMatchConfirmed
                ? "Face matched ✓"
                : "Face detected"
    );
} else {
    // Before match, use normal short tracking hold.
    if (!faceMatchConfirmed) {
        const elapsed =
            performance.now() -
            lastTargetTime;
        if (
            elapsed >
            CAMERA_CONFIG.maxTrackingHoldMs
        ) {
            targetBox = null;
            currentBox = null;
            clearOverlay();
        }
        setStatus(
            "Searching for face..."
        );
    } else {
        // After match, never revert the box to red.
        setStatus(
            attendanceSaved
                ? "Attendance tersimpan ✓"
                : "Face cocok ✓ Menyimpan attendance..."
        );
    }
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
const frameWidth =
    Number.isFinite(Number(aiFrameWidth)) &&
    Number(aiFrameWidth) > 0
        ? Number(aiFrameWidth)
        : 640;
const frameHeight =
    Number.isFinite(Number(aiFrameHeight)) &&
    Number(aiFrameHeight) > 0
        ? Number(aiFrameHeight)
        : 480;
const scaleX =
    rect.width /
    frameWidth;
const scaleY =
    rect.height /
    frameHeight;
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
// Green ONLY when backend confirms face + HRIS match.
const name =
    identity
        ? (
            identity.textContent
            || "Unknown"
        )
        : "Unknown";
const color =
    faceMatchConfirmed
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
faceMatchConfirmed = false;
attendanceProcessing = false;
attendanceSaved = false;
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
if (backToHrisButton) {
    backToHrisButton.addEventListener(
        "click",
        backToHris
    );
}
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
// Keep the canvas aligned when a mobile browser changes its visible viewport
// (for example after rotating the phone or opening/closing its browser bar).
if (window.visualViewport) {
    window.visualViewport.addEventListener(
        "resize",
        resizeOverlay
    );
}
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
            if (socket.readyState === WebSocket.OPEN && hrisAuthenticated) {
                socket.send(JSON.stringify({ type: "logout" }));
            }
        } catch (_) {}
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
