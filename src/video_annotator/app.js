import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let poseLandmarker;
let lastVideoTime = -1;
let currentLandmarks = null;

let startFrame = null;
let endFrame = null;
let frameRate = 30;
let currentLabel = "";
const MIN_CONFIDENCE = 0.5;

// stop random errors when the browser & mediapipe disagree on what time it is:
let lastTimestamp = 0;
const MIN_INCREMENT_MS = 1;


const CONNECTED_PAIRS = [
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
  ['left_hip', 'right_hip'],
  ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle']
];

const LANDMARK_NAMES = [
  "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
  "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
  "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
  "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
  "right_heel", "left_foot_index", "right_foot_index"
];

// File upload
document.getElementById("videoUploader").addEventListener("change", function (e) {
  let file = e.target.files[0];
  if (file) {
    video.src = URL.createObjectURL(file);
    video.load();
  }
});

// Video metadata
video.addEventListener("loadedmetadata", () => {
  frameRate = video.frameRate || 30;
});

window.stepFrame = async function (dir) {
  video.pause();
  video.currentTime += dir * (1 / frameRate);
  if (dir < 0) {
    await resetLandmarker();
  }
};

window.seekToFrame = async function () {
  const frame = parseInt(document.getElementById("seekFrame").value);
  video.pause();
  const previousTime = video.currentTime;
  video.currentTime = frame / frameRate;
  if (video.currentTime < previousTime) {
    await resetLandmarker();
  }
};

window.markStart = function () {
  startFrame = Math.floor(video.currentTime * frameRate);
  currentLabel = prompt("Enter technique label:", currentLabel || "unknown") || "unknown";
  updateStatus("Start marked at frame " + startFrame);
};

window.markEnd = function () {
  endFrame = Math.floor(video.currentTime * frameRate);
  updateStatus("End marked at frame " + endFrame);
};

window.exportClip = exportClip;
function updateStatus(msg) {
  document.getElementById("status").textContent = msg;
}

// allow the user to click a center point on the canvas to indicate
// which person they want to analyse
let selectedPersonIndex = 0;
let selectionPoint = null; // {x, y} in canvas coordinates

canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) / canvas.width;
  const y = (e.clientY - rect.top) / canvas.height;
  selectionPoint = { x, y };
  console.log("Selected point:", selectionPoint);
});

// -----------------------------------------
// 🧠 Load Pose Landmarker
// -----------------------------------------
(async function initPoseAndLoop() {
  updateStatus("Loading pose model...");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numPoses: 2
  });
  updateStatus("Model ready — upload a video to begin.");

  requestAnimationFrame(renderLoop);
})();

// Re-initialize model after seeking
async function resetLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 2,
  });
}

// Create a _separate_ pose landmarker in IMAGE mode for exporting annotated sequences,
// otherwise we also get timestamp mismatch errors on export because we typically mark 
// the start, play forward to the end, then click export - so at that point we have to seek
// backwards:
let poseLandmarkerForExport = null;

async function initExportLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  poseLandmarkerForExport = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "IMAGE",
    numPoses: 2,
  });
}


  // (re-)initialise the dedicated image-mode landmarker:
  initExportLandmarker()


// Return the index of the closest person to the given point
function getClosestPerson(landmarksList, point) {
  if (!landmarksList || !point) return 0;

  let closestIndex = 0;
  let closestDist = Infinity;

  for (let i = 0; i < landmarksList.length; i++) {
    const person = landmarksList[i];
    // Use midpoint between hips as center (landmark 23 & 24)
    const hipCenter = {
      x: (person[23].x + person[24].x) / 2,
      y: (person[23].y + person[24].y) / 2,
    };
    const dx = hipCenter.x - point.x;
    const dy = hipCenter.y - point.y;
    const dist = dx * dx + dy * dy;

    if (dist < closestDist) {
      closestDist = dist;
      closestIndex = i;
    }
  }

  return closestIndex;
}

// -----------------------------------------
// 🎞️ Main Render Loop with Pose Drawing
// -----------------------------------------
function renderLoop() {
  if (video.readyState >= 2 && poseLandmarker) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Generate strictly increasing timestamp
    let nowInMs = Math.floor(video.currentTime * 1000);
    if (nowInMs <= lastTimestamp) {
        nowInMs = lastTimestamp + MIN_INCREMENT_MS;
    }
    lastTimestamp = nowInMs;

    selectedPersonIndex = 0;
    
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        const results = poseLandmarker.detectForVideo(video, nowInMs);
        
        if (selectionPoint) {
            drawCrosshairs(ctx, selectionPoint)
            if (results.landmarks.length > 1) {
                selectedPersonIndex = getClosestPerson(results.landmarks, selectionPoint);
            }
        } else {
            console.log('selectionPoint is None');
        }
        currentLandmarks = results.landmarks[selectedPersonIndex];
    
        for (let i = 0; i < results.landmarks.length; i++) {
            const color = i === selectedPersonIndex ? "lime" : "red";
            // drawLandmarks(results.landmarks[i], color);
            drawPose(ctx, results.landmarks[i], color);
            }
        }
  }
  requestAnimationFrame(renderLoop);
}

// -----------------------------------------
// ✏️ Draw Dots and Skeleton Lines
// -----------------------------------------
function drawPose(ctx, landmarks, color="cyan") {
  if (!landmarks || landmarks.length === 0) return;

  ctx.lineWidth = 2;

  // Only draw if avg visibility is high enough
  const visibility = landmarks.map(lm => lm.visibility || 0);
  const avgVis = visibility.reduce((a, b) => a + b, 0) / visibility.length;
  if (avgVis < MIN_CONFIDENCE) return;

  // Dots
  ctx.fillStyle = "lime";
  landmarks.forEach(lm => {
    ctx.beginPath();
    ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 3, 0, 2 * Math.PI);
    ctx.fill();
  });

  // Lines
  ctx.strokeStyle = color;
  for (let [a, b] of CONNECTED_PAIRS) {
    let idxA = LANDMARK_NAMES.indexOf(a);
    let idxB = LANDMARK_NAMES.indexOf(b);
    if (idxA < 0 || idxB < 0) continue;
    let lmA = landmarks[idxA];
    let lmB = landmarks[idxB];
    if ((lmA.visibility || 0) > MIN_CONFIDENCE && (lmB.visibility || 0) > MIN_CONFIDENCE) {
      ctx.beginPath();
      ctx.moveTo(lmA.x * canvas.width, lmA.y * canvas.height);
      ctx.lineTo(lmB.x * canvas.width, lmB.y * canvas.height);
      ctx.stroke();
    }
  }
}

function drawCrosshairs(ctx, selectionPoint){
    // Draw selection crosshairs if selectionPoint exists
    if (selectionPoint) {
        const x = selectionPoint.x * canvas.width;
        const y = selectionPoint.y * canvas.height;

        ctx.strokeStyle = "yellow";
        ctx.lineWidth = 2;

        // Crosshair horizontal
        ctx.beginPath();
        ctx.moveTo(x - 10, y);
        ctx.lineTo(x + 10, y);
        ctx.stroke();

        // Crosshair vertical
        ctx.beginPath();
        ctx.moveTo(x, y - 10);
        ctx.lineTo(x, y + 10);
        ctx.stroke();
    }
}

// -----------------------------------------
// 💾 Export Annotated Range to ZIP
// -----------------------------------------
async function exportClip() {
  if (startFrame === null || endFrame === null || startFrame >= endFrame) {
    alert("Invalid start/end frames.");
    return;
  }

  const zip = new JSZip();
  const frameCanvas = document.createElement("canvas");
  frameCanvas.width = video.videoWidth;
  frameCanvas.height = video.videoHeight;
  const ctx2 = frameCanvas.getContext("2d");

  for (let i = startFrame; i <= endFrame; i++) {
    updateStatus(`Processing frame ${i} / ${endFrame}`);

    await new Promise(r => {
      video.currentTime = i / frameRate;
      video.onseeked = r;
    });

    ctx2.drawImage(video, 0, 0);
    const nowMs = video.currentTime * 1000;
    /* The line `const result = poseLandmarker.detectForVideo(video, nowMs);` is calling the
    `detectForVideo` method of the `poseLandmarker` object to detect poses in a video frame at a
    specific timestamp. */
    // const result = poseLandmarker.detectForVideo(video, nowMs);
    const result = poseLandmarkerForExport.detect(video);
    const pose = result.landmarks[0];
    if (!pose) continue;

    drawPose(ctx2, pose);

    const pngBlob = await new Promise(resolve => frameCanvas.toBlob(resolve, "image/png"));
    const frameName = `frame_${i.toString().padStart(4, '0')}`;
    zip.file(`${frameName}.png`, pngBlob);

    zip.file(`${frameName}.json`, JSON.stringify({
      frame: i,
      label: currentLabel,
      landmarks: pose
    }, null, 2));
  }

  const content = await zip.generateAsync({ type: "blob" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(content);
  a.download = `annotated_clip_${currentLabel || "unknown"}.zip`;
  a.click();

  updateStatus("Export complete!");
}
