const CONFIG = {
  modelPath: "best.onnx",
  labels: ["Kelas_Satu", "Kelas_Dua"],
  threshold: 0.45
};

let session;
let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let resultText = document.getElementById("result");

// 🔹 Start Webcam
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: true
  });
  video.srcObject = stream;
}

// 🔹 Load Model
async function loadModel() {
  session = await ort.InferenceSession.create(CONFIG.modelPath);
  console.log("Model loaded");
}

// 🔹 Preprocess frame
function preprocess() {
  ctx.drawImage(video, 0, 0, 640, 480);

  let imageData = ctx.getImageData(0, 0, 640, 480);
  let data = imageData.data;

  let input = new Float32Array(3 * 640 * 480);

  for (let i = 0; i < 640 * 480; i++) {
    input[i] = data[i * 4] / 255.0;
    input[i + 640 * 480] = data[i * 4 + 1] / 255.0;
    input[i + 2 * 640 * 480] = data[i * 4 + 2] / 255.0;
  }

  return new ort.Tensor("float32", input, [1, 3, 640, 480]);
}

// 🔹 Run AI
async function detect() {
  const inputTensor = preprocess();

  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;

  const results = await session.run(feeds);
  const output = results[session.outputNames[0]].data;

  let max = Math.max(...output);
  let index = output.indexOf(max);

  if (max > CONFIG.threshold) {
    resultText.innerText = CONFIG.labels[index] + " (" + (max * 100).toFixed(1) + "%)";
  } else {
    resultText.innerText = "Tidak yakin";
  }

  requestAnimationFrame(detect);
}

// 🔹 Init App
async function init() {
  await startCamera();
  await loadModel();
  detect();
}

init();
