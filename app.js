const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultText = document.getElementById("result");

let session;

// 🚀 Load webcam
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: true
  });
  video.srcObject = stream;
  return new Promise(resolve => {
    video.onloadedmetadata = () => resolve(video);
  });
}

// 🚀 Load ONNX model
async function loadModel() {
  session = await ort.InferenceSession.create("best.onnx");
  console.log("Model loaded");
}

// 🧠 Preprocess frame (ubah ke tensor)
function preprocess() {
  ctx.drawImage(video, 0, 0, 640, 480);

  const imageData = ctx.getImageData(0, 0, 640, 480);
  const { data } = imageData;

  // contoh: resize ke 224x224 (sesuaikan modelmu!)
  const input = new Float32Array(3 * 224 * 224);

  for (let i = 0; i < 224 * 224; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;

    input[i] = r;
    input[i + 224 * 224] = g;
    input[i + 2 * 224 * 224] = b;
  }

  return new ort.Tensor("float32", input, [1, 3, 224, 224]);
}

// 🔍 Inference loop
async function detect() {
  const inputTensor = preprocess();

  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;

  const results = await session.run(feeds);
  const output = results[session.outputNames[0]].data;

  // ambil nilai tertinggi
  const maxIndex = output.indexOf(Math.max(...output));

  resultText.innerText = "Prediksi: " + maxIndex;

  requestAnimationFrame(detect);
}

// 🚀 Start semua
async function main() {
  await setupCamera();
  await loadModel();
  detect();
}

main();
