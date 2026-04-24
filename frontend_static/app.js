const imageInput = document.getElementById("imageInput");
const confidenceRange = document.getElementById("confidenceRange");
const confidenceValue = document.getElementById("confidenceValue");
const detectButton = document.getElementById("detectButton");
const errorNode = document.getElementById("error");
const resultsSection = document.getElementById("resultsSection");
const imageStage = document.getElementById("imageStage");
const preview = document.getElementById("preview");
const totalCount = document.getElementById("totalCount");
const detectionList = document.getElementById("detectionList");

let selectedFile = null;
let previewUrl = "";

function setError(message) {
  errorNode.textContent = message || "";
}

function clearBBoxes() {
  imageStage.querySelectorAll(".bbox").forEach((node) => node.remove());
}

imageInput.addEventListener("change", (event) => {
  selectedFile = event.target.files && event.target.files[0] ? event.target.files[0] : null;
  clearBBoxes();
  detectionList.innerHTML = "";
  totalCount.textContent = "0";
  setError("");

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = "";
  }

  if (!selectedFile) {
    preview.removeAttribute("src");
    resultsSection.hidden = true;
    return;
  }

  previewUrl = URL.createObjectURL(selectedFile);
  preview.src = previewUrl;
  resultsSection.hidden = false;
});

confidenceRange.addEventListener("input", (event) => {
  confidenceValue.textContent = Number(event.target.value).toFixed(2);
});

detectButton.addEventListener("click", async () => {
  if (!selectedFile) {
    setError("Please choose an image before running detection.");
    return;
  }

  detectButton.disabled = true;
  detectButton.textContent = "Detecting...";
  setError("");
  clearBBoxes();
  detectionList.innerHTML = "";

  try {
    const formData = new FormData();
    formData.append("image", selectedFile);
    const conf = Number(confidenceRange.value).toFixed(2);

    const response = await fetch(`http://127.0.0.1:8000/detect?conf=${conf}`, {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Detection request failed.");
    }

    const detections = data.detections || [];
    totalCount.textContent = String(detections.length);

    detections.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = `${item.className} | Confidence ${item.confidence}`;
      detectionList.appendChild(li);

      const width = data.width || 1;
      const height = data.height || 1;
      const left = (item.bbox.x1 / width) * 100;
      const top = (item.bbox.y1 / height) * 100;
      const boxWidth = (item.bbox.width / width) * 100;
      const boxHeight = (item.bbox.height / height) * 100;

      const box = document.createElement("div");
      box.className = "bbox";
      box.style.left = `${left}%`;
      box.style.top = `${top}%`;
      box.style.width = `${boxWidth}%`;
      box.style.height = `${boxHeight}%`;
      box.style.borderColor = "#f43f5e";

      const badge = document.createElement("span");
      badge.textContent = `${item.className} (${item.confidence})`;
      badge.style.background = "#f43f5e";
      box.appendChild(badge);
      imageStage.appendChild(box);
    });
  } catch (error) {
    setError(error.message || "Unable to process image.");
  } finally {
    detectButton.disabled = false;
    detectButton.textContent = "Run Detection";
  }
});
