// Image Recognition System - client-side logic
// Handles drag & drop, file validation, preview, and calling /predict

document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");

  const errorMessage = document.getElementById("error-message");
  const previewSection = document.getElementById("preview-section");
  const imagePreview = document.getElementById("image-preview");

  const predictBtn = document.getElementById("predict-btn");
  const resetBtn = document.getElementById("reset-btn");
  const resetBtn2 = document.getElementById("reset-btn-2");

  const loading = document.getElementById("loading");
  const resultSection = document.getElementById("result-section");
  const resultLabel = document.getElementById("result-label");
  const confidenceFill = document.getElementById("confidence-bar-fill");
  const confidenceText = document.getElementById("confidence-text");
  const otherPredictionsEl = document.getElementById("other-predictions");

  const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png"];
  const ALLOWED_EXT = [".jpg", ".jpeg", ".png"];
  const MAX_SIZE_MB = 8;

  let selectedFile = null;

  // ---------- Helpers ----------
  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
  }

  function clearError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
  }

  function isValidFile(file) {
    const nameLower = file.name.toLowerCase();
    const hasValidExt = ALLOWED_EXT.some((ext) => nameLower.endsWith(ext));
    const hasValidType = ALLOWED_TYPES.includes(file.type) || file.type === "";

    if (!hasValidExt) {
      showError("Invalid file type. Only JPG, JPEG and PNG images are allowed.");
      return false;
    }
    if (!hasValidType && file.type !== "") {
      showError("Invalid file type. Only JPG, JPEG and PNG images are allowed.");
      return false;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      showError(`File is too large. Maximum size is ${MAX_SIZE_MB} MB.`);
      return false;
    }
    return true;
  }

  function resetUI() {
    selectedFile = null;
    fileInput.value = "";
    clearError();
    previewSection.classList.add("hidden");
    resultSection.classList.add("hidden");
    loading.classList.add("hidden");
    dropZone.classList.remove("hidden");
    imagePreview.src = "";
    otherPredictionsEl.innerHTML = "";
  }

  function handleFile(file) {
    clearError();
    if (!isValidFile(file)) {
      return;
    }
    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      previewSection.classList.remove("hidden");
      resultSection.classList.add("hidden");
      dropZone.classList.add("hidden");
    };
    reader.readAsDataURL(file);
  }

  // ---------- Drag & drop events ----------
  ["dragenter", "dragover"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  });

  dropZone.addEventListener("click", () => {
    fileInput.click();
  });

  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
  });

  // ---------- Predict ----------
  predictBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      showError("Please select an image first.");
      return;
    }

    clearError();
    previewSection.classList.add("hidden");
    loading.classList.remove("hidden");
    resultSection.classList.add("hidden");

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      loading.classList.add("hidden");

      if (!response.ok || data.error) {
        showError(data.error || "Something went wrong. Please try again.");
        previewSection.classList.remove("hidden");
        return;
      }

      displayResult(data);
    } catch (err) {
      loading.classList.add("hidden");
      previewSection.classList.remove("hidden");
      showError("Could not reach the server. Please check your connection and try again.");
    }
  });

  function displayResult(data) {
    resultLabel.textContent = data.prediction;
    const confidence = data.confidence;
    confidenceText.textContent = `${confidence}% confidence`;

    // animate bar fill
    confidenceFill.style.width = "0%";
    requestAnimationFrame(() => {
      confidenceFill.style.width = `${confidence}%`;
    });

    otherPredictionsEl.innerHTML = "";
    if (data.other_predictions && data.other_predictions.length > 0) {
      const heading = document.createElement("p");
      heading.style.color = "#64748b";
      heading.style.fontSize = "0.85rem";
      heading.style.margin = "0 0 6px";
      heading.textContent = "Other possibilities:";
      otherPredictionsEl.appendChild(heading);

      data.other_predictions.forEach((item) => {
        const row = document.createElement("div");
        row.className = "other-item";
        row.innerHTML = `<span>${item.label}</span><span>${item.confidence}%</span>`;
        otherPredictionsEl.appendChild(row);
      });
    }

    resultSection.classList.remove("hidden");
  }

  // ---------- Reset ----------
  resetBtn.addEventListener("click", resetUI);
  resetBtn2.addEventListener("click", resetUI);
});
