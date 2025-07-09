const fileInput = document.getElementById("fileInput");
const folderInput = document.getElementById("folderInput");
const fileList = document.getElementById("fileList");
const fileCount = document.getElementById("fileCount");
const folderPath = document.getElementById("folderPath");
const uploadButton = document.getElementById("uploadButton");

let filesSelected = [];
let filesMap = {}; // Map names to File objects

// Navegation between sites
function goTo(action) {
  if (action === 'index') {
    location.href = "index.html";
  } else if (action === 'upload') {
    location.href = "upload.html";
  } else if (action === 'compare') {
    location.href = "compare.html";
  }
}

// Individual file selection
fileInput.addEventListener("change", () => {
  for (const file of fileInput.files) {
    addFile(file);
  }
  updateCounter();
});

// Complete folder selection (no auto upload)
folderInput.addEventListener("change", () => {
  if (folderInput.files.length > 0) {
    const fullPath = folderInput.files[0].webkitRelativePath;
    const folder = fullPath.split("/")[0];
    if (folderPath) folderPath.textContent = `/${folder}/*`;

    for (const file of folderInput.files) {
      addFile(file);
    }
    updateCounter();
  }
});

// Add file to visual list
function addFile(file) {
  if (filesSelected.includes(file.name)) return;

  filesSelected.push(file.name);
  filesMap[file.name] = file;

  const li = document.createElement("li");
  li.textContent = file.name;

  const removeBtn = document.createElement("span");
  removeBtn.textContent = " ✖";
  removeBtn.style.cursor = "pointer";
  removeBtn.onclick = () => {
    li.remove();
    filesSelected = filesSelected.filter(f => f !== file.name);
    delete filesMap[file.name];
    updateCounter();
  };

  li.appendChild(removeBtn);
  fileList.appendChild(li);
}

// File counting
function updateCounter() {
  fileCount.textContent = `${filesSelected.length} elementos seleccionados`;
}

// Backend uploading
function subirArchivos() {
  if (filesSelected.length === 0) {
    alert("⚠️ No hay archivos para subir.");
    // window.location.href = "./compare.html";
    return;
  }

  const formData = new FormData();
  for (const name of filesSelected) {
    formData.append("files[]", filesMap[name]);
  }

  fetch("http://localhost:5000/upload", {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      alert("✅ Archivos subidos correctamente:\n" + data.uploaded.join("\n"));
      window.location.href = "./compare.html"; // Redirect after uploading
    })
    .catch(err => {
      alert("❌ Error al subir archivos.");
      console.error(err);
    });
}

// Button event for "Subir archivos" button
uploadButton.addEventListener("click", subirArchivos);
