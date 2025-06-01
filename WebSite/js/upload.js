
const fileInput = document.getElementById("fileInput");
const folderInput = document.getElementById("folderInput");
const fileList = document.getElementById("fileList");
const fileCount = document.getElementById("fileCount");
const folderPath = document.getElementById("folderPath");

let filesSelected = [];
let fileContents = {};  // Nuevo: Contendrá {filename: content}

//Ir al sitio correspondiente
function goTo(action) {
  if (action === 'index') {
    location.href = "index.html";
  } else if (action === 'upload') {
    location.href = "upload.html";
  } else if (action === 'compare') {
    location.href = "compare.html";
  }
}

// Leer archivo y almacenarlo
function readAndStoreFile(file) {
  const reader = new FileReader();
  reader.onload = function(event) {
    fileContents[file.name] = event.target.result;
    sessionStorage.setItem("fileContents", JSON.stringify(fileContents));
  };
  reader.readAsText(file);
}

fileInput.addEventListener("change", () => {
  for (const file of fileInput.files) {
    addFile(file.name);
    readAndStoreFile(file);
  }
  updateCounter();
});

folderInput.addEventListener("change", () => {
  if (folderInput.files.length > 0) {
    const fullPath = folderInput.files[0].webkitRelativePath;
    const folder = fullPath.split("/")[0];
    folderPath.innerText = folder;

    for (const file of folderInput.files) {
      addFile(file.webkitRelativePath);
      readAndStoreFile(file);
    }
    updateCounter();
  }
});

function addFile(name) {
  filesSelected.push(name);
  const li = document.createElement("li");
  li.innerText = name;
  fileList.appendChild(li);
}

function updateCounter() {
  fileCount.innerText = filesSelected.length;
}
