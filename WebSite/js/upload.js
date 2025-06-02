const fileInput = document.getElementById("fileInput");
const folderInput = document.getElementById("folderInput");
const fileList = document.getElementById("fileList");
const fileCount = document.getElementById("fileCount");
const folderPath = document.getElementById("folderPath");

let filesSelected = [];

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

fileInput.addEventListener("change", () => {
  for (const file of fileInput.files) {
    addFile(file.name);
  }
  updateCounter();
});

folderInput.addEventListener("change", () => {
  if (folderInput.files.length > 0) {
    const fullPath = folderInput.files[0].webkitRelativePath;
    const folder = fullPath.split("/")[0];
    folderPath.textContent = `/${folder}/*`;

    for (const file of folderInput.files) {
      addFile(file.name);
    }
    updateCounter();
  }
});

function addFile(name) {
  if (filesSelected.includes(name)) return;

  filesSelected.push(name);
  const li = document.createElement("li");
  li.textContent = name;

  const removeBtn = document.createElement("span");
  removeBtn.textContent = "✖";
  removeBtn.onclick = () => {
    li.remove();
    filesSelected = filesSelected.filter(f => f !== name);
    updateCounter();
  };

  li.appendChild(removeBtn);
  fileList.appendChild(li);
}

function updateCounter() {
  fileCount.textContent = `${filesSelected.length} elementos seleccionados`;
}
