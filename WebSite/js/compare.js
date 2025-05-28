const studentNames = ['Alumno A', 'Alumno B', 'Alumno C', 'Alumno D', 'Alumno E', 'Alumno F'];
const container = document.getElementById('studentOptions');
let selected = [];

studentNames.forEach(name => {
  const div = document.createElement('div');
  div.className = 'student';
  div.innerText = name;
  div.onclick = () => toggleSelect(div);
  container.appendChild(div);
});

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

function toggleSelect(el) {
  if (el.classList.contains('selected')) {
    el.classList.remove('selected');
    selected = selected.filter(name => name !== el.innerText);
  } else {
    if (selected.length < 2) {
      el.classList.add('selected');
      selected.push(el.innerText);
      updateCodeViews();
    } else {
      alert('Solo puedes seleccionar dos alumnos.');
    }
  }
}

function updateCodeViews() {
  if (selected.length === 2) {
    document.getElementById('codeTitle1').innerText = `Código ${selected[0]}`;
    document.getElementById('codeTitle2').innerText = `Código ${selected[1]}`;
    // Aquí podrías cargar el código real dinámicamente si tuvieras backend
  }
}
