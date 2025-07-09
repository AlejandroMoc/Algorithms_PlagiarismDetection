document.addEventListener("DOMContentLoaded", () => {
  const loader = document.getElementById("loaderOverlay");
  const tabla = document.querySelector("#tablaResultados tbody");

  fetch("http://localhost:5000/compare-all")
    .then(res => res.json())
    .then(data => {
      tabla.innerHTML = "";  // Clear previous table if exists

      data.comparaciones.forEach(fila => {
        const [archivo1, archivo2, plagio, tipos, similitud] = fila;

        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${archivo1}</td>
          <td>${archivo2}</td>
          <td class="${plagio === 'Sí' ? 'yes' : 'no'}">${plagio}</td>
          <td>${tipos.length > 0 ? tipos.join(", ") : "-"}</td>
          <td>${similitud}%</td>
        `;
        tabla.appendChild(tr);
      });
    })
    .catch(error => {
      console.error("Error al obtener resultados:", error);
    })
    .finally(() => {
      loader.style.display = "none"; // Hide loader when finished
    });
});

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