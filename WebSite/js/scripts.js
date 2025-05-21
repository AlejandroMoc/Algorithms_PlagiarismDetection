//Ir al sitio correspondiente
function goTo(action) {
  if (action === 'index') {
    location.href = "index.html";
  }
  if (action === 'comparar') {
    location.href = "comparar.html";

  } else if (action === 'upload') {
    //alert("Navegar a la interfaz de subida.");
    location.href = "upload.html";
  }
}