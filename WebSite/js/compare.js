
document.getElementById("compareForm").addEventListener("submit", function (e) {
    e.preventDefault();

    const file1 = document.getElementById("file1").files[0];
    const file2 = document.getElementById("file2").files[0];

    if (!file1 || !file2) {
        alert("Por favor, selecciona dos archivos.");
        return;
    }

    const reader1 = new FileReader();
    const reader2 = new FileReader();

    reader1.onload = function (e1) {
        const code1 = e1.target.result;

        reader2.onload = function (e2) {
            const code2 = e2.target.result;

            // Simular una comparación local simple (sólo como demostración)
            const similarity = compareCode(code1, code2);

            document.getElementById("results").innerText =
                "Resultado de la comparación: " + similarity.toFixed(2) + "% de similitud";
        };

        reader2.readAsText(file2);
    };

    reader1.readAsText(file1);
});

function compareCode(a, b) {
    // Comparador simple basado en longitud de coincidencias
    const minLength = Math.min(a.length, b.length);
    let matches = 0;
    for (let i = 0; i < minLength; i++) {
        if (a[i] === b[i]) matches++;
    }
    return (matches / Math.max(a.length, b.length)) * 100;
}
