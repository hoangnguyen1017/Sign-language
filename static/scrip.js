function upload() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    if (!file) {
        alert("Chọn một ảnh trước khi tải lên");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const img = document.getElementById("preview");
        img.src = `/static/uploads/${data.filename}`;
    })
    .catch(error => {
        console.error("Lỗi khi tải ảnh:", error);
    });
}

function clearImage() {
    const img = document.getElementById("preview");
    img.src = "";
    document.getElementById("fileInput").value = "";
}
