// Interactive modal script for clickable images
document.querySelectorAll(".clickable-image").forEach((img) => {
  img.addEventListener("click", () => {
    const modal = document.getElementById("imageModal");
    const modalImage = document.getElementById("modalImage");
    modalImage.src = img.src;
    modal.style.display = "flex";
  });
});

function closeModal() {
  const modal = document.getElementById("imageModal");
  modal.style.display = "none";
  const modalImage = document.getElementById("modalImage");
  modalImage.src = "";
}

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeModal();
});
