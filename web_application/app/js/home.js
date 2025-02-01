const images = Array.from(document.querySelectorAll(".image-card img"));
let currentIndex = 0;

function openModal(image) {
  const modal = document.getElementById("imageModal");
  const modalImage = document.getElementById("modalImage");

  currentIndex = images.indexOf(image);
  modalImage.src = image.src;
  modal.style.display = "flex";

  document.addEventListener("keydown", handleKeyDown);
}

function closeModal() {
  const modal = document.getElementById("imageModal");
  modal.style.display = "none";

  document.removeEventListener("keydown", handleKeyDown);
}

function handleKeyDown(event) {
  if (event.key === "ArrowRight") {
    currentIndex = (currentIndex + 1) % images.length;
  } else if (event.key === "ArrowLeft") {
    currentIndex = (currentIndex - 1 + images.length) % images.length;
  } else if (event.key === "Escape") {
    closeModal();
    return;
  }

  const modalImage = document.getElementById("modalImage");
  modalImage.src = images[currentIndex].src;
}
