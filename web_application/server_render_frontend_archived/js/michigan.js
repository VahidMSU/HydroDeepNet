const images = Array.from(document.querySelectorAll(".image-card img"));
let currentIndex = 0;

function openModal(image) {
  const modal = document.getElementById("imageModal");
  const modalImage = document.getElementById("modalImage");

  currentIndex = parseInt(image.getAttribute("data-index"), 10);
  modalImage.src = image.src;
  modal.style.display = "flex";

  // Add keydown listener
  document.addEventListener("keydown", handleKeyDown);
}

function closeModal() {
  const modal = document.getElementById("imageModal");
  modal.style.display = "none";

  // Remove keydown listener
  document.removeEventListener("keydown", handleKeyDown);
}

function handleKeyDown(event) {
  if (event.key === "ArrowRight") {
    currentIndex = (currentIndex + 1) % images.length; // Wrap to start
  } else if (event.key === "ArrowLeft") {
    currentIndex = (currentIndex - 1 + images.length) % images.length; // Wrap to end
  } else if (event.key === "Escape") {
    closeModal();
    return;
  }
  // Update modal image
  const modalImage = document.getElementById("modalImage");
  modalImage.src = images[currentIndex].src;
}
