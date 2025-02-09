document.addEventListener("DOMContentLoaded", function () {
  const confirmLogoutBtn = document.getElementById("confirm-logout");
  const cancelLogoutBtn = document.getElementById("cancel-logout");

  confirmLogoutBtn.addEventListener("click", function () {
    window.location.href = "/logout"; // Redirect to logout route
  });

  cancelLogoutBtn.addEventListener("click", function () {
    window.location.href = "/dashboard"; // Redirect back to dashboard
  });
});
