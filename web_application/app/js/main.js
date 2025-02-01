// main.js
"use strict";

$(document).ready(function () {
  // Initialize Bootstrap tooltips (if any element uses them)
  $('[data-bs-toggle="tooltip"]').tooltip();

  // Sidebar toggle functionality (requires an element with ID "sidebarToggle")
  $("#sidebarToggle").on("click", function () {
    $(".site-sidebar").toggleClass("active");
  });

  // Highlight the active nav link based on current URL
  var currentPath = window.location.pathname;
  $(".main-nav a").each(function () {
    var linkPath = $(this).attr("href");
    if (linkPath === currentPath) {
      $(this).addClass("active");
    }
  });

  // Smooth scrolling for any anchor links with class "smooth-scroll"
  $("a.smooth-scroll").on("click", function (e) {
    e.preventDefault();
    var target = $(this).attr("href");
    if ($(target).length) {
      $("html, body").animate(
        {
          scrollTop: $(target).offset().top,
        },
        600
      );
    }
  });

  // Back-to-top button functionality (if an element with ID "backToTop" exists)
  var $backToTop = $("#backToTop");
  if ($backToTop.length) {
    // Show or hide the button based on scroll position
    $(window).on("scroll", function () {
      if ($(this).scrollTop() > 300) {
        $backToTop.fadeIn();
      } else {
        $backToTop.fadeOut();
      }
    });
    // Scroll smoothly back to top when clicked
    $backToTop.on("click", function (e) {
      e.preventDefault();
      $("html, body").animate({ scrollTop: 0 }, 600);
    });
  }

  console.log("main.js loaded successfully.");
});
