document.addEventListener("DOMContentLoaded", () => {
  if (DOCUMENTATION_OPTIONS.pagename !== "user_guide/index") return;

  document.querySelectorAll(".bd-docs-nav details").forEach((details) => {
    const caption = details.querySelector("summary .caption-text");
    if (caption?.textContent.trim() === "Getting Started") {
      details.open = true;
    }
  });
});
