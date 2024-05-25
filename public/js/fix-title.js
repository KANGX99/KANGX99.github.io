window.addEventListener("scroll", function() {
  var header = document.querySelector("header");
  var title = document.querySelector(".site-title");
  var distance = window.scrollY;

  if (distance > 100) {
    header.classList.add("fixed-title");
    title.classList.add("fixed-title");
    header.classList.add("active");
    title.classList.add("active");
  } else {
    header.classList.remove("fixed-title");
    title.classList.remove("fixed-title");
    header.classList.remove("active");
    title.classList.remove("active");
  }
});