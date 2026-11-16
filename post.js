/* Blog post: the contents list marks the section being read. */
(function () {
  const links = [...document.querySelectorAll(".toc a")];
  const sections = links.map((a) => document.getElementById(a.getAttribute("href").slice(1))).filter(Boolean);
  if (!sections.length) return;
  function update() {
    const line = innerHeight * 0.3; // a section counts as "being read" once its heading passes this line
    let current = sections[0];
    for (const s of sections) if (s.getBoundingClientRect().top <= line) current = s;
    links.forEach((a) => a.classList.toggle("here", a.getAttribute("href") === "#" + current.id));
  }
  addEventListener("scroll", update, { passive: true });
  addEventListener("resize", update);
  update();
})();
