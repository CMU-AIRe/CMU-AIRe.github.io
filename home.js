/* Home page: papers and people point at each other.
   - Pointing at a person narrows the paper list to theirs (from both tabs); the papers
     come in one after another from the top. Clicking the person keeps it that way.
   - Pointing at a paper fades everyone but its authors; clicking it opens its preview and keeps that.
   - Each column keeps its own click: one person and one paper can be chosen at the same time.
     While a column has something clicked, pointing in that column changes nothing.
   - The tabs switch between publications and preprints. */
(function () {
  const root = document.querySelector("[data-home]");
  if (!root) return;
  const papers = [...root.querySelectorAll("[data-paper]")];
  const people = [...root.querySelectorAll("[data-person]")];
  const tabs = [...root.querySelectorAll("[data-show]")];
  const authorsOf = (paper) => (paper.dataset.authors || "").split(" ").filter(Boolean);
  const list = root.querySelector(".papers");
  const listBox = list.closest(".scroll");
  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;

  let kind = "published";
  let person = null;        // the clicked person's id
  let paper = null;         // the opened paper
  let pointedPerson = null; // while the pointer or keyboard focus is on a person
  let pointedPaper = null;  // ... or on a paper
  let timer = 0;
  let intent = 0;    // a short wait before a person's papers replace the list
  let shownKey = ""; // which papers are listed, so the entrance only plays when that changes

  function render() {
    const who = person || pointedPerson;  // a click wins over pointing
    const which = paper || pointedPaper;

    papers.forEach((p) => {
      p.hidden = who ? !authorsOf(p).includes(who) : p.dataset.kind !== kind;
    });
    const shown = papers.filter((p) => !p.hidden);
    const key = shown.map((p) => p.dataset.paper).join(" ");
    if (key !== shownKey) {
      shownKey = key;
      listBox.scrollTop = 0;
      if (!calm) {
        // Play the entrance again: each paper drops in a moment after the one above it.
        shown.forEach((p, i) => {
          p.classList.remove("enter");
          p.style.animationDelay = Math.min(i, 12) * 32 + "ms";
        });
        void list.offsetWidth;
        shown.forEach((p) => p.classList.add("enter"));
      }
      // A click or a tab that leaves the opened paper out of the list closes it.
      // Merely pointing at someone does not; the paper comes back when the pointer leaves.
      if (paper && paper.hidden && !(pointedPerson && !person)) { open(null); return render(); }
    }

    const keep = which ? new Set(authorsOf(which)) : null;
    people.forEach((el) => {
      el.classList.toggle("dim", !!keep && !keep.has(el.dataset.person));
      el.classList.toggle("on", el.dataset.person === who);
    });

    tabs.forEach((t) => t.setAttribute("aria-pressed", !who && t.dataset.show === kind));
    root.querySelectorAll(".scroll").forEach(updateFade);
  }

  /* Pointing. Each column only listens while nothing in it is clicked. */
  function pointAt(target) {
    clearTimeout(timer);
    clearTimeout(intent);
    if (target.person) {
      if (person) return;
      // Passing over a person on the way somewhere else should not reshuffle the list.
      intent = setTimeout(() => { pointedPerson = target.person; render(); }, 90);
    } else {
      if (paper) return;
      pointedPaper = target.paper;
      render();
    }
  }
  function pointAway() {
    clearTimeout(timer);
    clearTimeout(intent);
    timer = setTimeout(() => { pointedPerson = null; pointedPaper = null; render(); }, 160); // no flicker between neighbours
  }
  function watch(el, target) {
    el.addEventListener("mouseenter", () => pointAt(target));
    el.addEventListener("mouseleave", pointAway);
    el.addEventListener("focusin", () => pointAt(target));
    el.addEventListener("focusout", pointAway);
  }

  /* Opening a paper shows its preview and links. */
  function setOpen(el, isOpen) {
    const row = el.querySelector(".row"), detail = el.querySelector(".detail"), video = el.querySelector("video");
    el.classList.toggle("open", isOpen);
    row.setAttribute("aria-expanded", isOpen);
    detail.inert = !isOpen;
    if (video) { if (isOpen) video.play().catch(() => {}); else video.pause(); }
  }
  function open(el) {
    if (paper) setOpen(paper, false);
    paper = el;
    if (paper) setOpen(paper, true);
    pointedPaper = null;
    remember();
  }
  function choose(id) {
    person = id;
    pointedPerson = null;
    remember();
  }

  /* The address names what is chosen, so a view can be shared: #paper, #person-id, or both. */
  function remember() {
    const parts = [paper && paper.dataset.paper, person && "person-" + person].filter(Boolean);
    history.replaceState(null, "", parts.length ? "#" + parts.join("&") : location.pathname + location.search);
  }

  people.forEach((el) => {
    const id = el.dataset.person;
    watch(el, { person: id });
    el.addEventListener("click", (event) => {
      event.preventDefault();
      clearTimeout(intent);
      const lettingGo = person === id; // clicking the chosen person again lets go
      if (lettingGo && paper) kind = paper.dataset.kind; // keep the open paper in the list that comes back
      choose(lettingGo ? null : id);
      render();
    });
  });
  papers.forEach((el) => {
    watch(el, { paper: el });
    el.querySelector(".row").addEventListener("click", (event) => {
      event.preventDefault();
      open(paper === el ? null : el); // clicking the open paper again closes it
      render();
    });
  });
  tabs.forEach((t) => t.addEventListener("click", () => {
    kind = t.dataset.show;
    choose(null); // a tab always shows its whole list
    render();
  }));
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (paper) open(null); else if (person) choose(null); // close the paper first, then let go of the person
    render();
  });

  /* A soft fade at the bottom of a column while there is more to scroll. */
  function updateFade(box) {
    box.classList.toggle("more-below", box.scrollTop + box.clientHeight < box.scrollHeight - 4);
  }
  root.querySelectorAll(".scroll").forEach((box) => {
    box.addEventListener("scroll", () => updateFade(box), { passive: true });
  });
  addEventListener("resize", () => root.querySelectorAll(".scroll").forEach(updateFade));

  /* Restore a shared view. */
  for (const part of decodeURIComponent(location.hash.slice(1)).split("&")) {
    const linkedPaper = papers.find((p) => p.dataset.paper === part);
    const linkedPerson = part.startsWith("person-") && people.find((p) => p.dataset.person === part.slice(7));
    if (linkedPerson) person = linkedPerson.dataset.person;
    if (linkedPaper) { kind = linkedPaper.dataset.kind; paper = linkedPaper; setOpen(paper, true); }
  }
  render();
  if (paper) paper.scrollIntoView({ block: "nearest" });
})();
