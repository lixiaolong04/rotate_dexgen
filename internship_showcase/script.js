document.documentElement.classList.add("js");

const header = document.querySelector(".site-header");
const progress = document.querySelector(".scroll-progress span");
const menuToggle = document.querySelector(".menu-toggle");
const navigation = document.querySelector(".site-nav");
const navLinks = [...document.querySelectorAll(".site-nav a")];

function updateScrollUI() {
  const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
  const ratio = maxScroll > 0 ? window.scrollY / maxScroll : 0;
  progress.style.width = `${Math.min(1, Math.max(0, ratio)) * 100}%`;
  header.classList.toggle("is-scrolled", window.scrollY > 28);
}

updateScrollUI();
window.addEventListener("scroll", updateScrollUI, { passive: true });
window.addEventListener("resize", updateScrollUI);

menuToggle?.addEventListener("click", () => {
  const open = menuToggle.getAttribute("aria-expanded") !== "true";
  menuToggle.setAttribute("aria-expanded", String(open));
  navigation.classList.toggle("is-open", open);
});

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    navigation.classList.remove("is-open");
    menuToggle?.setAttribute("aria-expanded", "false");
  });
});

const revealObserver = new IntersectionObserver(
  (entries, observer) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("is-visible");
      observer.unobserve(entry.target);
    });
  },
  { threshold: 0.12, rootMargin: "0px 0px -5%" },
);

document.querySelectorAll(".reveal").forEach((element) => {
  const delay = element.dataset.delay;
  if (delay) element.style.setProperty("--reveal-delay", `${delay}ms`);
  revealObserver.observe(element);
});

const sectionObserver = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;

    navLinks.forEach((link) => {
      link.classList.toggle("is-active", link.hash === `#${visible.target.id}`);
    });
  },
  { threshold: [0.18, 0.35, 0.55], rootMargin: "-12% 0px -45%" },
);

document
  .querySelectorAll("main section[id]")
  .forEach((section) => sectionObserver.observe(section));

const videoObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      const video = entry.target;
      if (entry.isIntersecting) {
        video.play().catch(() => {});
      } else {
        video.pause();
      }
    });
  },
  { threshold: 0.28 },
);

document.querySelectorAll("video[data-playback-rate]").forEach((video) => {
  const rate = Number(video.dataset.playbackRate);
  if (!Number.isFinite(rate) || rate <= 0) return;

  const applyPlaybackRate = () => {
    video.defaultPlaybackRate = rate;
    video.playbackRate = rate;
  };

  if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
    applyPlaybackRate();
  } else {
    video.addEventListener("loadedmetadata", applyPlaybackRate, { once: true });
  }
});

document.querySelectorAll("video[data-autoplay]").forEach((video) => {
  video.muted = true;
  videoObserver.observe(video);
});

function animateNumber(element) {
  if (element.dataset.animated) return;
  element.dataset.animated = "true";
  const target = Number(element.dataset.count);
  const decimals = Number(element.dataset.decimals || 0);
  const duration = 1100;
  const started = performance.now();

  function tick(now) {
    const elapsed = Math.min(1, (now - started) / duration);
    const eased = 1 - Math.pow(1 - elapsed, 3);
    element.textContent = (target * eased).toFixed(decimals);
    if (elapsed < 1) requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);
}

const counterObserver = new IntersectionObserver(
  (entries, observer) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      animateNumber(entry.target);
      observer.unobserve(entry.target);
    });
  },
  { threshold: 0.6 },
);

document.querySelectorAll("[data-count]").forEach((counter) => counterObserver.observe(counter));

const fullscreenButton = document.querySelector("[data-fullscreen]");

async function toggleFullscreen() {
  try {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
  } catch (error) {
    console.warn("Fullscreen is not available in this browser.", error);
  }
}

fullscreenButton?.addEventListener("click", toggleFullscreen);
document.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() === "p" && !event.ctrlKey && !event.metaKey) {
    toggleFullscreen();
  }
});

const lightbox = document.querySelector(".lightbox");
const lightboxImage = lightbox?.querySelector("img");
const lightboxClose = lightbox?.querySelector("button");

document.querySelectorAll("[data-lightbox]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!lightbox || !lightboxImage) return;
    lightboxImage.src = button.dataset.lightbox;
    lightbox.showModal();
  });
});

lightboxClose?.addEventListener("click", () => lightbox.close());
lightbox?.addEventListener("click", (event) => {
  if (event.target === lightbox) lightbox.close();
});

document.querySelectorAll("[data-year]").forEach((element) => {
  element.textContent = String(new Date().getFullYear());
});
