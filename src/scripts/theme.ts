const THEME_KEY = "theme";
const LIGHT = "light";
const DARK = "dark";

function getPreferredTheme(): string {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored) return stored;
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? DARK
    : LIGHT;
}

function getActiveTheme(): string {
  return localStorage.getItem(THEME_KEY) ?? getPreferredTheme();
}

function reflect(theme?: string): void {
  const currentTheme = theme ?? getActiveTheme();
  const root = document.documentElement;
  root.setAttribute("data-theme", currentTheme);
  root.classList.toggle("dark", currentTheme === DARK);
  
  const themeBtn = document.querySelector("#theme-btn");
  if (themeBtn) {
    themeBtn.setAttribute("aria-label", currentTheme);
  }

  const bg = window.getComputedStyle(document.body).backgroundColor;
  document
    .querySelector("meta[name='theme-color']")
    ?.setAttribute("content", bg);
}

function toggleTheme(): void {
  const currentTheme = getActiveTheme();
  const newTheme = currentTheme === LIGHT ? DARK : LIGHT;
  localStorage.setItem(THEME_KEY, newTheme);
  (window as unknown as { __theme?: { value: string } }).__theme = { value: newTheme };
  reflect(newTheme);
}

function setup(): void {
  reflect();
  const themeBtn = document.querySelector("#theme-btn");
  if (themeBtn && !themeBtn.hasAttribute("data-theme-listener")) {
    themeBtn.setAttribute("data-theme-listener", "true");
    themeBtn.addEventListener("click", toggleTheme);
  }
}

// Initial setup
setup();

// Ensure theme is set on new document BEFORE swap occurs to prevent flash or mismatch
document.addEventListener("astro:before-swap", event => {
  const currentTheme = getActiveTheme();
  const newDoc = (event as { newDocument: Document }).newDocument;
  if (newDoc && newDoc.documentElement) {
    newDoc.documentElement.setAttribute("data-theme", currentTheme);
    newDoc.documentElement.classList.toggle("dark", currentTheme === DARK);
  }

  const color = document
    .querySelector("meta[name='theme-color']")
    ?.getAttribute("content");
  if (color && newDoc) {
    newDoc.querySelector("meta[name='theme-color']")?.setAttribute("content", color);
  }
});

// Re-run setup after View Transitions swap
document.addEventListener("astro:after-swap", setup);

// Sync with OS-level dark/light preference changes
window
  .matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", ({ matches }) => {
    const newTheme = matches ? DARK : LIGHT;
    localStorage.setItem(THEME_KEY, newTheme);
    reflect(newTheme);
  });
