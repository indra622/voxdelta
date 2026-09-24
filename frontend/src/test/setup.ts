import "@testing-library/jest-dom/vitest";

// jsdom has no layout engine; the smooth scroll on reset is a no-op under test.
window.scrollTo = () => {};
