## 2025-03-22 - Toggle Button Accessibility
**Learning:** For custom tab or toggle button groups (like Scan Type or Report Format selectors), visual classes like `bg-teal-600 text-white` are not enough for screen readers. Using `role="group"` on the container with a proper label (`aria-labelledby` or `aria-label`), and setting `aria-pressed={isActive}` on the buttons is critical for conveying the current selection state to assistive technologies. Keyboard focus styles (like `focus-visible:ring-2`) should also be applied to these buttons since they are interactive elements.
**Action:** Always verify that custom UI controls that act as radio buttons or tabs have appropriate ARIA roles (`group`, `radiogroup`, or `tablist`) and state attributes (`aria-pressed`, `aria-checked`, or `aria-selected`), along with clear visual focus indicators.

## 2026-05-04 - Add 'Skip to main content' link
**Learning:** Implementing a keyboard-accessible skip link in a React SPA with Tailwind CSS requires styling it with `sr-only focus:not-sr-only focus:absolute focus:z-[100]` so it appears when focused. The target container (e.g., `<main>`) must have `tabIndex={-1}` to allow programmatic focus and an `outline-none` class to avoid an unwanted default focus ring.
**Action:** When adding skip links, ensure both the link styling and the target container's focusability are correctly implemented.
