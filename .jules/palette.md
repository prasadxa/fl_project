## 2025-03-22 - Toggle Button Accessibility
**Learning:** For custom tab or toggle button groups (like Scan Type or Report Format selectors), visual classes like `bg-teal-600 text-white` are not enough for screen readers. Using `role="group"` on the container with a proper label (`aria-labelledby` or `aria-label`), and setting `aria-pressed={isActive}` on the buttons is critical for conveying the current selection state to assistive technologies. Keyboard focus styles (like `focus-visible:ring-2`) should also be applied to these buttons since they are interactive elements.
**Action:** Always verify that custom UI controls that act as radio buttons or tabs have appropriate ARIA roles (`group`, `radiogroup`, or `tablist`) and state attributes (`aria-pressed`, `aria-checked`, or `aria-selected`), along with clear visual focus indicators.

## 2026-04-16 - Skip to Main Content Link
**Learning:** For React SPAs, when implementing a "Skip to main content" link, the target `<main>` container must have `tabIndex={-1}` to allow programmatic focus without breaking keyboard flow, and `outline-none` to prevent unwanted default focus rings when skipped to.
**Action:** Always ensure skip-to-content targets have both an ID and `tabIndex={-1}` with `outline-none`.
