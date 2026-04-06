## 2025-03-22 - Toggle Button Accessibility
**Learning:** For custom tab or toggle button groups (like Scan Type or Report Format selectors), visual classes like `bg-teal-600 text-white` are not enough for screen readers. Using `role="group"` on the container with a proper label (`aria-labelledby` or `aria-label`), and setting `aria-pressed={isActive}` on the buttons is critical for conveying the current selection state to assistive technologies. Keyboard focus styles (like `focus-visible:ring-2`) should also be applied to these buttons since they are interactive elements.
**Action:** Always verify that custom UI controls that act as radio buttons or tabs have appropriate ARIA roles (`group`, `radiogroup`, or `tablist`) and state attributes (`aria-pressed`, `aria-checked`, or `aria-selected`), along with clear visual focus indicators.

## 2024-04-06 - Add Skip to Main Content Link for Accessibility
**Learning:** When implementing a 'Skip to main content' link in a React SPA, ensure the target container (e.g., `<main>`) has `tabIndex={-1}` to allow it to receive programmatic focus, and use an appropriate class like `outline-none` to prevent an unwanted default focus ring from appearing when skipped to.
**Action:** Always add `tabIndex={-1}` and `outline-none` to the `<main>` element when adding a skip link.
