## 2025-03-22 - Toggle Button Accessibility
**Learning:** For custom tab or toggle button groups (like Scan Type or Report Format selectors), visual classes like `bg-teal-600 text-white` are not enough for screen readers. Using `role="group"` on the container with a proper label (`aria-labelledby` or `aria-label`), and setting `aria-pressed={isActive}` on the buttons is critical for conveying the current selection state to assistive technologies. Keyboard focus styles (like `focus-visible:ring-2`) should also be applied to these buttons since they are interactive elements.
**Action:** Always verify that custom UI controls that act as radio buttons or tabs have appropriate ARIA roles (`group`, `radiogroup`, or `tablist`) and state attributes (`aria-pressed`, `aria-checked`, or `aria-selected`), along with clear visual focus indicators.

## 2024-04-20 - Skip to main content in React SPAs
**Learning:** When implementing a 'Skip to main content' link in a React SPA with Tailwind, ensuring the target container (e.g., `<main>`) has `tabIndex={-1}` allows it to receive programmatic focus. Using `outline-none` on the target prevents an unwanted default focus ring from appearing when it is skipped to, keeping the visual experience clean.
**Action:** Always add `tabIndex={-1}` and `outline-none` to the target element when adding a 'Skip to main content' link.
