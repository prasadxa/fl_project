## 2025-03-22 - Toggle Button Accessibility
**Learning:** For custom tab or toggle button groups (like Scan Type or Report Format selectors), visual classes like `bg-teal-600 text-white` are not enough for screen readers. Using `role="group"` on the container with a proper label (`aria-labelledby` or `aria-label`), and setting `aria-pressed={isActive}` on the buttons is critical for conveying the current selection state to assistive technologies. Keyboard focus styles (like `focus-visible:ring-2`) should also be applied to these buttons since they are interactive elements.
**Action:** Always verify that custom UI controls that act as radio buttons or tabs have appropriate ARIA roles (`group`, `radiogroup`, or `tablist`) and state attributes (`aria-pressed`, `aria-checked`, or `aria-selected`), along with clear visual focus indicators.

## 2024-08-05 - Skip-to-content links
**Learning:** Skip-to-content links are essential for keyboard accessibility but can display an unsightly default browser outline when the target container (e.g., `<main>`) receives focus.
**Action:** Always add `tabIndex="-1"` and `focus:outline-none` to the target container when implementing skip-to-content links using Tailwind to ensure programmatic focus is received properly without visual artifacts.
