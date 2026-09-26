## 2025-03-22 - Toggle Button Accessibility
**Learning:** For custom tab or toggle button groups (like Scan Type or Report Format selectors), visual classes like `bg-teal-600 text-white` are not enough for screen readers. Using `role="group"` on the container with a proper label (`aria-labelledby` or `aria-label`), and setting `aria-pressed={isActive}` on the buttons is critical for conveying the current selection state to assistive technologies. Keyboard focus styles (like `focus-visible:ring-2`) should also be applied to these buttons since they are interactive elements.
**Action:** Always verify that custom UI controls that act as radio buttons or tabs have appropriate ARIA roles (`group`, `radiogroup`, or `tablist`) and state attributes (`aria-pressed`, `aria-checked`, or `aria-selected`), along with clear visual focus indicators.

## 2025-03-24 - Accessible Pagination Navigation
**Learning:** This app uses `<div>` wrappers for Prev/Next pagination buttons rather than native `<nav>` elements. Without semantic tags, screen reader users miss the context that these buttons control page navigation.
**Action:** Always wrap pagination controls in a container with `role="navigation"` and `aria-label="Pagination"`, and ensure the navigation buttons have clear visual focus states (e.g., `focus-visible:ring-2`).
