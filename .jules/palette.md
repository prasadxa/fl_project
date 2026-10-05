## 2025-03-22 - Toggle Button Accessibility
**Learning:** For custom tab or toggle button groups (like Scan Type or Report Format selectors), visual classes like `bg-teal-600 text-white` are not enough for screen readers. Using `role="group"` on the container with a proper label (`aria-labelledby` or `aria-label`), and setting `aria-pressed={isActive}` on the buttons is critical for conveying the current selection state to assistive technologies. Keyboard focus styles (like `focus-visible:ring-2`) should also be applied to these buttons since they are interactive elements.
**Action:** Always verify that custom UI controls that act as radio buttons or tabs have appropriate ARIA roles (`group`, `radiogroup`, or `tablist`) and state attributes (`aria-pressed`, `aria-checked`, or `aria-selected`), along with clear visual focus indicators.

## 2025-10-24 - Pagination Accessibility
**Learning:** Pagination containers that use a generic `<div>` wrapper instead of a standard `<nav>` element are not automatically exposed to screen readers as navigation landmarks.
**Action:** Always add `role="navigation"` and an `aria-label="Pagination"` to generic `<div>` containers that wrap pagination controls (like Prev/Next buttons) to clearly identify their purpose to assistive technologies.
