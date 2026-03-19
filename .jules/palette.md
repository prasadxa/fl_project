## 2025-05-18 - Keyboard Accessible File Drop Zones
**Learning:** Using `className="hidden"` on a file input inside a custom drop zone breaks keyboard navigation because screen readers and the tab sequence ignore `display: none` elements.
**Action:** Use a `<label>` wrapper instead of a `<div>`, change the input's class to `sr-only`, and apply `focus-within:ring-2` (or similar) to the label so it visually reflects focus. This pattern ensures the file input is navigable by keyboard while maintaining the custom drop zone styling.
