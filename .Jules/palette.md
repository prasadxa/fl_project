## 2024-05-15 - Accessible File Uploads with Visual Focus
**Learning:** Using a hidden `<input type="file">` wrapped in a `<div>` breaks keyboard accessibility for drag-and-drop zones because `hidden` or `display: none` elements cannot receive focus. Relying on clicking the parent div via JS is mouse-dependent.
**Action:** Always use a semantic `<label>` wrapping an `sr-only` input for custom file uploads. Apply `focus-within:ring-2` to the label to ensure screen readers can interact with the input and sighted keyboard users receive clear visual focus indicators.
