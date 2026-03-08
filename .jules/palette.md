## 2025-03-08 - Accessible drag-and-drop file inputs
**Learning:** Custom drag-and-drop zones built as `div` elements with `display: none` file inputs are completely hidden from screen readers and inaccessible via keyboard. Using a `label` element directly simplifies event handling and enables native focus styles.
**Action:** Always wrap file upload zones in `<label>` instead of `<div onClick>`, replace `hidden` with `sr-only` for the file input, and apply `focus-within:ring-X` to the outer label for keyboard focus indicators.
