## 2024-03-11 - Accessible File Upload Dropzones
**Learning:** Using a simple `<div>` with `onClick={...click()}` and a `hidden` input for file dropzones breaks keyboard navigation and screen readers.
**Action:** Always use a `<label>` element that wraps the `<input type="file">`. Use the `sr-only` class on the input instead of `hidden` so it stays in the accessibility tree, add an `aria-label`, and add `focus-within:ring-2` to the wrapping label to correctly display visual focus when a keyboard user tabs to the hidden input.
