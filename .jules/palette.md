## 2024-05-18 - File Upload Accessibility Pattern
**Learning:** Using `hidden` on a file input completely removes it from the keyboard navigation flow.
**Action:** Always wrap file drop zones in a `<label>` tag, use `sr-only` on the `<input type="file">`, and apply `focus-within:ring` styles to the label so that keyboard users receive a clear visual focus indicator when the invisible input receives focus.
