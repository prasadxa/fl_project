## 2024-03-13 - File Upload Keyboard Accessibility
**Learning:** File upload dropzones created using `<div>` elements with hidden `<input type="file">` elements are completely inaccessible to keyboard users because `hidden` inputs cannot receive focus.
**Action:** Always wrap file upload dropzones in `<label>` elements and use `.sr-only` instead of `.hidden` on the input, paired with `:focus-within` styles on the wrapper to ensure visual focus indicators are shown to keyboard users.
