## 2026-03-10 - File Upload Dropzone Accessibility
**Learning:** Implementing custom div-based file upload dropzones completely breaks accessibility unless explicitly handled. Screen readers can't identify the element, and keyboard users can't trigger the hidden input.
**Action:** When creating custom drag-and-drop file upload zones, always add `role="button"`, `tabIndex={0}`, `aria-label`, an `onKeyDown` handler (Enter/Space) to trigger the hidden file input, and visual focus states (e.g., `focus-visible:ring-2`).
