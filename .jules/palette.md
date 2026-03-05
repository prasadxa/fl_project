## 2025-03-05 - Add Keyboard Accessibility to File Upload Dropzone
**Learning:** Found a missing `role="button"` and keyboard navigation attributes (`tabIndex`, `onKeyDown`) on the main `div` functioning as a file upload trigger. This prevented keyboard-only users and screen readers from correctly engaging with the primary action.
**Action:** Always verify that custom interaction elements (like `div` acting as buttons) have `role="button"`, `tabIndex={0}`, and `onKeyDown` (Enter/Space) handlers, along with `aria-label`s.
