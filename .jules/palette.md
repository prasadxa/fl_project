## 2024-03-21 - Accessible File Dropzones
**Learning:** Using `<label>` with `sr-only` inputs instead of `hidden` ensures keyboard navigability for custom file dropzones, while applying `focus-within` to the label provides necessary visual focus indicators.
**Action:** Use this pattern instead of a `div` with an `onClick` ref trigger to ensure screen reader and keyboard accessibility natively.
