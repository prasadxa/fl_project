from playwright.sync_api import sync_playwright, expect

def test_skip_to_main_content():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('http://localhost:5173')

        # 1. Focus the "Skip to main content" link programmatically
        skip_link = page.locator('a:has-text("Skip to main content")')
        skip_link.focus()

        # Verify it's visible and focused
        expect(skip_link).to_be_visible()
        expect(skip_link).to_be_focused()
        page.screenshot(path='skip_link_focused.png')

        browser.close()

if __name__ == '__main__':
    test_skip_to_main_content()
