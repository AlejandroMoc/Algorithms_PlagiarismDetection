import unittest
from unittest.mock import patch, mock_open
from package.scrap.grab import duplicate_to_clipboard, fetch_content, display, clipboard, inscribe


class TestScrapFunctions(unittest.TestCase):
    def setUp(self):
        """Prepare peculiar data for future tests"""
        self.sample_html = """
            <html>
                <body>
                    <textarea id="cl1pTextArea">Test content</textarea>
                </body>
            </html>
        """
        self.test_content = "Test content"

    @patch("subprocess.run")
    def test_duplicate_to_clipboard_linux(self, mock_run):
        # Simulate Linux environment for clipboard testing
        with patch("sys.platform", "linux"), patch(
            "shutil.which", return_value="/usr/bin/xclip"
        ):
            duplicate_to_clipboard("test content")

        mock_run.assert_called_once_with(
            ["xclip", "-selection", "clipboard"], input=b"test content", check=True
        )

    @patch("subprocess.run")
    def test_duplicate_to_clipboard_windows(self, mock_run):
        with patch("sys.platform", "win32"):
            duplicate_to_clipboard("test text")
            mock_run.assert_called_once_with(
                ["C:\\Windows\\System32\\clip.exe"], input=b"test text", check=True
            )

    @patch("subprocess.run")
    def test_duplicate_to_clipboard_macos(self, mock_run):
        with patch("sys.platform", "darwin"):
            duplicate_to_clipboard("test text")
            mock_run.assert_called_once_with(
                ["/usr/bin/pbcopy"], input=b"test text", check=True
            )

    @patch("builtins.print")
    @patch("package.scrap.grab.fetch_content", return_value="test content")
    def test_display(self, mock_fetch_content, mock_print):
        display("test-url")
        mock_print.assert_called_once_with("The content is: ", "test content")

    @patch("builtins.print")
    @patch("package.scrap.grab.fetch_content", return_value="test content")
    @patch("package.scrap.grab.duplicate_to_clipboard")
    def test_clipboard(self, mock_duplicate_to_clipboard, mock_fetch_content, mock_print):
        clipboard("test-url")
        mock_duplicate_to_clipboard.assert_called_once_with("test content")
        mock_print.assert_called_once_with("Content copied to clipboard.")

    @patch("builtins.open", new_callable=mock_open)
    @patch("package.scrap.grab.fetch_content", return_value="test content")
    def test_inscribe(self, mock_fetch_content, mock_open):
        inscribe("test-url", "test.txt")
        mock_open.assert_called_once_with("test.txt", "w")
        mock_open().write.assert_called_once_with("test content")


if __name__ == "__main__":
    unittest.main()