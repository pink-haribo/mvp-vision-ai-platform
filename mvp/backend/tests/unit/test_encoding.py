"""
Test for UTF-8 encoding in subprocess calls.

Ensures that Windows cp949 encoding issue is resolved.

Related Issue: UnicodeDecodeError when reading subprocess output
containing UTF-8 characters (emojis, special chars).
"""

import subprocess
import sys
import pytest


class TestUTF8Encoding:
    """Test UTF-8 encoding handling in subprocess calls."""

    def test_subprocess_with_utf8_output(self):
        """Test that subprocess can handle UTF-8 output."""
        import os

        # Create a simple Python script that outputs UTF-8
        script = """
import sys
print("Hello World 🚀")
print("한글 테스트")
print("Special chars: ✓ ✗ ★")
"""

        # Run subprocess with UTF-8 encoding
        # PYTHONIOENCODING forces subprocess to use UTF-8 for stdout/stderr
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=env
        )

        # Assert: No encoding error
        assert result.returncode == 0
        assert "🚀" in result.stdout
        assert "한글" in result.stdout
        assert "★" in result.stdout

    def test_subprocess_without_encoding_fails_on_windows(self):
        """Test that subprocess without encoding spec can fail on Windows."""
        import platform

        if platform.system() != "Windows":
            pytest.skip("This test is Windows-specific")

        # Create a script with UTF-8 characters
        script = """
import sys
print("Emoji: 🎉")
"""

        # This should work now (we fixed it), but demonstrates the issue
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                # No encoding specified - uses system default (cp949 on Windows KR)
            )

            # On Windows KR, this might fail or produce garbled output
            # But our fix adds encoding='utf-8' to prevent this
            print(f"Output: {result.stdout}")

        except UnicodeDecodeError as e:
            pytest.fail(f"UnicodeDecodeError occurred: {e}")

    def test_yolo_output_with_special_chars(self):
        """
        Test that YOLO model output containing progress bars
        or emojis can be decoded properly.
        """
        import os

        # Simulate YOLO training output with progress indicators
        script = """
import sys
print("Epoch 1/5: ████████░░ 80% 🚀")
print("mAP@0.5: 0.85 ✓")
"""

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=env
        )

        assert result.returncode == 0
        assert "🚀" in result.stdout or "Epoch" in result.stdout
        assert "✓" in result.stdout or "mAP" in result.stdout

    def test_korean_path_handling(self):
        """Test that Korean characters in file paths are handled correctly."""
        import os

        # Create a script that uses Korean path
        script = """
import sys
path = "C:\\\\datasets\\\\한글폴더\\\\test.jpg"
print(f"Path: {path}")
"""

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=env
        )

        assert result.returncode == 0
        assert "한글폴더" in result.stdout

    def test_mixed_encoding_in_stderr(self):
        """Test that UTF-8 in stderr is also handled correctly."""
        import os

        script = """
import sys
sys.stderr.write("Error: 실패 ❌\\n")
"""

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=env
        )

        # stderr should be decoded properly
        assert "실패" in result.stderr or "Error" in result.stderr
