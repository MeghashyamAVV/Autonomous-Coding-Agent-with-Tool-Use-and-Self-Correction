import subprocess
import tempfile
import os


def execute_code(code: str, timeout: int = 20) -> dict:
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        success = result.returncode == 0
        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "success": success,
            "output": result.stdout.strip() if success else result.stderr.strip()
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Code execution timed out after {timeout} seconds.",
            "success": False,
            "output": f"Code execution timed out after {timeout} seconds."
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "success": False,
            "output": str(e)
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def extract_code_from_response(response: str) -> str:
    """
    Extracts Python code from LLM response.
    Handles ```python ... ``` and plain code responses.
    """
    if "```python" in response:
        start = response.find("```python") + len("```python")
        end = response.find("```", start)
        return response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        return response[start:end].strip()
    else:
        return response.strip()