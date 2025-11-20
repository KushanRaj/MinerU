"""
LaTeX to PDF Compilation Service

Standalone FastAPI service that compiles LaTeX documents to PDF.
Single endpoint that accepts LaTeX content and returns PDF bytes or error details.

Requirements:
    pip install fastapi uvicorn pydantic

System Requirements:
    - pdflatex must be installed (texlive-latex-base)
    - Required LaTeX packages: amsmath, amssymb, geometry, fancyhdr, graphicx, enumitem, tikz, pgfplots

Usage:
    uvicorn latex-service:app --host 0.0.0.0 --port 8001
"""

import base64
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LaTeX Compilation Service",
    description="Compiles LaTeX documents to PDF",
    version="1.0.0",
)


# Request/Response Models
class ImageFile(BaseModel):
    """Image file for inclusion in LaTeX document."""

    filename: str = Field(
        ..., description="Image filename (e.g., 'fig1.png', 'q5_fig0.png')"
    )
    data: str = Field(..., description="Base64-encoded image data")


class CompileRequest(BaseModel):
    """Request model for LaTeX compilation."""

    latex_content: str = Field(
        ...,
        description="LaTeX document content. Can be wrapped in markdown code fences or raw LaTeX.",
    )
    filename: str = Field(
        default="document",
        description="Base filename for the document (without extension)",
    )
    wrap_content: bool = Field(
        default=False,
        description="If true, wraps content in complete document structure with preamble",
    )
    title: str | None = Field(
        default=None, description="Document title (only used if wrap_content=True)"
    )
    images: List[ImageFile] = Field(
        default_factory=list,
        description="Optional images to include in compilation (referenced via \\includegraphics{filename})",
    )
    convert_markdown: bool = Field(
        default=False,
        description="If true, converts markdown syntax to LaTeX before compilation",
    )


class CompileErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Detailed error message")
    stdout: str | None = Field(None, description="LaTeX compilation stdout")
    stderr: str | None = Field(None, description="LaTeX compilation stderr")


class MarkdownToHtmlRequest(BaseModel):
    """Request model for Markdown to HTML conversion."""

    markdown_content: str = Field(
        ..., description="Markdown content to convert to HTML"
    )


class HtmlToMarkdownRequest(BaseModel):
    """Request model for HTML to Markdown conversion."""

    html_content: str = Field(..., description="HTML content to convert to Markdown")


class ConversionResponse(BaseModel):
    """Response model for conversion operations."""

    content: str = Field(..., description="Converted content")


# LaTeX Utilities
def strip_markdown_code_fence(content: str) -> str:
    r"""
    Extract LaTeX content from markdown code fences.

    Handles patterns like:
    ```latex
    \documentclass{article}
    ...
    ```

    Args:
        content: String that may contain markdown code fences

    Returns:
        Clean LaTeX content without code fences
    """
    # Pattern: ```latex or ```tex or ``` followed by content, then closing ```
    pattern = r"```(?:latex|tex)?\s*\n(.*?)\n```"
    match = re.search(pattern, content, re.DOTALL)

    if match:
        return match.group(1).strip()

    return content.strip()


def markdown_to_pdf_with_pandoc(
    md_content: str,
    filename: str = "document",
    title: str = "",
    images: List[ImageFile] = [],
) -> bytes:
    """
    Convert markdown directly to PDF using Pandoc.

    Args:
        md_content: Markdown content string
        filename: Base filename (not used, for API consistency)
        title: Document title
        images: List of images to include

    Returns:
        PDF bytes

    Raises:
        RuntimeError: If Pandoc conversion fails
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_pdf = temp_path / f"{filename}.pdf"

            # Save images to temp directory
            for image_file in images:
                image_path = temp_path / image_file.filename
                try:
                    image_bytes = base64.b64decode(image_file.data)
                    image_path.write_bytes(image_bytes)
                    logger.info(
                        f"Saved image: {image_file.filename} ({len(image_bytes)} bytes)"
                    )
                except Exception as e:
                    logger.error(f"Failed to save image {image_file.filename}: {e}")
                    raise RuntimeError(
                        f"Failed to save image {image_file.filename}: {str(e)}"
                    )

            # If title is provided explicitly, use it and keep markdown content as-is
            # If no title provided, don't extract or remove anything - let Pandoc handle the first heading naturally
            extracted_title = title if title else None
            logger.info(
                f"Title parameter: {repr(title)}, extracted_title: {repr(extracted_title)}"
            )

            # Write markdown to file as-is (Pandoc will use first # heading naturally without adding "Document")
            md_file = temp_path / f"{filename}.md"
            md_file.write_text(md_content, encoding="utf-8")

            # Use xelatex as primary engine (handles Unicode and title suppression better)
            # Fall back to pdflatex if xelatex fails
            pdf_engines = ["xelatex", "pdflatex"]
            last_error = None

            for engine in pdf_engines:
                # Build Pandoc command arguments
                pandoc_args = [
                    "pandoc",
                    str(md_file),
                    "-o",
                    str(output_pdf),
                    "--standalone",  # Required for complete document with all packages
                    f"--pdf-engine={engine}",
                    "-V",
                    "geometry:margin=2.5cm",
                    "-V",
                    "documentclass=article",
                    "-V",
                    "fontsize=12pt",
                ]

                # Only set title if explicitly provided
                if extracted_title:
                    logger.info(f"Adding title parameter: {repr(extracted_title)}")
                    pandoc_args.extend(["-V", f"title={extracted_title}"])
                else:
                    logger.info(
                        "No title parameter - letting Pandoc use first heading naturally"
                    )

                # Use Pandoc to convert markdown → PDF directly
                result = subprocess.run(
                    pandoc_args,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(temp_path),  # Set working directory for image resolution
                )

                if result.returncode == 0:
                    # Success! Break out of retry loop
                    logger.info(f"Pandoc PDF generation successful with {engine}")
                    break
                else:
                    last_error = result.stderr
                    # Check if it's a Unicode error
                    if "Unicode character" in result.stderr and engine == "pdflatex":
                        logger.warning(
                            "Unicode error with pdflatex, retrying with xelatex"
                        )
                        continue
                    else:
                        # Non-Unicode error or already tried xelatex, fail immediately
                        logger.error(
                            f"Pandoc PDF generation failed with {engine}: {result.stderr}"
                        )
                        raise RuntimeError(
                            f"Pandoc PDF generation failed: {result.stderr}"
                        )

            # If we exhausted all engines
            if result.returncode != 0:
                logger.error(
                    f"Pandoc PDF generation failed with all engines: {last_error}"
                )
                raise RuntimeError(f"Pandoc PDF generation failed: {last_error}")

            if not output_pdf.exists():
                raise RuntimeError("PDF was not generated by Pandoc")

            pdf_bytes = output_pdf.read_bytes()
            logger.info(f"Pandoc PDF generation successful ({len(pdf_bytes)} bytes)")

            return pdf_bytes

    except subprocess.TimeoutExpired:
        logger.error("Pandoc PDF generation timed out after 60 seconds")
        raise RuntimeError("Pandoc PDF generation timed out after 60 seconds")
    except FileNotFoundError:
        logger.error("Pandoc not found - please install pandoc")
        raise RuntimeError("Pandoc not installed. Install with: brew install pandoc")
    except Exception as e:
        logger.exception(f"Unexpected error during Pandoc PDF generation: {str(e)}")
        raise RuntimeError(f"Pandoc PDF generation error: {str(e)}")


def convert_markdown_to_html(markdown_content: str) -> str:
    """
    Convert Markdown to HTML using Pandoc with KaTeX support.

    Args:
        markdown_content: Markdown content to convert

    Returns:
        HTML body content (extracted from full document)

    Raises:
        RuntimeError: If Pandoc conversion fails
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        md_file = temp_path / "input.md"
        html_file = temp_path / "output.html"

        try:
            # Write Markdown content to file
            md_file.write_text(markdown_content, encoding="utf-8")

            logger.info("Starting Pandoc Markdown to HTML conversion")

            # Run pandoc to convert Markdown to HTML with KaTeX
            result = subprocess.run(
                [
                    "pandoc",
                    str(md_file),
                    "-o",
                    str(html_file),
                    "--katex",
                    "--standalone",
                    "-V",
                    "documentclass=article",
                    "-V",
                    "fontsize=12pt",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error("Pandoc conversion failed")
                raise RuntimeError(
                    f"Pandoc conversion failed: {result.stderr}",
                )

            # Read HTML output
            html_content = html_file.read_text(encoding="utf-8")

            # Extract body content (between <body> and </body> tags)
            import re

            body_match = re.search(r"<body[^>]*>(.*?)</body>", html_content, re.DOTALL)
            if body_match:
                body_content = body_match.group(1).strip()
                logger.info(f"Pandoc conversion successful: {len(body_content)} chars")
                return body_content
            else:
                logger.info("No body tags found, returning full HTML")
                return html_content

        except subprocess.TimeoutExpired:
            logger.error("Pandoc conversion timed out")
            raise RuntimeError("Pandoc conversion timed out after 30 seconds")
        except FileNotFoundError:
            logger.error("Pandoc not found - please install pandoc")
            raise RuntimeError(
                "Pandoc not installed. Install with: brew install pandoc"
            )
        except Exception as e:
            logger.exception(f"Unexpected error during Pandoc conversion: {str(e)}")
            raise RuntimeError(f"Pandoc conversion error: {str(e)}")


def convert_html_to_markdown(html_content: str) -> str:
    """
    Convert HTML to Markdown using Pandoc.

    Args:
        html_content: HTML content to convert

    Returns:
        Markdown content

    Raises:
        RuntimeError: If Pandoc conversion fails
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        html_file = temp_path / "input.html"
        md_file = temp_path / "output.md"

        try:
            # Write HTML content to file
            html_file.write_text(html_content, encoding="utf-8")

            logger.info("Starting Pandoc HTML to Markdown conversion")

            # Run pandoc to convert HTML to Markdown
            result = subprocess.run(
                ["pandoc", str(html_file), "-t", "markdown", "-o", str(md_file)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error("Pandoc conversion failed")
                raise RuntimeError(f"Pandoc conversion failed: {result.stderr}")

            # Read Markdown output
            markdown_content = md_file.read_text(encoding="utf-8")
            logger.info(f"Pandoc conversion successful: {len(markdown_content)} chars")

            return markdown_content

        except subprocess.TimeoutExpired:
            logger.error("Pandoc conversion timed out")
            raise RuntimeError("Pandoc conversion timed out after 30 seconds")
        except FileNotFoundError:
            logger.error("Pandoc not found - please install pandoc")
            raise RuntimeError(
                "Pandoc not installed. Install with: brew install pandoc"
            )
        except Exception as e:
            logger.exception(f"Unexpected error during Pandoc conversion: {str(e)}")
            raise RuntimeError(f"Pandoc conversion error: {str(e)}")


def wrap_latex_content(content: str, title: str = "Document") -> str:
    """
    Wrap LaTeX content in a complete document structure.

    Args:
        content: LaTeX content (body only)
        title: Document title

    Returns:
        Complete LaTeX document with preamble
    """
    return f"""\\documentclass[12pt,a4paper]{{article}}

% Packages
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{geometry}}
\\usepackage{{fancyhdr}}
\\usepackage{{graphicx}}
\\usepackage{{enumitem}}
\\usepackage{{longtable}}
\\usepackage{{booktabs}}
\\usepackage{{tikz}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}

% Page layout
\\geometry{{a4paper, margin=2.5cm}}
\\pagestyle{{fancy}}
\\fancyhf{{}}
\\rhead{{\\thepage}}
\\lhead{{{title}}}

% Title styling
\\title{{\\vspace{{-1cm}}\\textbf{{\\Large {title}}}}}
\\author{{}}
\\date{{}}

% Spacing
\\setlength{{\\parskip}}{{0.5em}}
\\setlength{{\\parindent}}{{0pt}}

\\begin{{document}}

\\maketitle
\\vspace{{-1cm}}

{content}

\\end{{document}}
"""


def compile_latex_to_pdf(
    latex_content: str, filename: str = "document", images: List[ImageFile] = []
) -> bytes:
    """
    Compile LaTeX content to PDF, optionally with images.

    Args:
        latex_content: LaTeX document content as string
        filename: Base filename for the document (without extension)
        images: List of images to save in temp directory (optional)

    Returns:
        PDF file as bytes

    Raises:
        RuntimeError: If LaTeX compilation fails with error details
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tex_file = temp_path / f"{filename}.tex"
        pdf_file = temp_path / f"{filename}.pdf"

        try:
            # Save images to temp directory
            for image_file in images:
                image_path = temp_path / image_file.filename
                try:
                    image_bytes = base64.b64decode(image_file.data)
                    image_path.write_bytes(image_bytes)
                    logger.info(
                        f"Saved image: {image_file.filename} ({len(image_bytes)} bytes)"
                    )
                except Exception as e:
                    logger.error(f"Failed to save image {image_file.filename}: {e}")
                    raise RuntimeError(
                        f"Failed to save image {image_file.filename}: {str(e)}",
                        {"stdout": "", "stderr": str(e)},
                    )

            # Write LaTeX content to file
            tex_file.write_text(latex_content, encoding="utf-8")

            logger.info(
                f"Starting LaTeX compilation for {filename} ({len(latex_content)} chars)"
            )

            # Run pdflatex twice to resolve references
            for run in range(2):
                result = subprocess.run(
                    [
                        "pdflatex",
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "-output-directory",
                        str(temp_path),
                        str(tex_file),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,  # 60 second timeout
                )

                if result.returncode != 0:
                    logger.error(f"LaTeX compilation failed on run {run + 1}")
                    error_data = {
                        "run": run + 1,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                    raise RuntimeError(
                        f"LaTeX compilation failed (run {run + 1})", error_data
                    )

            # Check if PDF was generated
            if not pdf_file.exists():
                raise RuntimeError(
                    "PDF file was not generated despite successful compilation",
                    {"stdout": "", "stderr": ""},
                )

            # Read PDF bytes
            pdf_bytes = pdf_file.read_bytes()
            logger.info(f"LaTeX compilation successful: {len(pdf_bytes)} bytes")

            return pdf_bytes

        except subprocess.TimeoutExpired:
            logger.error("LaTeX compilation timed out after 60 seconds")
            raise RuntimeError(
                "LaTeX compilation timed out after 60 seconds",
                {"stdout": "", "stderr": "Timeout"},
            )
        except RuntimeError:
            # Re-raise RuntimeError with compilation details
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during LaTeX compilation: {str(e)}")
            raise RuntimeError(
                f"LaTeX compilation error: {str(e)}", {"stdout": "", "stderr": str(e)}
            )


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "LaTeX Compilation Service",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Detailed health check with pdflatex availability."""
    try:
        result = subprocess.run(
            ["pdflatex", "--version"], capture_output=True, text=True, timeout=5
        )
        pdflatex_available = result.returncode == 0
        pdflatex_version = result.stdout.split("\n")[0] if pdflatex_available else None
    except Exception:
        pdflatex_available = False
        pdflatex_version = None

    return {
        "status": "healthy" if pdflatex_available else "degraded",
        "pdflatex_available": pdflatex_available,
        "pdflatex_version": pdflatex_version,
    }


@app.post(
    "/compile",
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "Successfully compiled PDF document",
        },
        400: {"model": CompileErrorResponse, "description": "LaTeX compilation error"},
        500: {"model": CompileErrorResponse, "description": "Internal server error"},
    },
)
async def compile_latex(request: CompileRequest):
    """
    Compile LaTeX document to PDF.

    Accepts LaTeX content and returns compiled PDF bytes.
    Handles markdown code fences and optional document wrapping.

    Args:
        request: CompileRequest with LaTeX content and options

    Returns:
        PDF document as binary response

    Raises:
        HTTPException: On compilation failure with error details
    """
    try:
        # Strip markdown code fences if present
        content = strip_markdown_code_fence(request.latex_content)

        # Convert markdown to PDF directly with Pandoc if requested
        if request.convert_markdown:
            logger.info("Converting markdown to PDF with Pandoc")
            title = request.title if request.title is not None else ""
            pdf_bytes = markdown_to_pdf_with_pandoc(
                content, request.filename, title, request.images
            )
        else:
            # LaTeX workflow
            # Wrap content if requested
            if request.wrap_content:
                title = request.title or "Document"
                content = wrap_latex_content(content, title)

            # Compile to PDF with optional images
            pdf_bytes = compile_latex_to_pdf(content, request.filename, request.images)

        # Return PDF as binary response
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{request.filename}.pdf"'
            },
        )

    except RuntimeError as e:
        # LaTeX compilation error
        error_msg = str(e.args[0]) if e.args else "Unknown compilation error"
        error_details = e.args[1] if len(e.args) > 1 else {}

        logger.error(f"Compilation error: {error_msg}")
        if error_details.get("stdout"):
            logger.error(f"LaTeX stdout:\n{error_details['stdout']}")
        if error_details.get("stderr"):
            logger.error(f"LaTeX stderr:\n{error_details['stderr']}")

        raise HTTPException(
            status_code=400,
            detail={
                "error": "compilation_error",
                "message": error_msg,
                "stdout": error_details.get("stdout", ""),  # Truncate to 1000 chars
                "stderr": error_details.get("stderr", ""),
            },
        )

    except Exception as e:
        # Unexpected error
        logger.exception(f"Unexpected error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": f"Internal server error: {str(e)}",
                "stdout": None,
                "stderr": None,
            },
        )


@app.post(
    "/markdown-to-html",
    response_model=ConversionResponse,
    responses={
        200: {
            "model": ConversionResponse,
            "description": "Successfully converted Markdown to HTML",
        },
        400: {"model": CompileErrorResponse, "description": "Conversion error"},
        500: {"model": CompileErrorResponse, "description": "Internal server error"},
    },
)
async def markdown_to_html(request: MarkdownToHtmlRequest):
    """
    Convert Markdown to HTML using Pandoc with KaTeX support.

    Args:
        request: MarkdownToHtmlRequest with Markdown content

    Returns:
        HTML body content (without wrapper)

    Raises:
        HTTPException: On conversion failure with error details
    """
    try:
        html_content = convert_markdown_to_html(request.markdown_content)
        return ConversionResponse(content=html_content)

    except RuntimeError as e:
        error_msg = str(e)
        logger.error(f"Conversion error: {error_msg}")

        raise HTTPException(
            status_code=400,
            detail={
                "error": "conversion_error",
                "message": error_msg,
                "stdout": "",
                "stderr": "",
            },
        )

    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": f"Internal server error: {str(e)}",
                "stdout": None,
                "stderr": None,
            },
        )


@app.post(
    "/html-to-markdown",
    response_model=ConversionResponse,
    responses={
        200: {
            "model": ConversionResponse,
            "description": "Successfully converted HTML to Markdown",
        },
        400: {"model": CompileErrorResponse, "description": "Conversion error"},
        500: {"model": CompileErrorResponse, "description": "Internal server error"},
    },
)
async def html_to_markdown(request: HtmlToMarkdownRequest):
    """
    Convert HTML to Markdown using Pandoc.

    Args:
        request: HtmlToMarkdownRequest with HTML content

    Returns:
        Markdown content

    Raises:
        HTTPException: On conversion failure with error details
    """
    try:
        markdown_content = convert_html_to_markdown(request.html_content)
        return ConversionResponse(content=markdown_content)

    except RuntimeError as e:
        error_msg = str(e)
        logger.error(f"Conversion error: {error_msg}")

        raise HTTPException(
            status_code=400,
            detail={
                "error": "conversion_error",
                "message": error_msg,
                "stdout": "",
                "stderr": "",
            },
        )

    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": f"Internal server error: {str(e)}",
                "stdout": None,
                "stderr": None,
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
