# 📖 MinerU API & Output Format Complete Guide

## 🎯 Quick Answers

### Does it generate Markdown by default?
**YES!** Markdown is the **primary output format** and is generated automatically.

### What are the supported outputs?
1. **Markdown file** (`.md`) - **Main output** ✅
2. **Content List JSON** (`_content_list.json`) - Structured data
3. **Middle JSON** (`_middle.json`) - Full parsing data
4. **Model JSON** (`_model.json`) - Raw model output
5. **Layout PDF** (`_layout.pdf`) - Visual debugging
6. **Images** (extracted as separate files)

---

## 📍 Documentation References

### 1. Output Files Documentation
**Location**: `/home/kushan/MinerU/docs/en/reference/output_files.md`

Key sections:
- Lines 1-11: Overview of all output files
- Lines 446-695: VLM backend specific outputs
- Lines 634-693: Content List format (what your API will return)

### 2. VLM Backend Code
**Location**: `/home/kushan/MinerU/mineru/backend/vlm/vlm_analyze.py`

Key function:
```python
# Line 132-156
def doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend="transformers",
    model_path: str | None = None,
    server_url: str | None = None,
    **kwargs,
):
    # Converts PDF to PIL images
    images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
    
    # Runs VLM model inference (this is where vLLM server is called)
    results = predictor.batch_two_step_extract(images=images_pil_list)
    
    # Converts model output to structured JSON
    middle_json = result_to_middle_json(results, images_list, pdf_doc, image_writer)
    return middle_json, results
```

### 3. Markdown Generation Code
**Location**: `/home/kushan/MinerU/mineru/backend/vlm/vlm_middle_json_mkcontent.py`

Key function:
```python
# Line 237-265
def union_make(pdf_info_dict: list,
               make_mode: str,
               img_buket_path: str = '',
               ):
    
    formula_enable = True  # Formulas enabled by default
    table_enable = True    # Tables enabled by default

    output_content = []
    for page_info in pdf_info_dict:
        paras_of_layout = page_info.get('para_blocks')
        
        if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
            # THIS GENERATES MARKDOWN
            page_markdown = mk_blocks_to_markdown(paras_of_layout, make_mode, 
                                                  formula_enable, table_enable, 
                                                  img_buket_path)
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.CONTENT_LIST:
            # THIS GENERATES STRUCTURED JSON
            for para_block in paras_of_layout:
                para_content = make_blocks_to_content_list(para_block, 
                                                           img_buket_path, 
                                                           page_idx, page_size)
                output_content.append(para_content)

    if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
        return '\n\n'.join(output_content)  # Returns Markdown string
    elif make_mode == MakeMode.CONTENT_LIST:
        return output_content  # Returns JSON list
```

---

## 🔄 How It Actually Works

### Processing Pipeline:

```
1. PDF Input
   │
   ├─> Convert to Images (PIL format)
   │   Location: mineru/utils/pdf_image_tools.py
   │
2. vLLM Model Inference
   │   Location: mineru_vl_utils (external package)
   │   Function: batch_two_step_extract()
   │   
   │   Step 1: Layout Detection
   │   ├─ Detects: text, titles, images, tables, equations, code, lists
   │   ├─ Returns: Bounding boxes + content types
   │   
   │   Step 2: Content Extraction
   │   ├─ Text: Direct recognition
   │   ├─ Formulas: Converts to LaTeX
   │   ├─ Tables: Converts to HTML
   │   ├─ Code: Extracts with syntax
   │   
3. Post-Processing
   │   Location: mineru/backend/vlm/model_output_to_middle_json.py
   │   
   ├─> Creates middle.json (structured data)
   │   ├─ Page info, blocks, lines, spans
   │   ├─ Merges cross-page tables
   │   ├─ Heading hierarchy (optional)
   │
4. Output Generation
   │   Location: mineru/backend/vlm/vlm_middle_json_mkcontent.py
   │   
   ├─> Markdown (.md)
   │   ├─ Formulas: wrapped in $$ delimiters
   │   ├─ Tables: HTML tables
   │   ├─ Images: ![alt](path)
   │   ├─ Code: ```language blocks
   │   
   ├─> Content List JSON
   │   ├─ Flat structure
   │   ├─ Reading order preserved
   │   ├─ Each block has type, content, bbox, page_idx
```

---

## 📋 Supported Content Types

Based on VLM backend (from `output_files.md` lines 457-481):

```python
{
  "text": "Plain text",
  "title": "Title",
  "equation": "Display (interline) formula",
  "image": "Image",
  "image_caption": "Image caption",
  "image_footnote": "Image footnote",
  "table": "Table",
  "table_caption": "Table caption",
  "table_footnote": "Table footnote",
  "phonetic": "Phonetic annotation",
  "code": "Code block",
  "code_caption": "Code caption",
  "ref_text": "Reference / citation entry",
  "algorithm": "Algorithm block (treated as code subtype)",
  "list": "List container",
  "header": "Page header",
  "footer": "Page footer",
  "page_number": "Page number",
  "aside_text": "Side / margin note",
  "page_footnote": "Page footnote"
}
```

---

## 📤 Output Formats in Detail

### 1. Markdown (.md) - **Primary Output**

**Generation**: Automatic (default)  
**Location**: Saved as `{filename}.md`

**Content includes:**
```markdown
# Title (h1)

## Heading 2

Regular text paragraph with **bold** and *italic*.

Inline equation: $E = mc^2$

Display equation:
$$
\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$

![Image caption](images/image_hash.jpg)

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |

```python
# Code block
def hello():
    print("world")
```
```

**How to get it via API:**
```python
from demo import parse_doc

parse_doc(
    path_list=["document.pdf"],
    output_dir="output/",
    backend="vlm-http-client",
    server_url="http://localhost:30000"
)

# Markdown saved at: output/document/document.md
with open("output/document/document.md") as f:
    markdown = f.read()
```

### 2. Content List JSON (`_content_list.json`)

**Generation**: Automatic  
**Location**: Saved as `{filename}_content_list.json`

**Structure** (from `output_files.md` lines 634-693):
```json
[
  {
    "type": "text",
    "text": "Document content...",
    "text_level": 1,
    "bbox": [62, 480, 946, 904],
    "page_idx": 0
  },
  {
    "type": "image",
    "img_path": "images/hash.jpg",
    "img_caption": ["Figure 1. Caption text"],
    "img_footnote": [],
    "bbox": [62, 480, 946, 904],
    "page_idx": 1
  },
  {
    "type": "equation",
    "text": "$$\nQ_{%} = f(P) + g(T)\n$$",
    "text_format": "latex",
    "bbox": [62, 480, 946, 904],
    "page_idx": 2
  },
  {
    "type": "table",
    "img_path": "images/table_hash.jpg",
    "table_caption": ["Table 1. Data"],
    "table_footnote": ["* indicates significance"],
    "table_body": "<html><body><table>...</table></body></html>",
    "bbox": [62, 480, 946, 904],
    "page_idx": 3
  },
  {
    "type": "code",
    "sub_type": "code",
    "code_body": "def hello():\n    print('world')",
    "code_caption": ["Listing 1. Example"],
    "bbox": [62, 480, 946, 904],
    "page_idx": 4
  },
  {
    "type": "list",
    "sub_type": "text",
    "list_items": [
      "Item 1",
      "Item 2",
      "Item 3"
    ],
    "bbox": [62, 480, 946, 904],
    "page_idx": 5
  }
]
```

**Key Features:**
- **Flat structure**: Easy to iterate
- **Reading order preserved**: Items in correct sequence
- **Bounding boxes**: 0-1000 normalized coordinates
- **Page indices**: Track which page (0-indexed)
- **All content types**: Text, images, tables, equations, code, lists

### 3. Middle JSON (`_middle.json`)

**Generation**: Automatic  
**Location**: Saved as `{filename}_middle.json`

**Purpose**: Full parsing data with hierarchy  
**Use case**: Advanced secondary development

**Structure** (from `output_files.md` lines 170-348):
```json
{
  "pdf_info": [
    {
      "page_idx": 0,
      "page_size": [612.0, 792.0],
      "para_blocks": [
        {
          "type": "text",
          "bbox": [52, 61.95, 294, 82.99],
          "lines": [
            {
              "bbox": [52, 61.95, 294, 72.00],
              "spans": [
                {
                  "bbox": [54.0, 61.95, 296.22, 72.00],
                  "content": "text content",
                  "type": "text",
                  "score": 1.0
                }
              ]
            }
          ]
        }
      ],
      "images": [],
      "tables": [],
      "interline_equations": [],
      "discarded_blocks": []
    }
  ],
  "_backend": "vlm",
  "_version_name": "2.5.4"
}
```

### 4. Model JSON (`_model.json`)

**Generation**: Optional  
**Location**: Saved as `{filename}_model.json`

**Purpose**: Raw model output before post-processing  
**Use case**: Debugging, model evaluation

---

## 🎛️ Output Control

### Enable/Disable Outputs

When calling `mineru` or `parse_doc()`, you can control outputs:

```python
from demo import do_parse

do_parse(
    output_dir="output/",
    pdf_file_names=["document"],
    pdf_bytes_list=[pdf_bytes],
    p_lang_list=["en"],
    backend="vlm-http-client",
    server_url="http://localhost:30000",
    
    # Control what gets generated
    f_dump_md=True,              # Generate Markdown ✅
    f_dump_content_list=True,    # Generate content_list.json ✅
    f_dump_middle_json=True,     # Generate middle.json ✅
    f_dump_model_output=True,    # Generate model.json ✅
    f_dump_orig_pdf=True,        # Save original PDF ✅
    f_draw_layout_bbox=True,     # Generate layout.pdf visualization ✅
    
    # Markdown output mode
    f_make_md_mode=MakeMode.MM_MD,  # Multimodal MD (images as links)
                                     # or MakeMode.NLP_MD (text-only)
    
    # Feature toggles
    formula_enable=True,   # Include formulas (default: True)
    table_enable=True,     # Include tables (default: True)
)
```

### Environment Variables

```bash
# Control VLM backend features
export MINERU_VLM_FORMULA_ENABLE=true   # Enable formula parsing
export MINERU_VLM_TABLE_ENABLE=true     # Enable table parsing
```

---

## 🔌 API Integration Examples

### Example 1: Get Markdown String

```python
from mineru.cli.common import read_fn
from mineru.backend.vlm.vlm_analyze import doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make
from mineru.utils.enum_class import MakeMode
from mineru.data.data_reader_writer import FileBasedDataWriter

# Read PDF
pdf_bytes = read_fn("document.pdf")

# Parse with vLLM server
middle_json, raw_results = doc_analyze(
    pdf_bytes=pdf_bytes,
    image_writer=FileBasedDataWriter("output/images"),
    backend="http-client",
    server_url="http://localhost:30000"
)

# Generate Markdown
markdown_string = union_make(
    pdf_info_dict=middle_json["pdf_info"],
    make_mode=MakeMode.MM_MD,
    img_buket_path="images"
)

print(markdown_string)
```

### Example 2: Get Structured JSON

```python
from mineru.utils.enum_class import MakeMode

# ... same setup as above ...

# Generate Content List
content_list = union_make(
    pdf_info_dict=middle_json["pdf_info"],
    make_mode=MakeMode.CONTENT_LIST,
    img_buket_path="images"
)

# Now you have a list of dicts
for item in content_list:
    if item["type"] == "text":
        print(f"Text: {item['text'][:50]}...")
    elif item["type"] == "image":
        print(f"Image: {item['img_path']}")
    elif item["type"] == "table":
        print(f"Table: {len(item['table_body'])} chars")
```

### Example 3: FastAPI Backend Integration

```python
from fastapi import FastAPI, File, UploadFile
from mineru.backend.vlm.vlm_analyze import doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make
from mineru.utils.enum_class import MakeMode
from mineru.data.data_reader_writer import FileBasedDataWriter
import tempfile
import os

app = FastAPI()

@app.post("/api/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    # Read uploaded file
    pdf_bytes = await file.read()
    
    # Create temp directory for images
    with tempfile.TemporaryDirectory() as tmpdir:
        image_writer = FileBasedDataWriter(f"{tmpdir}/images")
        
        # Parse PDF
        middle_json, _ = doc_analyze(
            pdf_bytes=pdf_bytes,
            image_writer=image_writer,
            backend="http-client",
            server_url="http://localhost:30000"
        )
        
        # Generate both formats
        markdown = union_make(
            middle_json["pdf_info"],
            MakeMode.MM_MD,
            "images"
        )
        
        content_list = union_make(
            middle_json["pdf_info"],
            MakeMode.CONTENT_LIST,
            "images"
        )
        
        return {
            "markdown": markdown,
            "content_list": content_list,
            "filename": file.filename
        }
```

---

## 📊 Output Format Comparison

| Format | Type | Size | Best For | Generated By Default |
|--------|------|------|----------|---------------------|
| `.md` | Text | Small | Human reading, docs | ✅ Yes |
| `_content_list.json` | JSON | Medium | APIs, apps | ✅ Yes |
| `_middle.json` | JSON | Large | Advanced dev | ✅ Yes |
| `_model.json` | JSON | Large | Debugging | ❌ Optional |
| `_layout.pdf` | PDF | Medium | Visual QA | ❌ Optional |
| `/images/*.jpg` | Binary | Varies | Extracted images | ✅ Yes |

---

## 🎯 Default Behavior Summary

### When you call MinerU:

1. **PDF → Images**: Automatic
2. **Images → vLLM**: Automatic (calls your server at localhost:30000)
3. **vLLM → Structured Data**: Automatic
4. **Structured Data → Markdown**: **Automatic** ✅
5. **Structured Data → JSON**: **Automatic** ✅

### You DON'T need to:
- Instruct it to generate markdown (it's default)
- Manually convert images
- Handle image extraction
- Process tables/formulas separately

### You CAN control:
- Output modes (multimodal vs text-only)
- Formula/table inclusion
- Which files to generate
- Formula delimiters ($$, $, custom)

---

## 🔗 Key Files to Reference

1. **Output documentation**: `docs/en/reference/output_files.md`
2. **VLM processing**: `mineru/backend/vlm/vlm_analyze.py`
3. **Output generation**: `mineru/backend/vlm/vlm_middle_json_mkcontent.py`
4. **Demo example**: `demo/demo.py`
5. **API example**: Create based on `backend_service.py` above

---

## ✅ Quick Test

Run this to see all outputs:

```bash
cd /home/kushan/MinerU
source .venv/bin/activate

mineru -p demo/pdfs/demo1.pdf \
  -o vllm_learning/outputs/test \
  -b vlm-http-client \
  -u http://localhost:30000

# Check outputs:
ls -la vllm_learning/outputs/test/demo1/
# You'll see:
# - demo1.md              (Markdown - PRIMARY OUTPUT)
# - demo1_content_list.json
# - demo1_middle.json
# - demo1_layout.pdf
# - images/
```



