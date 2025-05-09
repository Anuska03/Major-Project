import io
import os
import re
import json
import base64
import logging
from PIL import Image
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from langchain.schema import Document as LangDocument
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = OpenAIEmbeddings()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_log = logging.getLogger(__name__)

class HierarchicalHeaderSplitter:
    def __init__(self):
        self.header_pattern = r'^(#{1,6})\s+(.+)$'

    def split_text(self, text: str) -> List[Dict[str, str]]:
        lines = text.split('\n')
        chunks, current_chunk, header_stack = [], [], []
        in_code_block = False

        def finalize_chunk():
            nonlocal current_chunk
            if current_chunk:
                content = '\n'.join(current_chunk).strip()
                if content:
                    chunks.append({
                        'page_content': content,
                        'metadata': {'headers': header_stack.copy(), 'type': 'pdf_section'}
                    })
                current_chunk = []

        for line in lines:
            header_match = re.match(self.header_pattern, line.strip())
            is_code_start = line.strip().startswith('```')
            is_code_end = line.strip().endswith('```') and not line.strip() == '```'
            if is_code_start:
                in_code_block = True
            elif is_code_end:
                in_code_block = False
            if header_match and not in_code_block:
                finalize_chunk()
                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()
                header_stack = [h for h in header_stack if h['level'] < level]
                header_stack.append({'level': level, 'text': header_text})
            current_chunk.append(line)
        finalize_chunk()
        return chunks

class PdfIngestor:
    def __init__(self, file_path: str, user_id: str, session_id: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at {file_path}")
        self.file_path = file_path
        self.user_id = user_id
        self.session_id = session_id
        self.embedding_model = embedding_model
        self.client = client
        self.splitter = HierarchicalHeaderSplitter()
        self.image_resolution_scale = 2.0

    def query_gpt_with_image(self, image: Image.Image) -> str:
        with io.BytesIO() as buffer:
            image.save(buffer, format="JPEG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze and extract full structured data from the following chart/table."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    ],
                }
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content

    def replace_embedded_images_with_interpretations(self, markdown: str, interpretations: List[Dict]) -> str:
        pattern = r'!\[(.*?)\]\s*\(data:image/(?:png|jpeg|jpg|gif);base64,[^\)]+\)'
        def replacement(match):
            alt_text = match.group(1)
            match_start = match.start()
            for interp in interpretations:
                if interp["start"] == match_start:
                    return interp["text"]
            return f"[Image not interpreted: {alt_text}]"
        return re.sub(pattern, replacement, markdown, flags=re.MULTILINE)

    def ingest(self) -> str:
        user_folder = os.path.join("data", self.user_id, self.session_id)
        os.makedirs(user_folder, exist_ok=True)
        _log.info(f"Created user folder: {user_folder}")

        # Copy input PDF to internal location
        pdf_path = os.path.join(user_folder, "usertemp.pdf")
        import shutil
        shutil.copy(self.file_path, pdf_path)
        _log.info(f"Copied {self.file_path} to {pdf_path}")

        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = self.image_resolution_scale
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True

        doc_converter = DocumentConverter({
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=PyPdfiumDocumentBackend,
                pipeline_options=pipeline_options
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline
            )
        })

        try:
            conv_res = doc_converter.convert(pdf_path)
        except Exception as e:
            _log.error(f"Document conversion failed: {e}")
            raise

        output_dir = Path(user_folder) / "extracted_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        doc_filename = conv_res.input.file.stem
        md_filename = output_dir / f"{doc_filename}-with-images.md"
        conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)
        _log.info(f"Saved markdown to {md_filename}")

        with open(md_filename, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        image_pattern = r'!\[(.*?)\]\s*\(data:image/(?:png|jpeg|jpg|gif);base64,[^\)]+\)'
        image_refs = []
        for match in re.finditer(image_pattern, markdown_content, re.MULTILINE):
            base64_data = match.group(0).split('base64,')[-1][:-1]
            image_refs.append({
                "alt_text": match.group(1),
                "base64_data": base64_data,
                "start": match.start(),
                "end": match.end(),
            })

        interpretations = []
        for ref in image_refs:
            try:
                img = Image.open(io.BytesIO(base64.b64decode(ref["base64_data"]))).convert("RGB")
                response = self.query_gpt_with_image(img)
                if "Nothing detected" not in response:
                    interpretations.append({"start": ref["start"], "text": response})
            except Exception as e:
                _log.error(f"Image decode error: {e}")

        modified_md = self.replace_embedded_images_with_interpretations(markdown_content, interpretations)
        chunks = self.splitter.split_text(modified_md)
        documents = [LangDocument(page_content=c["page_content"], metadata=c["metadata"]) for c in chunks]

        parsed_path = os.path.join(user_folder, "parsed.txt")
        try:
            with open(parsed_path, "w", encoding="utf-8") as f:
                for doc in documents:
                    f.write("====CHUNK START====\n")
                    f.write(f"Content: {doc.page_content}\n")
                    f.write(f"Metadata: {json.dumps(doc.metadata)}\n")
                    f.write("====CHUNK END====\n")
            _log.info(f"Saved parsed.txt to {parsed_path}")
            if not os.path.exists(parsed_path):
                raise IOError(f"Failed to save parsed.txt at {parsed_path}")
        except Exception as e:
            _log.error(f"Failed to save parsed.txt: {e}")
            raise

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        batch_size = 32
        faiss_index = None
        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]
                embeddings = self.embedding_model.embed_documents(batch_texts)
                if faiss_index is None:
                    faiss_index = FAISS.from_texts(batch_texts, self.embedding_model, metadatas=batch_metas)
                else:
                    faiss_index.add_texts(batch_texts, metadatas=batch_metas)
            faiss_index.save_local(folder_path=os.path.join(user_folder, "faiss"))
            _log.info(f"Saved FAISS index to {user_folder}/faiss")
        except Exception as e:
            _log.error(f"Failed to save FAISS index: {e}")
            raise

        return parsed_path