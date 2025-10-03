import os
import uuid
from typing import List, Dict, Any, Optional
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image
import io
import base64
from config import settings
from models.models import DocumentInfo, DocumentType

class DocumentProcessor:
    def __init__(self):
        self.uploaded_documents: Dict[str, DocumentInfo] = {}
        self.document_content: Dict[str, str] = {}
    
    async def process_uploaded_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process uploaded file and extract text content"""
        try:
            file_ext = os.path.splitext(filename)[1].lower()
            document_id = str(uuid.uuid4())
            
            # Determine document type
            doc_type = self._get_document_type(file_ext)
            
            # Extract text based on file type
            if doc_type == DocumentType.PDF:
                text_content = self._extract_pdf_text(file_content)
            elif doc_type == DocumentType.DOCX:
                text_content = self._extract_docx_text(file_content)
            elif doc_type == DocumentType.IMAGE:
                text_content = self._extract_image_text(file_content)
            elif doc_type == DocumentType.TXT:
                text_content = file_content.decode('utf-8')
            else:
                text_content = f"Unsupported file type: {file_ext}"
            
            # Create document info
            doc_info = DocumentInfo(
                id=document_id,
                name=filename,
                type=doc_type,
                upload_time=str(os.path.getctime(filename)) if os.path.exists(filename) else "unknown",
                size=len(file_content),
                content_summary=text_content[:200] + "..." if len(text_content) > 200 else text_content
            )
            
            # Store document
            self.uploaded_documents[document_id] = doc_info
            self.document_content[document_id] = text_content
            
            return {
                "success": True,
                "document_id": document_id,
                "filename": filename,
                "document_type": doc_type,
                "content_preview": text_content[:500],
                "message": f"Successfully processed {filename}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "document_id": None,
                "filename": filename,
                "error": str(e)
            }
    
    def _get_document_type(self, file_ext: str) -> DocumentType:
        """Map file extension to document type"""
        type_map = {
            '.pdf': DocumentType.PDF,
            '.txt': DocumentType.TXT,
            '.docx': DocumentType.DOCX,
            '.doc': DocumentType.DOCX,
            '.png': DocumentType.IMAGE,
            '.jpg': DocumentType.IMAGE,
            '.jpeg': DocumentType.IMAGE,
            '.pptx': DocumentType.PPTX,
            '.xlsx': DocumentType.XLSX
        }
        return type_map.get(file_ext, DocumentType.TXT)
    
    def _extract_pdf_text(self, file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_file = io.BytesIO(file_content)
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error extracting PDF text: {str(e)}"
    
    def _extract_docx_text(self, file_content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            doc_file = io.BytesIO(file_content)
            doc = Document(doc_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"Error extracting DOCX text: {str(e)}"
    
    def _extract_image_text(self, file_content: bytes) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(io.BytesIO(file_content))
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            return f"Error extracting image text: {str(e)}"
    
    def get_document_content(self, document_ids: List[str]) -> List[str]:
        """Get content for specific document IDs"""
        contents = []
        for doc_id in document_ids:
            if doc_id in self.document_content:
                contents.append(self.document_content[doc_id])
        return contents
    
    def get_all_documents(self) -> List[DocumentInfo]:
        """Get list of all uploaded documents"""
        return list(self.uploaded_documents.values())
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document"""
        if document_id in self.uploaded_documents:
            del self.uploaded_documents[document_id]
        if document_id in self.document_content:
            del self.document_content[document_id]
        return True