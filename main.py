import os
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import clip
from torchvision import transforms
import pytesseract
import pymupdf as fitz
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import re
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

### DOCUMENT PROCESSING ###

# Load text embedding model
text_model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

def preprocess_text(text):
    """Enhance retrieval by applying TF-IDF weighting."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform([text]).toarray()
    return " ".join(vectorizer.get_feature_names_out())

# Directory setup
base_dir = "business_flowcharts"
pdf_dir = os.path.join(base_dir, "documents")

# Read and process PDFs
pdf_texts = {}
pdf_filenames = sorted([f for f in os.listdir(pdf_dir) if f.endswith(".pdf")])

pdf_embeddings = []
for file in pdf_filenames:
    pdf_path = os.path.join(pdf_dir, file)
    raw_text = extract_text_from_pdf(pdf_path)
    processed_text = preprocess_text(raw_text)
    pdf_texts[file] = raw_text
    embedding = text_model.encode(processed_text)
    pdf_embeddings.append(embedding)

# Convert to FAISS-compatible format
pdf_embeddings = np.array(pdf_embeddings, dtype="float32")
pdf_embeddings /= np.linalg.norm(pdf_embeddings, axis=1, keepdims=True)  # Normalize (IMPORTANT)

# Create FAISS index
text_index = faiss.IndexFlatL2(pdf_embeddings.shape[1])
text_index.add(pdf_embeddings)

print("✅ Stored document text embeddings in FAISS database.")



### FLOWCHART IMG PROCESSING ###

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def preprocess_image(image_path):
    """Preprocess an image dynamically while maintaining aspect ratio."""
    image = Image.open(image_path).convert("RGB")

    # Resize while keeping aspect ratio
    aspect_ratio = image.width / image.height
    if aspect_ratio > 1:
        new_width = 224
        new_height = int(224 / aspect_ratio)
    else:
        new_height = 224
        new_width = int(224 * aspect_ratio)

    transform = transforms.Compose([
        transforms.Resize((new_height, new_width)),  # Maintain aspect ratio
        transforms.Pad((0, 0, 224 - new_width, 224 - new_height), fill=(255, 255, 255)),  # Pad with white
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),
        # CLIP normalization (ALSO IMPORATNT)
    ])

    return transform(image).unsqueeze(0)


def get_image_embedding(image_path):
    """Generate an embedding for a flowchart image using CLIP."""
    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_tensor).cpu().numpy()
    return embedding.flatten()


# Directory setup
image_dir = os.path.join(base_dir, "flowcharts")
image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

# Process images
image_embeddings = [get_image_embedding(os.path.join(image_dir, file)) for file in image_filenames]
image_embeddings = np.array(image_embeddings, dtype="float32")
image_embeddings /= np.linalg.norm(image_embeddings, axis=1, keepdims=True)  # Normalize

# Create FAISS index
image_index = faiss.IndexFlatL2(image_embeddings.shape[1])
image_index.add(image_embeddings)

print("✅ Stored flowchart image embeddings in FAISS database.")




def extract_text_from_image(image_path):
    """Extract text from a given flowchart image using OCR."""
    image = Image.open(image_path).convert("RGB")
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text.strip()

def get_text_embedding(text):
    """Generate an embedding for extracted text using Sentence-BERT."""
    return text_model.encode(text)

print("Done")


def get_query_image_embedding(image_path):
    """Generate a normalized embedding for a query image using CLIP."""
    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        embedding = clip_model.encode_image(image_tensor).cpu().numpy()

    return embedding.flatten() / np.linalg.norm(embedding)  # Normalize


def retrieve_relevant_data(query, flowchart_imgs=None, user_provided_pdfs=None, top_k=2):
    """Retrieve the most relevant documents & images for the query using FAISS."""

    if flowchart_imgs and isinstance(flowchart_imgs, str):
        flowchart_imgs = [flowchart_imgs]

    if user_provided_pdfs and isinstance(user_provided_pdfs, str):
        user_provided_pdfs = [user_provided_pdfs]

    # Convert query to text embedding
    query_embedding = text_model.encode(query).reshape(1, -1)
    query_embedding /= np.linalg.norm(query_embedding)

    # Search FAISS text database
    text_distances, text_results = text_index.search(query_embedding, top_k)
    retrieved_pdfs = [(pdf_filenames[idx], text_distances[0][i]) for i, idx in enumerate(text_results[0])]

    retrieved_images = []
    image_scores = []
    unknown_images = []
    unknown_pdfs = []

    indexed_flowcharts = set(image_filenames)
    indexed_pdfs = set(pdf_filenames)

    # Handle flowchart images
    if flowchart_imgs:
        for flowchart_img in flowchart_imgs:
            if os.path.basename(flowchart_img) not in indexed_flowcharts:
                print(f"🆕 Marking '{flowchart_img}' as [NEW FLOWCHART] (not in FAISS index)")
                unknown_images.append(flowchart_img)
                continue

            extracted_text = extract_text_from_image(flowchart_img)
            text_embedding = get_text_embedding(extracted_text)

            text_distances, text_results = text_index.search(text_embedding.reshape(1, -1), top_k)
            retrieved_pdfs += [(pdf_filenames[idx], text_distances[0][i]) for i, idx in enumerate(text_results[0])]

            image_query_embedding = get_query_image_embedding(flowchart_img)
            image_distances, image_results = image_index.search(image_query_embedding.reshape(1, -1), top_k)

            if image_results[0][0] >= 0:
                retrieved_images += [image_filenames[idx] for idx in image_results[0]]
                image_scores += list(image_distances[0])
            else:
                unknown_images.append(flowchart_img)

    # Handle user-provided PDFs
    if user_provided_pdfs:
        for user_pdf in user_provided_pdfs:
            if os.path.basename(user_pdf) not in indexed_pdfs:
                print(f"🆕 Marking '{user_pdf}' as [NEW PDF] (not in FAISS index)")
                unknown_pdfs.append(user_pdf)
            else:
                retrieved_pdfs.append((os.path.basename(user_pdf), 0))  # Mark as exact match (0 distance)

    # Sort PDFs & images by FAISS similarity scores
    retrieved_pdfs = sorted(set(retrieved_pdfs), key=lambda x: x[1])[:top_k]
    retrieved_pdfs = [pdf for pdf, _ in retrieved_pdfs]

    image_sorted = sorted(zip(retrieved_images, image_scores), key=lambda x: x[1])[:top_k]
    retrieved_images = [img for img, _ in image_sorted]

    retrieved_images += [f"[NEW FLOWCHART] {img}" for img in unknown_images]
    retrieved_pdfs += [f"[NEW PDF] {pdf}" for pdf in unknown_pdfs]

    return retrieved_pdfs, retrieved_images


print("✅ Retrieval function V2 (With user PDF) is ready.")



# Load Qwen model
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")





def query_qwen_with_rag(query, flowchart_imgs=None, user_provided_pdfs=None, top_k=1):
    # Work related keywords are here so that the model don't automatically fetch documents for prompts that has nothing to do with work.
    # Feel free to comment this out should it be needed.
    work_related_keywords = [
        "flowchart", "document", "database", "pdf", "incident",
        "management", "network", "troubleshooting", "cybersecurity",
        "procedure", "process", "policy", "workflow", "information"
    ]

    # Check if query is work-related
    if not any(keyword in query.lower() for keyword in work_related_keywords):
        # If not work-related, skip retrieval and context
        text_input = qwen_processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": query}]}],
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = qwen_processor(
            text=[text_input],
            return_tensors="pt"
        ).to(qwen_model.device)

        with torch.no_grad():
            output_ids = qwen_model.generate(**inputs, max_new_tokens=1024)

        response_text = qwen_processor.batch_decode(output_ids, skip_special_tokens=False)[0]

        match = re.search(r"assistant\s*\n(.*)", response_text, re.DOTALL)
        cleaned_response = match.group(1).strip() if match else response_text.strip()

        return cleaned_response, [], []

    # Otherwise, proceed normally
    if flowchart_imgs and isinstance(flowchart_imgs, str):
        flowchart_imgs = [flowchart_imgs]

    if user_provided_pdfs and isinstance(user_provided_pdfs, str):
        user_provided_pdfs = [user_provided_pdfs]

    retrieved_pdfs, retrieved_images = retrieve_relevant_data(
        query=query,
        flowchart_imgs=flowchart_imgs,
        user_provided_pdfs=user_provided_pdfs,
        top_k=top_k
    )

    # Extract text from retrieved PDFs, ignoring "[NEW PDF]" entries
    context = "\n".join([pdf_texts[pdf] for pdf in retrieved_pdfs if not pdf.startswith("[NEW PDF]")])

    # Extract and explicitly include text from user-provided PDFs
    user_pdf_texts = []
    descriptions = []

    if user_provided_pdfs:
        for user_pdf in user_provided_pdfs:
            is_new_pdf = os.path.basename(user_pdf) not in pdf_filenames

            if is_new_pdf:
                descriptions.append(f"🆕 This is a newly provided PDF: {os.path.basename(user_pdf)}")
                extracted_text = extract_text_from_pdf(user_pdf)
                user_pdf_texts.append(
                    f"[NEW PDF] User-Provided PDF '{os.path.basename(user_pdf)}':\n\n{extracted_text}\n\n"
                    "[NOTE: This PDF is explicitly NOT part of the existing database.]"
                )
            else:
                extracted_text = pdf_texts[os.path.basename(user_pdf)]
                user_pdf_texts.append(f"User-Provided PDF '{os.path.basename(user_pdf)}':\n\n{extracted_text}")

    user_pdf_section = "\n\n".join(user_pdf_texts)

    # Extract flowchart texts (unchanged)
    flowchart_texts = []
    valid_images = []

    if flowchart_imgs:
        for img_path in flowchart_imgs:
            is_new_flowchart = os.path.basename(img_path) not in image_filenames

            if is_new_flowchart:
                descriptions.append(f"🆕 This is a newly provided flowchart: {os.path.basename(img_path)}")
                extracted_text = extract_text_from_image(img_path)
                flowchart_texts.append(
                    f"[NEW FLOWCHART] Flowchart {len(flowchart_texts) + 1} (User-Provided):\n\n{extracted_text}\n\n"
                    "[NOTE: This flowchart is explicitly NOT part of the existing database.]"
                )
            else:
                extracted_text = extract_text_from_image(img_path)
                flowchart_texts.append(f"Flowchart {len(flowchart_texts) + 1}:\n\n{extracted_text}")

            valid_images.append(Image.open(img_path).convert("RGB"))

    flowchart_section = "\n\n".join(flowchart_texts)

    # PROMPT
    content = [
        *([{"type": "image", "image": img} for img in valid_images]),
        *([{"type": "text", "text": desc} for desc in descriptions]),
        {"type": "text", "text": (
            "Carefully review the provided context below:\n\n"
            f"Indexed Documents:\n{context}\n\n"
            f"User-Provided PDF Context:\n{user_pdf_section}\n\n"
            f"Flowchart Information:\n{flowchart_section}\n\n"
            "Important clarifications:\n"
            "- Documents labeled '[NEW PDF]' or '[NEW FLOWCHART]' are NOT part of the database.\n"
            "- Documents without these labels ARE part of the database.\n\n"
            "Your task:\n"
            "1. Clearly state whether the relevant document or flowchart is currently in the database.\n"
            "2. Provide a summary explaining its key contents and purpose, explicitly referencing the context provided. Pay attention if the user wants it brief or detailed.\n\n"
            f"Query: {query}"
        )},
    ]

    # Prepare Qwen inputs as before
    if not valid_images:
        text_input = qwen_processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = qwen_processor(
            text=[text_input],
            return_tensors="pt"
        ).to(qwen_model.device)
    else:
        text_input = qwen_processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = qwen_processor(
            text=[text_input],
            images=valid_images,
            padding=True,
            return_tensors="pt"
        ).to(qwen_model.device)

    if not hasattr(inputs, "input_ids"):
        print("⚠️ Error: Inputs are incorrectly formatted. Skipping generation.")
        return "Error: Invalid input formatting", retrieved_pdfs, retrieved_images

    with torch.no_grad():
        output_ids = qwen_model.generate(**inputs, max_new_tokens=1024)

    response_text = qwen_processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    match = re.search(r"<\|im_start\|>assistant(.*)<\|im_end\|>", response_text, re.DOTALL)
    cleaned_response = match.group(1).strip() if match else response_text.strip()

    return cleaned_response, retrieved_pdfs, retrieved_images


print("✅ Qwen RAG V2 (with user-provided PDFs) is ready.")



def query_qwen_text_only(user_query):
    start_time = time.time()

    # Query Qwen with RAG
    qwen_response_rag, retrieved_pdfs, retrieved_images = query_qwen_with_rag(user_query)

    end_time = time.time()
    execution_time = end_time - start_time

    # Print results
    print(f"\n🤖 Qwen's Response (With RAG):\n{qwen_response_rag}")
    print(f"🔍 Retrieved Documents: {retrieved_pdfs}")
    print(f"🖼️ Retrieved Flowcharts: {retrieved_images}")
    print(f"⏳ Execution Time: {execution_time:.2f} seconds")
    return qwen_response_rag


def query_qwen_with_files(user_query, user_flowchart_img=None, user_pdf=None):
    start_time = time.time()

    # Query Qwen with RAG
    qwen_response_rag, retrieved_pdfs, retrieved_images = query_qwen_with_rag(user_query, user_flowchart_img, user_pdf)

    end_time = time.time()
    execution_time = end_time - start_time

    # Print results
    print(f"\n🤖 Qwen's Response (With RAG):\n{qwen_response_rag}")
    print(f"🔍 Retrieved Documents: {retrieved_pdfs}")
    print(f"🖼️ Retrieved Flowcharts: {retrieved_images}")
    print(f"⏳ Execution Time: {execution_time:.2f} seconds")
    return qwen_response_rag


def compare_query_qwen_with_rag(query, flowchart_imgs=None, user_provided_pdfs=None, top_k=1):
    """Retrieve relevant document and flowchart data, and query Qwen for a comparison-focused response."""

    # Ensure inputs are in list format
    if flowchart_imgs and isinstance(flowchart_imgs, str):
        flowchart_imgs = [flowchart_imgs]
    if user_provided_pdfs and isinstance(user_provided_pdfs, str):
        user_provided_pdfs = [user_provided_pdfs]

    # Retrieve relevant PDFs and flowcharts
    retrieved_pdfs, retrieved_images = retrieve_relevant_data(
        query=query,
        flowchart_imgs=flowchart_imgs,
        user_provided_pdfs=user_provided_pdfs,
        top_k=top_k
    )

    # Extract context from indexed PDFs (exclude [NEW PDF] labels)
    context = "\n".join([pdf_texts[pdf] for pdf in retrieved_pdfs if not pdf.startswith("[NEW PDF]")])

    # Extract user-provided PDF text
    user_pdf_texts = []
    descriptions = []

    if user_provided_pdfs:
        for pdf_path in user_provided_pdfs:
            if not os.path.exists(pdf_path):
                print(f"⚠️ Warning: PDF '{pdf_path}' not found. Skipping.")
                continue

            is_new_pdf = os.path.basename(pdf_path) not in pdf_filenames
            extracted_text = extract_text_from_pdf(pdf_path)

            label = "[NEW PDF] " if is_new_pdf else ""
            note = "\n[NOTE: This PDF is not in the existing database.]" if is_new_pdf else ""
            user_pdf_texts.append(
                f"{label}User-Provided PDF '{os.path.basename(pdf_path)}':\n\n{extracted_text}{note}"
            )
            if is_new_pdf:
                descriptions.append(f"🆕 This is a newly provided PDF: {os.path.basename(pdf_path)}")

    user_pdf_section = "\n\n".join(user_pdf_texts)

    # Extract flowchart OCR text
    flowchart_texts = []
    valid_images = []

    if flowchart_imgs:
        for img_path in flowchart_imgs:
            is_new_flowchart = os.path.basename(img_path) not in image_filenames

            if not os.path.exists(img_path):
                print(f"⚠️ Warning: Image '{img_path}' not found. Skipping.")
                continue

            if is_new_flowchart:
                descriptions.append(f"🆕 This is a newly provided flowchart: {os.path.basename(img_path)}")
                extracted_text = extract_text_from_image(img_path)
                flowchart_texts.append(
                    f"[NEW FLOWCHART] Flowchart {len(flowchart_texts) + 1} (User-Provided):\n\n{extracted_text}\n\n"
                    "[NOTE: This flowchart is NOT in the existing database.]"
                )
            else:
                extracted_text = extract_text_from_image(img_path)
                flowchart_texts.append(f"Flowchart {len(flowchart_texts) + 1}:\n\n{extracted_text}")

            valid_images.append(Image.open(img_path).convert("RGB"))

    flowchart_section = "\n\n".join(flowchart_texts)

    # PROMPT
    content = [
        *([{"type": "image", "image": img} for img in valid_images]),
        *([{"type": "text", "text": desc} for desc in descriptions]),
        {"type": "text", "text": (
            "You are provided with the following information for comparison:\n\n"

            "### SECTION 1: Content from the Existing Database\n"
            "**Database PDFs:**\n"
            f"{context if context.strip() else '[No indexed documents retrieved]'}\n\n"

            "**Database Flowcharts (OCR Extracted Text):**\n"
            f"{flowchart_section if flowchart_section.strip() else '[No indexed flowcharts retrieved]'}\n\n"

            "### SECTION 2: User-Provided Content for Comparison\n"
            "**User PDFs:**\n"
            f"{user_pdf_section if user_pdf_section.strip() else '[No user PDFs provided]'}\n\n"

            "**User Flowcharts (OCR Extracted Text):**\n"
            f"{flowchart_section if flowchart_section.strip() else '[No user flowcharts provided]'}\n\n"

            "### Task:\n"
            "- Carefully compare the content in Section 2 (user-provided) against Section 1 (database).\n"
            "- If multiple flowcharts are provided, **compare the second one to the first**.\n"
            "- Highlight step-by-step differences, changes in structure, terminology, or flow.\n"
            "- Summarize each major difference clearly.\n\n"
            f"{query}"
        )}
    ]

    # Handle text-only queries if no images are involved
    if not valid_images:
        print("🔹 No images detected, processing as a pure text query.")
        text_input = qwen_processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = qwen_processor(
            text=[text_input],
            return_tensors="pt"
        ).to(qwen_model.device)
    else:
        text_input = qwen_processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = qwen_processor(
            text=[text_input],
            images=valid_images,
            padding=True,
            return_tensors="pt"
        ).to(qwen_model.device)

    # Sanity check before generating
    if not hasattr(inputs, "input_ids"):
        print("⚠️ Error: Inputs are incorrectly formatted. Skipping generation.")
        return "Error: Invalid input formatting", retrieved_pdfs, retrieved_images

    # Generate response
    with torch.no_grad():
        output_ids = qwen_model.generate(**inputs, max_new_tokens=1024)

    # Decode and clean response
    response_text = qwen_processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    match = re.search(r"<\|im_start\|>assistant(.*)<\|im_end\|>", response_text, re.DOTALL)
    cleaned_response = match.group(1).strip() if match else response_text.strip()

    return cleaned_response, retrieved_pdfs, retrieved_images


print("✅ Improved Comparison RAG is set!")

def query_qwen_comparison(user_query, user_flowchart_imgs=None, user_pdfs=None):
    start_time = time.time()

    qwen_response_rag, retrieved_pdfs, retrieved_images = compare_query_qwen_with_rag(
        query=user_query,
        flowchart_imgs=user_flowchart_imgs,
        user_provided_pdfs=user_pdfs
    )

    end_time = time.time()
    print(f"\n🤖 Qwen's Response (With RAG):\n{qwen_response_rag}")
    print(f"🔍 Retrieved Documents: {retrieved_pdfs}")
    print(f"🖼️ Retrieved Flowcharts: {retrieved_images}")
    print(f"⏳ Execution Time: {end_time - start_time:.2f} seconds")