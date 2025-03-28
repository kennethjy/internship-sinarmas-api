from main import query_qwen_text_only, query_qwen_with_files, query_qwen_comparison
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


UPLOAD_FOLDER = "uploads"

# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/uploadFile")
async def upload_photo(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to the upload folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        if os.path.exists(file_path):
            i = 1
            while os.path.exists(file_path + f'({i})'):
                i += 1
            file_path = file_path + f'({i})'
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Return a JSON response with the URL of the uploaded photo
        return file_path
    except Exception as e:
        # Handle any errors that occur during file upload
        return e


@app.get("/query-qwen/text-only/{query}")
def query_qwen_text(query):
    response = 'query_qwen_text_only(query)'
    print(response)
    return {'response': response}


@app.get("/query-qwen/with_files/{query}")
def query_qwen_text(query, flowchartPath=None, pdfPath=None):
    response = 'query_qwen_with_files(query, flowchartPath, pdfPath)'
    return {'response': response}


@app.get("/query-qwen/comparison/{query}")
def query_qwen_text(query, flowchartPaths=None, pdfPaths=None):
    def toArray(filePaths):
        if len(filePaths) == 0:
            return None
        else:
            return filePaths.split(',')
    response = query_qwen_comparison(query, toArray(flowchartPaths), toArray(pdfPaths))
    return {'response': response}

