import os
import json
import time
import fitz
import requests
import subprocess
import re
import glob
import asyncio
import hashlib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_community.chat_models import ChatOpenAI
from fpdf import FPDF
from uuid import uuid4


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def safe_rename(src, dst, retries=10, wait=0.5):
    for i in range(retries):
        try:
            os.rename(src, dst)
            return
        except PermissionError:
            print(f"Retry {i + 1}: waiting to unlock file...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to rename {src} to {dst}")

class WebSearch:
    def __init__(self, rag, runner, num_results=3):
        self.rag = rag
        self.runner = runner
        self.num_results = num_results
        self.api_key = "BSAG1bUT6HmbrWMeg04gT41nPxfvW6_"

    def search_web(self, query: str) -> str:
        try:
            headers = {
                "X-Subscription-Token": self.api_key
            }
            params = {
                "q": query,
                "count": self.num_results
            }
            r = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
            results = data.get("web", {}).get("results", [])
            snippets = [res.get("description", "") for res in results if "description" in res]
            return "\n\n".join(f"[Web] {s}" for s in snippets)
        except Exception as e:
            print(f"[Web Search Error] {e}")
            return ""

    def run(self):
        print("\n[INFO] You have entered Web Search Mode.")
        while True:
            query = input("\nPlease enter your question (or type 'stop' to end): ").strip()
            if query.lower() == "stop":
                print("[INFO] Web Search completed. Moving on to next steps...\n")
                break

            web_context = self.search_web(query)

            if self.rag.is_new_topic(query):
                self.rag.dialog_history = []

            source_filter = self.runner.infer_source_filter(query)
            context = self.rag.retrieve_similar_chunks(query, source_filter=source_filter, include_web=False)
            context = web_context

            if context:
                prompt = self.rag.generate_prompt(query, context, use_history=False)
                answer = self.rag.generate_response(prompt)
                latency = self.rag.answer_latencies[-1] if self.rag.answer_latencies else 0.0
                print(f"[Latency] Response time: {latency:.2f} seconds\nAnswer:\n{answer}")

                self.rag.used_prompts.append(prompt)
                self.rag.dialog_history.append((query, answer))
                self.rag.last_summary = {"text": "", "summary": ""}

                evaluation = self.rag.evaluate_answer_with_llm(query, context, answer)
                self.rag.questions.append(query)
                self.rag.answers.append(answer)
                self.rag.evaluations.append(evaluation)
                print(f"[Debug] Recorded (WebSearch) Q{len(self.rag.questions)}: {query}")
                print("[INFO] Answer evaluated and recorded.")
            else:
                print("[WARN] No context found for the query.")
                self.rag.questions.append(query)
                self.rag.answers.append("No relevant content was retrieved.")
                self.rag.evaluations.append("The context was not found. Evaluation skipped.")
                print(f"[Debug] Recorded (WebSearch-Fallback) Q{len(self.rag.questions)}: {query}")


class MCP:
    def __init__(self, url: str, rag):
        self.url = url
        self.temp_path = "D:/Coding/MinerU/input/temp_download.pdf"
        self.title_name = ""
        self.final_path = ""
        self.md_path = ""
        self.rag = rag
        self.pdf_folder = rag.pdf_folder

    def download_pdf(self):
        try:
            r = requests.get(self.url, timeout=10)
            r.raise_for_status()
            with open(self.temp_path, 'wb') as f:
                f.write(r.content)
            print(f"PDF downloaded：{self.temp_path}")
        except Exception as e:
            print(f"PDF failed to download：{e}")
            exit(1)

    def extract_title(self):
        try:
            doc = fitz.open(self.temp_path)
            text = doc[0].get_text().strip()
            title_line = text.split('\n')[0]
            self.title_name = re.sub(r'[\\/*?:"<>|\\s]+', '_', title_line)[:50]
            self.final_path = f"D:/Coding/MinerU/input/{self.title_name}.pdf"
            doc.close()
            time.sleep(0.3)
            if os.path.exists(self.final_path):
                print(f"File already exists: {self.final_path}, skipping rename.")
            else:
                safe_rename(self.temp_path, self.final_path)
                print(f"Renamed and moved to: {self.final_path}")
        except Exception as e:
            print(f"Failed to process file: {e}")
            exit(1)

    def run_mineru(self):
        bat_path = "D:/Coding/MinerU/Run_MinerU_PDF_Process.bat"
        try:
            subprocess.call([bat_path])
            print("MinerU started successfully.")
        except Exception as e:
            print(f"MinerU failed to start: {e}")
            exit(1)

    def wait_for_md(self, timeout=60):
        print("Waiting for .md output file...")
        root_folder = "D:/Coding/MinerU/output/"
        pattern = os.path.join(root_folder, "**", "*.md")
        expected = self.title_name.lower().replace("_", "").replace(" ", "")
        matches = glob.glob(pattern, recursive=True)
        for match in matches:
            filename = os.path.basename(match).lower().replace("_", "").replace(" ", "")
            if expected in filename:
                self.md_path = match
                print(f"Pre-check: found MD file: {self.md_path}")
                return

        waited = 0
        while waited < timeout:
            matches = glob.glob(pattern, recursive=True)
            for match in matches:
                filename = os.path.basename(match).lower().replace("_", "").replace(" ", "")
                if expected in filename:
                    self.md_path = match
                    print(f"Found MD file: {self.md_path}")
                    return
            time.sleep(2)
            waited += 2

        raise TimeoutError(f"Timeout: No matching .md file found for {self.title_name}")

    def insert_into_rag(self):
        rag = self.rag
        text = rag.extract_text_from_md(self.final_path)
        chunks = rag.split_text(text)
        vectors, raw_chunks = rag.vectorize_chunks(chunks)

        from qdrant_client.models import PointStruct
        from uuid import uuid4
        filename = self.title_name + ".pdf"
        points = [
            PointStruct(
                id=uuid4().int >> 64,
                vector=vec,
                payload={"pdf": filename, "text": chunk, "source": "web"}
            )
            for vec, chunk in zip(vectors, raw_chunks)
        ]
        rag.pdf_chunks[filename] = raw_chunks
        rag.qdrant_client.upsert(collection_name=rag.collection_name, points=points)
        print(f"External document \"{filename}\" has been inserted into the vector database.")

        processed_file_path = r"C:/Users/11200/Desktop/RAG/processed_files.json"
        if os.path.exists(processed_file_path):
            with open(processed_file_path, "r", encoding="utf-8") as f:
                processed_data = json.load(f)
        else:
            processed_data = {}

        file_hash = self.get_file_hash(self.final_path)
        mtime = os.path.getmtime(self.final_path)
        processed_data[filename] = {
            "hash": file_hash,
            "mtime": mtime,
            "type": "pdf",
            "id_list": [point.id for point in points]
        }

        with open(processed_file_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2)

        return rag

    def get_file_hash(self, file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def insert_into_rag_external(self, rag):
        text = rag.extract_text_from_md(self.final_path)
        chunks = rag.split_text(text)
        vectors, raw_chunks = rag.vectorize_chunks(chunks)

        from qdrant_client.models import PointStruct
        from uuid import uuid4
        filename = self.title_name + ".pdf"
        points = [
            PointStruct(
                id=uuid4().int >> 64,
                vector=vec,
                payload={"pdf": filename, "text": chunk, "source": "web"}
            )
            for vec, chunk in zip(vectors, raw_chunks)
        ]
        rag.pdf_chunks[filename] = raw_chunks
        rag.qdrant_client.upsert(collection_name=rag.collection_name, points=points)
        print(f"[INFO] External PDF \"{filename}\" inserted into vector DB.")

    def run(self):
        self.download_pdf()
        self.extract_title()
        self.run_mineru()
        self.wait_for_md()
        return self.insert_into_rag()

class RAGSystem:
    def __init__(self, api_key: str, base_url: str, golden_context_path: str, eval_prompt_path: str, collection_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.embed_model = SentenceTransformer("BAAI/bge-large-en")
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.collection_name = collection_name
        self.rating_map = {"Excellent": 1.0, "Good": 0.8, "Fair": 0.6, "Bad": 0.4, "Worse": 0.2}
        self.weights = {"Relevance": 0.5, "Faithfulness": 0.2, "Completeness": 0.1, "Clarity": 0.1, "Conciseness": 0.1}
        self.pdf_folder = r"D:/Coding/MinerU/input"
        self.excel_folder = r"C:/Users/11200/Desktop/RAG/Excel"
        self.collection_name = collection_name
        self.pdf_chunks = {}
        self.pdf_vectors = {}
        self.questions = []
        self.answers = []
        self.answer_latencies = []
        self.evaluation_latencies = []
        self.evaluations = []
        self.websearch = WebSearch(self, runner=None)
        self.query_to_pdf = {}
        self.golden_contexts = self.load_golden_contexts(golden_context_path)
        self.eval_template = self.load_eval_template(eval_prompt_path)
        self.dialog_history = []
        self.last_summary = {"text": "", "summary": ""}
        self.used_prompts = []
        self.answer_time_total = 0.0
        self.evaluation_time_total = 0.0
        self.fake_eval_results = []
        self.pdf_chunks = {}
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )

        monitor = FileMonitor(
            pdf_folder=self.pdf_folder,
            excel_folder=self.excel_folder,
            qdrant_client=self.qdrant_client,
            collection_name=self.collection_name,
            embed_model=self.embed_model,
            processed_file_path=r"C:/Users/11200/Desktop/RAG/processed_files.json"
            )
        monitor.sync_all_files()

        self.load_all_pdfs()
        self.load_all_excels()

    def extract_text_from_excel(self, file_path: str) -> str:
        try:
            df = pd.read_excel(file_path, sheet_name=None)
            combined_text = ""
            for sheet_name, sheet in df.items():
                combined_text += f"[Sheet: {sheet_name}]\n"
                combined_text += sheet.to_string(index=False) + "\n\n"
            return combined_text.strip()
        except Exception as e:
            print(f"Failed to parse Excel file {file_path}: {e}")
            return ""

    def extract_text_from_md(self, file_path: str) -> str:
        base_name = os.path.splitext(os.path.basename(file_path))[0].lower().replace("_", "").replace(" ", "")
        pattern = os.path.join("D:/Coding/MinerU/output/", "**", "*.md")
        matches = glob.glob(pattern, recursive=True)

        for match in matches:
            match_name = os.path.basename(match).lower().replace("_", "").replace(" ", "")
            if base_name in match_name:
                try:
                    with open(match, "r", encoding="utf-8") as f:
                        print(f"Dynamically match to the MD file: {match}")
                        return f.read()
                except Exception as e:
                    print(f"Cannot read MD file: {e}")
                    return ""

        print(f"No matching MD file was found: {base_name}")
        return ""

    def load_all_pdfs(self):
        files = os.listdir(self.pdf_folder)
        for filename in files:
            full_path = os.path.join(self.pdf_folder, filename)
            if filename.endswith(".pdf"):
                text = self.extract_text_from_md(full_path)
                chunks = self.split_text(text)
                vectors, raw_chunks = self.vectorize_chunks(chunks)
                points = [
                    PointStruct(
                        id=uuid4().int >> 64,
                        vector=vec,
                        payload={
                            "pdf": filename,
                            "text": chunk,
                            "source": "pdf",
                            "modality": "document"
                        }
                    )
                    for vec, chunk in zip(vectors, raw_chunks)
                ]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                self.pdf_chunks[filename] = raw_chunks
                print(f"[INFO] Local PDF '{filename}' has been added to the vector DB.")


    def load_all_excels(self):
        files = os.listdir(self.excel_folder)
        for filename in files:
            if filename.endswith('.xls') or filename.endswith('.xlsx'):
                full_path = os.path.join(self.excel_folder, filename)
                text = self.extract_text_from_excel(full_path)
                if text:
                    chunks = self.split_text(text)
                    vectors, raw_chunks = self.vectorize_chunks(chunks)
                    points = [
                        PointStruct(
                            id=uuid4().int >> 64,
                            vector=vec,
                            payload={
                                "pdf": filename,
                                "text": chunk,
                                "source": "excel",
                                "modality": "table"
                            }
                        ) for vec, chunk in zip(vectors, raw_chunks)
                    ]
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    self.pdf_chunks[filename] = raw_chunks
                    print(f"Excel file '{filename}' loaded from Excel folder.")

    def load_golden_contexts(self, file_path):
        contexts = {}
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    key = (item["pdf"].strip(), item["query"].strip().lower())
                    contexts[key] = item["golden_context"]
                    self.query_to_pdf[item["query"].strip().lower()] = item["pdf"].strip()
        return contexts

    def load_eval_template(self, pdf_path):
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(pdf_path)
            if "{context}" in text and "{question}" in text and "{answer}" in text:
                return text
            else:
                raise ValueError("The evaluation prompt must contain {context}, {question}, and {answer} placeholders.")
        except Exception as e:
            print(f"Failed to load evaluation template: {e}")
            exit(1)

    def split_text(self, text: str, max_length=500) -> list:
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) < max_length:
                current += para + "\n\n"
            else:
                chunks.append(current.strip())
                current = para + "\n\n"
        if current:
            chunks.append(current.strip())
        return chunks

    def vectorize_chunks(self, chunks: list) -> tuple:
        vectors = self.embed_model.encode(chunks, convert_to_numpy=True)
        return vectors, chunks

    def vectorize_query(self, query: str):
        return self.embed_model.encode([query], convert_to_numpy=True)[0].tolist()

    def retrieve_similar_chunks(self, query: str, source_filter: str = None, include_web: bool = False):
        query_vector = self.vectorize_query(query)
        filter_condition = None
        if source_filter:
            filter_condition = {"must": [{"key": "source", "match": {"value": source_filter}}]}

        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=filter_condition,
            limit=10,
            with_payload=True
        )

        sheet_chunks = []
        file_chunks = []
        for hit in search_results:
            text = hit.payload.get("text", "")
            if text.startswith("[Sheet:"):
                sheet_chunks.append(text)
            else:
                file_chunks.append(text)

        if sheet_chunks:
            retrieved = "\n".join(sheet_chunks)
        else:
            retrieved = "\n".join(file_chunks)

        has_local = bool(sheet_chunks or file_chunks)

        web_chunks = []
        if include_web:
            web_snippets = self.websearch.search_web(query)
            if web_snippets:
                web_chunks = [s.strip() for s in web_snippets.split("\n\n") if s.strip()]
                retrieved += "\n" + "\n".join(web_chunks)
        has_web = bool(web_chunks)

        return retrieved, has_local, has_web

    def is_new_topic(self, query: str, threshold=0.8) -> bool:
        if not self.dialog_history:
            return False
        last_question = self.dialog_history[-1][0]
        vec1 = self.vectorize_query(query)
        vec2 = self.vectorize_query(last_question)
        similarity = cosine_similarity(vec1, vec2)
        print(f"[DEBUG] Topic similarity with last question: {similarity:.4f}")
        return similarity < threshold

    def is_history_too_long(self, max_tokens=3000) -> bool:
        history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.dialog_history])
        return len(history_text.split()) > max_tokens

    def summarize_dialogue(self, dialog_history: list) -> str:
        dialog_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in dialog_history])

        if self.last_summary["text"] == dialog_text:
            print("[INFO] Using cached summary for dialog history.")
            return self.last_summary["summary"]

        summary_prompt = f"""Summarize the following conversation into a concise background paragraph:

    {dialog_text}

    Summary:"""

        llm = ChatOpenAI(openai_api_key=self.api_key, openai_api_base=self.base_url, model_name="gpt-4")
        summary = llm.invoke(summary_prompt).content.strip()

        self.last_summary = {"text": dialog_text, "summary": summary}

        return summary

    def generate_prompt(self, query: str, context: str, use_history=True) -> str:
        history = ""
        if use_history and self.dialog_history:
            if self.is_history_too_long():
                print("[INFO] History too long. Using compressed summary.")
                history = self.summarize_dialogue(self.dialog_history)
            else:
                print("[INFO] Using raw history.")
                history = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.dialog_history])

        return f"""You are a helpful assistant. Use the following history and context to answer.

    History:
    {history}

    Context:
    {context}

    Current Question:
    {query}

    Answer:"""

    def generate_response(self, prompt: str, has_local: bool = False, has_web: bool = False) -> str:
        start = time.time()
        llm = ChatOpenAI(openai_api_key=self.api_key, openai_api_base=self.base_url, model_name="gpt-4")
        response = llm.invoke(prompt).content.strip()
        latency = time.time() - start
        self.answer_time_total += latency
        self.answer_latencies.append(latency)

        context_part = prompt.split("Context:", 1)[1].split("Current Question:")[0].strip()
        if not context_part:
            print("[NOTICE] Answer likely based on pre-trained knowledge (no context retrieved).")
        elif has_local:
            print("[NOTICE] Answer based on residual database records.")
        elif has_web:
            print("[NOTICE] Answer based on web search results.")
        else:
            print("[NOTICE] Answer based on unknown context source.")
        return response

    def clean_text_for_pdf(self, text):
        return text.encode('latin-1', errors='replace').decode('latin-1')

    def compute_weighted_score(self, llm_output: str) -> float:
        score = 0.0
        parsed_metrics = set()
        for line in llm_output.strip().splitlines():
            if ":" in line:
                metric, level = line.split(":", 1)
                metric = metric.strip().replace("-", "").replace("*", "").replace("#", "").strip()
                level = level.strip().replace("*", "").strip()
                if metric in self.weights and metric not in parsed_metrics:
                    score += 100 * self.weights[metric] * self.rating_map.get(level, 0)
                    parsed_metrics.add(metric)
        return round(score, 2)

    async def evaluate_answer_async(self, question, context, answer):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self.evaluate_answer_with_llm, question, context, answer
        )
        return (question, answer, result)

    def evaluate_answer_with_llm(self, question, context, answer):
        pdf = self.query_to_pdf.get(question.strip().lower(), "")
        golden = self.golden_contexts.get((pdf, question.strip().lower()))
        eval_context = golden if golden else context
        eval_prompt = self.eval_template.format(context=eval_context, question=question, answer=answer)
        start = time.time()
        for attempt in range(3):
            try:
                evaluator = ChatOpenAI(openai_api_key=self.api_key, openai_api_base=self.base_url, model_name="gpt-4")
                result = evaluator.invoke(eval_prompt).content.strip()
                latency = time.time() - start
                self.evaluation_time_total += latency
                self.evaluation_latencies.append(latency)
                score = self.compute_weighted_score(result)
                return self.clean_text_for_pdf(f"{result}\n\nWeighted Score: {score}")
            except Exception as e:
                print(f"[Retry {attempt + 1}] LLM evaluation failed: {e}")
                time.sleep(2)
        print("[ERROR] LLM evaluation failed after 3 retries.")
        return "Evaluation failed due to repeated API errors."

    def test_llm_evaluator(self, questions, fake_answer_path):
        try:
            with open(fake_answer_path, "r", encoding="utf-8") as f:
                fake_answers = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"Failed to load fake answer file: {e}")
            return

        if len(fake_answers) != len(questions):
            print("Number of fake answers does not match number of user questions.")
            return

        async def batch_evaluate():
            tasks = []
            for question, fake_answer in zip(questions, fake_answers):
                context = self.retrieve_similar_chunks(question, include_web=False)
                if not context:
                    print(f"No context found for question: {question}")
                    continue
                self.answer_latencies.append(0.0)

                await asyncio.sleep(0.12)

                task = self.evaluate_answer_async(question, context, fake_answer)
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            self.fake_eval_results.extend(results)

        asyncio.run(batch_evaluate())

    def save_to_pdf(self, questions, answers, evaluations, output_path, include_prompt=False):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Times", size=12)

        pdf.add_page()
        pdf.multi_cell(0, 10, f"Total Time Consumption in Generating Answers: {self.answer_time_total:.2f} seconds",
                       align='L')
        pdf.multi_cell(0, 10, f"Total Time Consumption in Evaluating Answers: {self.evaluation_time_total:.2f} seconds",
                       align='L')
        pdf.ln()
        pdf.set_font("Times", style='B', size=14)
        pdf.cell(0, 10, "LLM Evaluation Summary", ln=True, align='L')
        pdf.set_font("Times", size=12)
        pdf.ln()

        for idx, (q, a, e) in enumerate(zip(questions, answers, evaluations), 1):
            pdf.add_page()
            pdf.multi_cell(0, 10, self.clean_text_for_pdf(f"Evaluation {idx}:\n{e}"), align='L')
            pdf.ln()
            pdf.multi_cell(0, 10, self.clean_text_for_pdf(f"Question {idx}: {q}"), align='L')
            pdf.ln()
            pdf.multi_cell(0, 10, self.clean_text_for_pdf(f"Answer {idx}: {a}"), align='L')
            pdf.ln()

            if idx - 1 < len(self.answer_latencies):
                latency = self.answer_latencies[idx - 1]
                pdf.multi_cell(0, 10, f"Response Latency: {latency:.2f} seconds", align='L')
            else:
                pdf.multi_cell(0, 10, "Response Latency: N/A", align='L')

            if idx - 1 < len(self.evaluation_latencies):
                eval_latency = self.evaluation_latencies[idx - 1]
                pdf.multi_cell(w=0, h=10, txt=f"Evaluation Latency: {eval_latency:.2f} seconds", align='L')
            else:
                pdf.multi_cell(0, 10, "Evaluation Latency: N/A", align='L')

            if include_prompt:
                if idx - 1 < len(self.used_prompts):
                    prompt = self.used_prompts[idx - 1]
                    pdf.set_font("Times", style='I', size=10)
                    pdf.multi_cell(0, 8, self.clean_text_for_pdf(f"[Prompt used for Q{idx}]\n{prompt}"), align='L')
                    pdf.ln()
                else:
                    pdf.set_font("Times", style='I', size=10)
                    pdf.multi_cell(0, 8, f"[Prompt used for Q{idx}]\nPrompt was not generated due to missing context.",
                                   align='L')
                    pdf.ln()

        if self.fake_eval_results:
            pdf.add_page()
            pdf.set_font("Times", style='B', size=14)
            pdf.cell(0, 10, "LLM Evaluator Robustness Test Results", ln=True, align='L')
            pdf.set_font("Times", size=12)
            pdf.ln()
            for idx, (q, a, e) in enumerate(self.fake_eval_results, 1):
                pdf.multi_cell(0, 10, self.clean_text_for_pdf(f"[Fake Test {idx}] Question: {q}"), align='L')
                pdf.ln()
                pdf.multi_cell(0, 10, self.clean_text_for_pdf(f"Fake Answer: {a}"), align='L')
                pdf.ln()
                pdf.multi_cell(0, 10, self.clean_text_for_pdf(f"LLM Evaluation: {e}"), align='L')
                pdf.ln()

                lat_idx = len(self.answers) + idx - 1
                if lat_idx < len(self.answer_latencies):
                    latency = self.answer_latencies[lat_idx]
                    pdf.multi_cell(0, 10, f"Response Latency: {latency:.2f} seconds", align='L')
                else:
                    pdf.multi_cell(0, 10, "Response Latency: N/A", align='L')

                if lat_idx < len(self.evaluation_latencies):
                    eval_latency = self.evaluation_latencies[lat_idx]
                    pdf.multi_cell(0, 10, f"Evaluation Latency: {eval_latency:.2f} seconds", align='L')
                else:
                    pdf.multi_cell(0, 10, "Evaluation Latency: N/A", align='L')

        pdf.output(output_path)

    def compare_with_external_text(self, external_text: str, target_pdf: str):
        if target_pdf not in self.pdf_chunks:
            print(f"The target PDF was not found: {target_pdf}")
            return

        external_vector = self.vectorize_query(external_text)
        chunks = self.pdf_chunks[target_pdf]
        chunk_vectors, _ = self.vectorize_chunks(chunks)

        similarities = [cosine_similarity(external_vector, vec) for vec in chunk_vectors]
        top_k = sorted(zip(chunks, similarities), key=lambda x: x[1], reverse=True)[:3]

        print(f"\nThe paragraphs in the external text that are most relevant to {target_pdf}")
        for i, (chunk, score) in enumerate(top_k, 1):
            print(f"\n[Top {i}]similarity：{score:.4f}\n{chunk}\n")


class Runner:
    print("Welcome to the Multi-PDF RAG System.")
    api_key = "sk-tJdnxHf0PCDgY2fBRe19tcAmvyLiHnUSaO8EVypK1w12jHHM"
    base_url = "https://api.nuwaapi.com/v1"
    pdf_folder = "D:/Coding/MinerU/input/"
    golden_path = "C:/Users/11200/Desktop/RAG/Golden Context.json"
    eval_prompt_path = "C:/Users/11200/Desktop/RAG/eval_prompt.pdf"
    output_path = "C:/Users/11200/Desktop/RAG/Answer.pdf"

    def infer_source_filter(self, query: str) -> str:
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["table", "excel", "figure", "sheet", "xls", "xlsx"]):
            return "excel"
        else:
            return "pdf"

    def run(self):
        print("Welcome to the Multi-PDF RAG System.")

        qdrant_client = QdrantClient(host="localhost", port=6333)

        rag_args = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "golden_context_path": self.golden_path,
            "eval_prompt_path": self.eval_prompt_path,
            "collection_name": "rag_data"
        }

        rag = RAGSystem(**rag_args)

        websearch_used = False
        enable_web = input("Enable temporary Web Search mode? (Yes/No): ").strip().lower()
        if enable_web == "yes":
            websearch_used = True
            websearch = WebSearch(rag, self)
            websearch.run()

        url = input("Please enter the external PDF URL (or type 'skip' to skip):\n> ").strip()
        if url.lower() != "skip":
            mcp = MCP(url, rag)
            rag = mcp.run()

        print("\n[INFO] Now you can continue asking questions (type 'stop' to end):")
        while True:
            query = input("Your question: ").strip()
            if query.lower() == "stop":
                break
            if not query:
                print("[WARN] Empty query. Skipping...")
                continue

            if rag.is_new_topic(query):
                rag.dialog_history = []

            source_filter = self.infer_source_filter(query)
            context, has_local, has_web = rag.retrieve_similar_chunks(query, source_filter=source_filter)

            if context:
                prompt = rag.generate_prompt(query, context)
                answer = rag.generate_response(prompt, has_local=has_local, has_web=has_web)
                latency = rag.answer_latencies[-1]
                print(f"[Latency] Response time: {latency:.2f}s\nAnswer:\n{answer}")

                evaluation = rag.evaluate_answer_with_llm(query, context, answer)
                rag.used_prompts.append(prompt)
                rag.questions.append(query)
                rag.answers.append(answer)
                rag.evaluations.append(evaluation)
            else:
                print("[WARN] No context retrieved.")
                rag.questions.append(query)
                rag.answers.append("No context retrieved.")
                rag.evaluations.append("Evaluation skipped due to missing context.")

        print("\n[Debug] The following is the list of problems actually recorded by the system:")
        for i, q in enumerate(rag.questions):
            print(f"Q{i + 1}: {q}")
        print(f"\n[Debug] A total of {len(rag.questions)} questions were recorded.")

        rag.test_llm_evaluator(rag.questions, "C:/Users/11200/Desktop/RAG/fake_answers.txt")

        rag.save_to_pdf(rag.questions, rag.answers, rag.evaluations, self.output_path, include_prompt=True)
        print(f"[INFO] All results saved to: {self.output_path}")

class FileMonitor:
    def __init__(self, pdf_folder, excel_folder, qdrant_client, collection_name, embed_model, processed_file_path):
        self.pdf_folder = pdf_folder
        self.excel_folder = excel_folder
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.processed_file_path = processed_file_path

    def get_file_hash(self, path):
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def get_excel_sheet_names(self, file_path):
        import pandas as pd
        try:
            xl = pd.ExcelFile(file_path)
            return xl.sheet_names
        except Exception as e:
            print(f"[Monitor] Failed to read sheets from {file_path}: {e}")
            return []

    def scan_current_files(self):
        def get_excel_sheet_hashes(file_path):
            sheet_hashes = {}
            try:
                xl = pd.ExcelFile(file_path)
                for sheet in xl.sheet_names:
                    df = xl.parse(sheet)
                    sheet_content = df.to_csv(index=False).encode('utf-8')
                    sheet_hash = hashlib.md5(sheet_content).hexdigest()
                    sheet_hashes[sheet] = {"hash": sheet_hash}
            except Exception as e:
                print(f"[Monitor] Failed to hash sheets from {file_path}: {e}")
            return sheet_hashes

        print(f"[Monitor] Scanning PDF folder: {self.pdf_folder}")
        print(f"[Monitor] Scanning Excel folder: {self.excel_folder}")
        files = {}
        for folder, exts, filetype in [(self.pdf_folder, ['.pdf'], 'pdf'),
                                       (self.excel_folder, ['.xls', '.xlsx'], 'excel')]:
            for filename in os.listdir(folder):
                if any(filename.endswith(ext) for ext in exts):
                    path = os.path.join(folder, filename)
                    hash_val = self.get_file_hash(path)
                    mtime = os.path.getmtime(path)
                    if filetype == 'pdf':
                        files[filename] = {"hash": hash_val, "mtime": mtime, "type": "pdf"}
                    else:
                        files[filename] = {
                            "hash": hash_val,
                            "mtime": mtime,
                            "type": "excel",
                            "sheets": get_excel_sheet_hashes(path)
                        }
        return files

    def delete_vectors_by_ids(self, id_list):
        from qdrant_client.models import PointIdsList
        if id_list:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=id_list)
            )

    def sync_all_files(self):
        processed_file_path = self.processed_file_path
        previous = {}
        if os.path.exists(processed_file_path):
            with open(processed_file_path, "r", encoding="utf-8") as f:
                previous = json.load(f)

        current = self.scan_current_files()
        to_add, to_update, to_delete = [], [], []

        for fname, meta in current.items():
            if fname not in previous:
                to_add.append(fname)
            elif previous[fname]["hash"] != meta["hash"]:
                to_update.append(fname)

        for fname in previous:
            if fname not in current:
                to_delete.append(fname)

        print(f"[Monitor] To Add: {to_add}, To Update: {to_update}, To Delete: {to_delete}")

        for fname in to_delete + to_update:
            ids = previous.get(fname, {}).get("id_list", [])
            self.delete_vectors_by_ids(ids)
            print(f"[Monitor] Deleted vectors for {fname}")

            sheets = previous.get(fname, {}).get("sheets", {})
            for sheet_key, sheet_data in sheets.items():
                ids = sheet_data.get("id_list", [])
                self.delete_vectors_by_ids(ids)
                print(f"[Monitor] Deleted vectors for Sheet Row: {sheet_key} in {fname}")

        for fname in to_add + to_update:
            folder = self.pdf_folder if fname.endswith('.pdf') else self.excel_folder
            path = os.path.join(folder, fname)
            if fname.endswith('.pdf'):
                doc = fitz.open(path)
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
                chunks = text.split('\n\n')
                vectors = self.embed_model.encode(chunks, convert_to_numpy=True)
                points = []
                ids = []
                for vec, chunk in zip(vectors, chunks):
                    point_id = uuid4().int >> 64
                    ids.append(point_id)
                    points.append(PointStruct(
                        id=point_id,
                        vector=vec,
                        payload={"pdf": fname, "text": chunk, "source": "pdf"}
                    ))
                self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
                current[fname]["id_list"] = ids
                current[fname]["type"] = "pdf"
                print(f"[Monitor] Added/Updated PDF {fname}")
            else:
                df = pd.read_excel(path, sheet_name=None)
                sheet_ids = {}
                points = []
                for sheet_name, sheet in df.items():
                    for idx, row in sheet.iterrows():
                        row_text = f"[Sheet: {sheet_name}] Row {idx + 1}: " + row.to_string(index=False)
                        vector = self.embed_model.encode([row_text], convert_to_numpy=True)[0]
                        point_id = uuid4().int >> 64
                        points.append(PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={"pdf": fname, "text": row_text, "source": "excel"}
                        ))
                        row_key = f"{sheet_name}_row_{idx + 1}"
                        sheet_ids[row_key] = {
                            "hash": hashlib.md5(row_text.encode("utf-8")).hexdigest(),
                            "id_list": [point_id]
                        }
                self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
                current[fname]["type"] = "excel"
                current[fname]["sheets"] = sheet_ids
                print(f"[Monitor] Added/Updated Excel {fname}")

        with open(processed_file_path, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        print("[Monitor] File sync complete.")

    def upsert_sheet_vectors(self, file_path, filename, sheet_name):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        text = f"[Sheet: {sheet_name}]\n" + df.to_string(index=False)
        vectors = self.embed_model.encode([text], convert_to_numpy=True)
        point_id = uuid4().int >> 64
        point = PointStruct(
            id=point_id,
            vector=vectors[0],
            payload={"pdf": filename, "sheet": sheet_name, "text": text, "source": "excel"}
        )
        self.qdrant_client.upsert(collection_name=self.collection_name, points=[point])
        print(f"[Monitor] Synced Sheet: {sheet_name} in {filename}")
        return point_id

    def delete_sheet_vectors(self, filename, sheet_name):
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector={
                "filter": {
                    "must": [
                        {"key": "pdf", "match": {"value": filename}},
                        {"key": "sheet", "match": {"value": sheet_name}}
                    ]
                }
            }
        )
        print(f"[Monitor] Deleted vectors for Sheet: {sheet_name} in {filename}")


if __name__ == '__main__':
    runner = Runner()
    runner.run()

