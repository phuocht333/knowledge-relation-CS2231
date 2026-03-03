from google import genai

from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.qa.prompts import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE
from src.retrieval.retriever import HybridRetriever


class LegalQAAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def answer(self, question: str) -> dict:
        # Retrieve context
        retrieval = self.retriever.retrieve(question)

        # Build prompt
        user_prompt = QA_USER_TEMPLATE.format(
            context=retrieval["context"],
            question=question,
        )

        # Call Gemini
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=QA_SYSTEM_PROMPT,
            ),
        )

        answer_text = response.text

        return {
            "answer": answer_text,
            "cited_articles": retrieval["cited_articles"],
            "num_nodes": retrieval["num_nodes_retrieved"],
            "vector_results": retrieval["vector_results"],
        }
