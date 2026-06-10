import json

from ollama import chat

from embeddings_util import (
    get_embedding,
    similarity
)

THRESHOLD = 300
TOP_K = 3


class FAQAgent:

    def __init__(self, faq_file):

        with open(faq_file, "r") as file:

            self.faqs = json.load(file)

        print("Generating FAQ embeddings...")

        for faq in self.faqs:

            faq["embedding"] = get_embedding(
                faq["question"]
            )

        print("FAQ embeddings ready.")

    def retrieve_top_k(self, user_question):

        user_embedding = get_embedding(
            user_question
        )

        results = []

        for faq in self.faqs:

            score = similarity(
                user_embedding,
                faq["embedding"]
            )

            results.append(
                {
                    "faq": faq,
                    "score": score
                }
            )

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:TOP_K]

    def build_context(self, retrieved_docs):

        context = ""

        for idx, item in enumerate(retrieved_docs, start=1):

            context += f"""
                        FAQ {idx}

                        Question:
                        {item['faq']['question']}

                        Answer:
                        {item['faq']['answer']}

                        ------------------
                        """

        return context

    def generate_answer(
        self,
        question,
        context
    ):

        response = chat(
            model="llama3",
            messages=[
                {
                    "role": "system",
                    "content": """
                                You are a helpful FAQ assistant.

                                Answer ONLY using the provided context.

                                If the answer is not present in the context,
                                reply exactly:

                                I don't have enough information to answer that.
                                """
                },
                {
                    "role": "user",
                    "content": f"""
                                Context:

                                {context}

                                Question:

                                {question}
                                """
                }
            ]
        )

        return response["message"]["content"]

    def ask(self, user_question):

        top_docs = self.retrieve_top_k(
            user_question
        )

        best_score = top_docs[0]["score"]

        if best_score < THRESHOLD:

            return {
                "answer":
                "I don't have enough information to answer that.",
                "score": best_score,
                "retrieved_docs": []
            }

        context = self.build_context(
            top_docs
        )

        answer = self.generate_answer(
            user_question,
            context
        )

        return {
            "answer": answer,
            "score": best_score,
            "retrieved_docs": top_docs
        }