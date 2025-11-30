import os
import json
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


class RAGGenerator:

    PROMPT_TEMPLATE = """
        You are an expert product advisor helping users choose the best option from retrieved e-commerce products.

        ## Instructions:
        1. First understand the user's intent.
        2. Then evaluate each product using these criteria:
        - Relevance to the user's request.
        - Key attributes present in the data (price, rating, brand).
        - Whether the product clearly satisfies the user needs.
        - Completeness and reliability of metadata.
        3. Select ONE best product and present it in this format:
        - Best Product: [Product Name] ([Product PID])
        - Why: [Explain in plain language why this product is the best fit, referring to specific 
                attributes like price, features, quality, or fit to user's needs.]
        4. If a second product is almost as good, include it as a quick alternative.
        5. If no product is relevant or matches the user's intent, output EXACTLY the following message:
           "There are no good products that fit the request based on the retrieved results."
        Do all reasoning internally; the final answer must follow the output format strictly.

        ## Retrieved Products:
        {retrieved_results}

        ## User Request:
        {user_query}

        ## Output Format:
        - Best Product: ... 
        - Why: ... 
        - Alternative (optional): ...
    """

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 20) -> dict:
        """
        Generate a response using the retrieved search results. 
        Returns:
            dict: Contains the generated suggestion and the quality evaluation.
        """
        DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."
        if not retrieved_results:
            return "There are no good products that fit the request based on the retrieved results."

        formatted_json = json.dumps([
            {
                "pid": d.pid,
                "title": d.title,
                "description": d.description,
                "brand": d.brand,
                "price": d.selling_price,
                "rating": d.average_rating
            }
            for d in retrieved_results[:top_N]
        ], indent=2)

        prompt = self.PROMPT_TEMPLATE.format(
            retrieved_results=formatted_json,
            user_query=user_query
        )

        try:
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name
            )

            return completion.choices[0].message.content

        except Exception as e:
            return f"Error in RAG generation: {e}"


            
