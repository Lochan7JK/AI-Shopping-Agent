import base64
import json
import os
import sqlite3
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from reviews_api import get_product_rating

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")

# llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)  # not working, maybe deprecated
# vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
vision_llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_products(query: str, max_price: Optional[float] = None, is_organic: Optional[bool] = None) -> str:
    """
    Search the product database by keyword (matched against name, description, and category).
    Optionally filter by maximum price and/or organic status.
    Returns a JSON array of matching products, each with: id, name, category, price,
    description, is_organic.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    sql = "SELECT id, name, category, price, description, is_organic FROM products WHERE 1=1"
    params: list = []

    # llm is an alternate to this "keywork search"/"substring matching" below:
    if query:
        sql += " AND (name LIKE ? OR description LIKE ? OR category LIKE ?)"
        like = f"%{query}%"
        params.extend([like, like, like])

    if max_price is not None:
        sql += " AND price <= ?"
        params.append(max_price)

    if is_organic is not None:
        sql += " AND is_organic = ?"
        params.append(1 if is_organic else 0)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    products = [
        {
            "id":          row[0],
            "name":        row[1],
            "category":    row[2],
            "price":       row[3],
            "description": row[4],
            "is_organic":  bool(row[5]),
        }
        for row in rows  # list comprehension used
    ]
    return json.dumps(products)


@tool
def get_rating(product_id: int) -> str:
    """
    Get the average customer rating and total review count for a product by its ID.
    Returns a JSON object with: product_id, average_rating, review_count.
    """
    result = get_product_rating(product_id) # in reviews_api.py
    return json.dumps(result)



# CHECKOUT -  just take the product record
#             and put it in store.db (No payment processing)
@tool
def checkout(product_id: int) -> str:  
    """
    Place an order for the given product ID. Saves the order to the database and returns
    a confirmation message with the order ID, product name, and price.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor() # select product record from products table
    cursor.execute("SELECT name, price FROM products WHERE id = ?", (product_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return f"Error: product with ID {product_id} not found."

    name, price = row
    cursor.execute( # put in the orders table
        "INSERT INTO orders (product_id, product_name, price) VALUES (?, ?, ?)",
        (product_id, name, price),
    )
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return (
        f"Order #{order_id} confirmed! '{name}' has been successfully ordered for ${price:.2f}. "
        f"Your order will arrive in 3-5 business days. Thank you for shopping with us!"
    )


@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyze a product image and return its key attributes as a JSON object.
    Use this when the user uploads a photo of a product they are interested in.
    The returned attributes can be used directly with search_products.
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}" 

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_data}"},
            # image string in base64 format
        },
        {
            "type": "text",
            "text": (
                "Look at this product image and extract its key attributes. "
                "Return ONLY a JSON object with these fields:\n"
                "- product_type: what kind of product it is (e.g. honey, olive oil, almonds)\n"
                "- search_query: a short keyword to search for it (e.g. 'honey', 'olive oil')\n"
                "- is_organic: true if the label says organic, false if not, null if unclear\n"
                "- description: one sentence describing the product"
            ),
        },
    ])

    response = vision_llm.invoke([message])
    return response.content


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = create_agent(
    tools=[search_products, get_rating, checkout, describe_product_image],
    model=llm,
    system_prompt=(
        "You are a helpful shopping assistant. Follow these rules strictly.\n\n"

        "IMAGE SEARCH — when the user provides an image path:\n"
        "1. Call describe_product_image with the path to identify the product.\n"
        "2. Use the returned search_query and is_organic to call search_products.\n"
        "3. Continue with the BROWSING flow from step 2 onwards.\n\n"

        "BROWSING — when the user describes what they want to buy:\n"
        "1. Call search_products to find matching items (apply any price/organic filters given).\n"
        "2. For each candidate, call get_rating to retrieve its average rating.\n"
        "3. Filter by the user's minimum rating if specified.\n"
        "4. Present qualifying products as a numbered list. For each item use this exact format "
        "   (plain text, no backticks, no code blocks, no bold, no italic):\n\n"
        "   #<number>. <name> (ID:<product_id>) — $<price> ★<rating> — <organic or non-organic>\n\n"
        "   Add a blank line between each product entry for readability. "
        "   Always include (ID:X) so you can reference it later.\n"
        "5. If only one product qualifies, still show it in the list and ask: "
        "   'Would you like to order it? Just say yes or give me the number.'\n"
        "6. Do NOT call checkout at this stage.\n\n"

        "ORDERING — when the user confirms they want to buy (e.g. 'yes', 'sure', 'go ahead', "
        "'order number 2', 'the first one', 'get me #3'):\n"
        "1. Look at your previous message to find the (ID:X) for the chosen product "
        "   (if only one was listed and the user says 'yes', use that product's ID).\n"
        "2. Call checkout with that product_id (the number from (ID:X)).\n"
        "3. Confirm the order to the user in plain text.\n\n"

        "Never place an order unless the user explicitly confirms. "
        "Never guess a product_id — always take it from the (ID:X) in your own previous message."
    ),
)

### SEARCH PRODUCT FUNC TEST
# if __name__ == "__main__":
#     # result = search_products("honey", max_price=20, is_organic=True) 
#     # print(result)
#     # not working because of @tool

#     result = search_products.invoke({
#         "query": "honey",
#         "max_price": 20,
#         "is_organic": True
#     })
#     print(result)
   

### PRODUCTT IMAGE FUNC TEST
# if __name__ == "__main__":
    # describe_product_image("resources/honey.png")  OR
    # image_path = os.path.join("resources", "honey.png")
    # response = describe_product_image(image_path)
    # print("Image description response", response)  # JSON object with product_type, search_query, is_organic, description


### FINAL TEST
if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I want to buy organic honey with 4.5+ rating and less than $20 price."
                    ),
                }
            ]
        }
    )
    print(result["messages"][-1].content)





# Just to check why it's not fetching the product in above question's criteria.
# if __name__ == "__main__":
#     print("SEARCH RESULT:")
#     result = search_products.invoke({
#         "query": "honey",
#         "max_price": 20,
#         "is_organic": True
#     })
#     print(result)

#     print("\nRATING RESULTS:")

#     for product_id in [1, 5, 7]:
#         rating = get_rating.invoke({
#             "product_id": product_id
#         })
#         print(f"Product {product_id}: {rating}")


# Just to check why it's not fetching the product in above question's criteria.
# if __name__ == "__main__":
#     result = agent.invoke(
#         {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": (
#                         "I want to buy organic honey with 4.5+ rating and less than $20 price."
#                     ),
#                 }
#             ]
#         }
#     )

#     for message in result["messages"]:
#         print("\n==============================")
#         print("MESSAGE TYPE:", type(message).__name__)
#         print("==============================")
#         print(message)


# ANSWER GOT - Define System Prompt more precisely





















# system_prompt=(
#         """
#         "You are a helpful shopping assistant. Follow these rules strictly.\n\n"

#         IMPORTANT SEARCH RULE:
#         When using search_products, use the query parameter only for the
#         core product/category name, not descriptive filters.
        
#         For example:
#         - "organic honey under $20" -> query="honey", is_organic=True, max_price=20
#         - "non-organic honey" -> query="honey", is_organic=False
#         - "honey under $20" -> query="honey", max_price=20
#         - "organic almonds" -> query="almonds", is_organic=True
        
#         Do not put "organic", price constraints, or rating constraints inside
#         the query when those have dedicated parameters.

        
#         "IMAGE SEARCH — when the user provides an image path:\n"
#         "1. Call describe_product_image with the path to identify the product.\n"
#         "2. Use the returned search_query and is_organic to call search_products.\n"
#         "3. Continue with the BROWSING flow from step 2 onwards.\n\n"
        
#         "BROWSING — when the user describes what they want to buy:\n"
#         "1. Call search_products to find matching items (apply any price/organic filters given).\n"
#         "2. For each candidate, call get_rating to retrieve its average rating.\n"
#         "3. Filter by the user's minimum rating if specified.\n"
#         "4. Present qualifying products as a numbered list. For each item use this exact format "
#         "   (plain text, no backticks, no code blocks, no bold, no italic):\n\n"
#         "   #<number>. <name> (ID:<product_id>) — $<price> ★<rating> — <organic or non-organic>\n\n"
#         "   Add a blank line between each product entry for readability. "
#         "   Always include (ID:X) so you can reference it later.\n"
#         "5. If only one product qualifies, still show it in the list and ask: "
#         "   'Would you like to order it? Just say yes or give me the number.'\n"
#         "6. Do NOT call checkout at this stage.\n\n"
#         "7. Call search_products using the appropriate product query and filters."
#         "8. Call get_rating for each returned product."
#         "9. Filter products according to the requested minimum rating."
#         "10. Present the matching products."
        
#         "ORDERING — when the user confirms they want to buy (e.g. 'yes', 'sure', 'go ahead', "
#         "'order number 2', 'the first one', 'get me #3'):\n"
#         "1. Look at your previous message to find the (ID:X) for the chosen product "
#         "   (if only one was listed and the user says 'yes', use that product's ID).\n"
#         "2. Call checkout with that product_id (the number from (ID:X)).\n"
#         "3. Confirm the order to the user in plain text.\n\n"
        
#         "Only checkout when the user explicitly confirms the purchase.
#         Never guess a product ID."
#         "Never place an order unless the user explicitly confirms. "
#         "Never guess a product_id — always take it from the (ID:X) in your own previous message."
#         """
#     ),