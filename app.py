from agent import FAQAgent

agent = FAQAgent(
    "faqs.json"
)

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    result = agent.ask(
        user_input
    )

    print("\nTop Retrieved FAQs:")

    for idx, item in enumerate(
        result["retrieved_docs"],
        start=1
    ):

        print(
            f"\n{idx}. "
            f"{item['faq']['question']}"
        )

        print(
            f"Score: "
            f"{item['score']:.2f}"
        )

    print(
        f"\nBest Score: "
        f"{result['score']:.2f}"
    )

    print(
        f"\nBot: "
        f"{result['answer']}"
    )