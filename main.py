from legal_assistant import ask_lawyer

query = input("Ask a question related to copyright law:\n")
answer = ask_lawyer(query)

print(answer)
