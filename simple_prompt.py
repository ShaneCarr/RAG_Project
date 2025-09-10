def answer_question(query):
    context = search(query)
    context_text = "\n\n".join([f"{f}\n{c}" for f, c in context])

    prompt = f"""
    You are an expert software engineer. 
    Answer the question using ONLY the following code context:

    {context_text}

    Question: {query}
    """

    resp = openai.ChatCompletion.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp['choices'][0]['message']['content']
