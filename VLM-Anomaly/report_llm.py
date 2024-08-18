import g4f
from g4f.client import Client

client = Client()

def generate_report(query):

    prompt = "Prompt: Generate a detailed medical report for a brain MRI diagnosis. The report should take into account the identified brain regions (e.g., frontal lobe, occipital lobe, etc.) and their respective probabilities of anomaly as output by a model. The diagnosis should include a clear assessment of the severity of the abnormality based on these probabilities, with insights on potential medical conditions or abnormalities related to each region. Conclude the report with recommendations for further tests, treatments, or consultations.\n Query:"
    chat_completion = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt+query}], stream=True)

    for completion in chat_completion:
        print(completion.choices[0].delta.content or "", end="", flush=True)
