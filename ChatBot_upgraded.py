import fitz  # PyMuPDF
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Override nltk.data path to your known good nltk_data directory
nltk.data.path = [r"C:\Users\Ankur Banerjee\AppData\Roaming\nltk_data"]

# Use custom PunktSentenceTokenizer directly to avoid loading 'punkt' resource (bypasses punkt_tab error)
punkt_params = PunktParameters()
sent_tokenizer = PunktSentenceTokenizer(punkt_params)

# Load DialoGPT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, max_tokens=500, tokenizer=None):
    # Using custom tokenizer instead of sent_tokenize to avoid punkt_tab error
    sentences = sent_tokenizer.tokenize(text)
    chunks = []
    current_chunk = ""
    current_len = 0

    for sent in sentences:
        sent_len = len(tokenizer.tokenize(sent)) if tokenizer else len(sent.split())
        if current_len + sent_len > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sent
            current_len = sent_len
        else:
            current_chunk += " " + sent
            current_len += sent_len

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_response(user_input, context, chat_history_ids=None, attention_mask=None):
    prompt = f"Summarize the following document:\n{context}\n"
    if user_input.strip():
        prompt += f"Question: {user_input}\nAnswer:"

    new_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
    new_attention_mask = torch.ones_like(new_input_ids)

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        attention_mask = torch.cat([attention_mask, new_attention_mask], dim=-1)
    else:
        bot_input_ids = new_input_ids
        attention_mask = new_attention_mask

    max_model_length = 1024
    # Truncate if too long
    if bot_input_ids.shape[-1] > max_model_length:
        bot_input_ids = bot_input_ids[:, -max_model_length:]
        attention_mask = attention_mask[:, -max_model_length:]

    # Compute max_length so it's input length + generation length
    input_len = bot_input_ids.shape[-1]
    max_output_len = 150
    total_max_length = min(input_len + max_output_len, max_model_length)  # don't exceed model max

    chat_history_ids = model.generate(
        bot_input_ids,
        attention_mask=attention_mask,
        max_length=total_max_length,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
    )

    response = tokenizer.decode(chat_history_ids[:, input_len:][0], skip_special_tokens=True)
    return response, chat_history_ids, attention_mask


if __name__ == "__main__":
    pdf_path = r"C:\Users\Ankur Banerjee\Downloads\IJCRT23058131.pdf"  # Your PDF file
    pdf_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pdf_text, max_tokens=500, tokenizer=tokenizer)
    print(f"PDF text chunked into {len(chunks)} parts.")

    chat_history = None
    attention_mask = None
    print("Chatbot ready (type 'exit' to quit).")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Use last chunk for context; you can improve retrieval here
        context = chunks[-1]

        response, chat_history, attention_mask = generate_response(user_input, context, chat_history, attention_mask)
        print("Bot:", response)
