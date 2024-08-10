#install reqd libraries
pip install pywhatkit
pip install selenium
pip install torch torchvision torchaudio
pip install transformers[torch] -U
pip install accelerate -U
pip install openai==0.28

#import necessary libraries
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Set up the WebDriver
service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

# Open WhatsApp Web
driver.get('https://web.whatsapp.com')

#login
time.sleep(30)

# Find the chat by searching for the contact name from which you want to train data
contact_name = ' '  #enter contact name
search_box = driver.find_element(By.XPATH, '')  #find XPATH by inspecting element
search_box.click()
search_box.send_keys(contact_name)
search_box.send_keys(Keys.ENTER)

#store all messages of the conversation, or one sided by using the reqd div 
messages = driver.find_elements(By.XPATH, '//div[contains(@class, "message-out")')

#messages contain date and message. split and save only the message
chat_data=[messages[i].text.split('\n')[-2] for i in range(len(messages)) if len(messages[i].text.split('\n')) >= 2]

# Save to a text file
with open('strings.txt', 'w') as file:
    for s in chat_data:
        file.write(s + '\n')

# Read from a text file
with open('strings.txt', 'r') as file:
    retrieved_strings = [line.strip() for line in file]

# Print to verify
print(retrieved_strings)

# Step 4: Import necessary libraries for transformer
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import datasets

# Create a DataFrame with message pairs
data = []
messages=retrieved_strings
for i in range(1, len(messages)):
    data.append({
        'input': messages[i-1],
        'response': messages[i]
    })

df = pd.DataFrame(data)

from transformers import GPT2Tokenizer, GPT2LMHeadModel
# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add padding token to tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Tokenize the input-output pairs
def tokenize_function(examples):
    return tokenizer(examples["input"], truncation=True, padding=True)

# Convert to dataset format
dataset = datasets.Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["input"])

# Define data collator and training arguments
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model_save_path = "./trained_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}")

# Set the device
device = torch.device("mps" if torch.has_mps else "cpu")
model.to(device)

prompt=input()
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the same device as the model
outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Input: {prompt}")
print(f"Generated response: {response}")

#