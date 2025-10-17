from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "apple/OpenELM-1_1B-Instruct"
save_dir = "./models/OpenELM-1_1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer.save_pretrained(save_dir)

model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(save_dir)
