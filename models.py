from transformers import MistralForCausalLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import LlamaForCausalLM,BitsAndBytesConfig
import torch

   
class adSent_Mistral:
    def __init__(
        self,
        model="Open-Orca/Mistral-7B-OpenOrca",
        tokenizer="Open-Orca/Mistral-7B-OpenOrca",
        device='cuda',
        max_length=2048,
    ):
        self._model = MistralForCausalLM.from_pretrained(model,
                                                            load_in_8bit=True, 
                                                            device_map="balanced"
                                                            )
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer,use_fast=False)
        self._tokenizer.eos_token = '<\s>'
        self._tokenizer.pad_token=self._tokenizer.eos_token
        self._device = device
        self._params = {"max_length": max_length}

    
    def sent_attack(self, prompt):
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self._params['max_length'], padding="max_length")
        inputs = inputs.to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, 
                                           max_new_tokens=2048, 
                                           pad_token_id=self._tokenizer.eos_token_id)
        response = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return response
       
class adSent_llama:
    def __init__(
        self,
        model="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer="meta-llama/Llama-3.1-8B-Instruct",
        device='cuda',
        max_length=1024,
        access_token=None
    ):
        
        self._model = AutoModelForCausalLM.from_pretrained(model,
                                                            torch_dtype=torch.float16, 
                                                            device_map="balanced",use_auth_token=access_token
                                                            )
        self._model.gradient_checkpointing_enable()
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer,use_fast=True, use_auth_token=access_token)
        self.label_token_fake = self._tokenizer("fake", add_special_tokens=False).input_ids[0]
        self.label_token_real = self._tokenizer("real", add_special_tokens=False).input_ids[0]
        self._params = {"max_length": max_length}
        self._device = device

    
    def get_response_classification(self, prompt):    
        messages = [
        {"role": "system", "content": "You are a fact checking journalist."},
        {"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt"
                                 #,truncation=True, max_length=self._params['max_length']
                                 ).to(self._device)
        #inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self._params['max_length'], padding="max_length")
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, 
                                            max_new_tokens=1,
                                            pad_token_id=self._tokenizer.eos_token_id,
                                            eos_token_id=self._tokenizer.eos_token_id,
                                            early_stopping=True 
                                           #pad_token_id=self._tokenizer.eos_token_id
                                           )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.lower()
    
    def get_response_classification_pbc(self, prompt):    
        messages = [
        {"role": "system", "content": "You are a fact checking journalist."},
        {"role": "user", "content": prompt}]
        
        index2label={0: "real", 1: "fake"}
        answer_sets = {
             "real": ['real', 'Real','REAL'],
             "fake": ['fake', 'Fake', 'FAKE'],
        }
        
        answer_sets_token_id = {}
        for label, answer_set in answer_sets.items():
            answer_sets_token_id[label] = []
            for answer in answer_set:
                answer_sets_token_id[label].append(self._tokenizer(answer).input_ids[-1])
                

        text = self._tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt"
                                 #,truncation=True, max_length=self._params['max_length']
                                 ).to(self._device)
        
        with torch.no_grad():
            output = self._model.generate(**inputs, 
                                           max_new_tokens=1,output_scores=True, return_dict_in_generate=True,
                                            pad_token_id=self._tokenizer.eos_token_id,
                                            eos_token_id=self._tokenizer.eos_token_id,
                                            early_stopping=True
                                           #pad_token_id=self._tokenizer.eos_token_id
                                           )
        
        #generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        
        pbc_probas = output.scores[0][:, answer_sets_token_id['real']+answer_sets_token_id['fake']].softmax(-1)
        yes_proba_matrix = pbc_probas[:, :len(answer_sets['real'])].sum(dim=1)
        no_proba_matrix = pbc_probas[:, len(answer_sets['real']):].sum(dim=1)
        probas = torch.cat((yes_proba_matrix.reshape(-1, 1), no_proba_matrix.reshape(-1, 1)), -1)
        probas_per_first_token=torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in probas_per_first_token.values]
        sequences = [index2label[int(indice)] for indice in probas_per_first_token.indices]

        return sequences[0], sequence_probas
    
    def training(self, prompt, label_word):
        """
        Perform one forward pass for training.
        
        Args:
            prompt (str): News article with instruction prompt prepended.
            label (int): Dummy label (used only to keep consistent signature, actual label will be derived from input).

        Returns:
            loss (Tensor): Training loss.
        """
        
       
        messages = [
            {"role": "system", "content": "You are a fact checking journalist."},
            {"role": "user", "content": prompt}
        ]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self._tokenizer(
            text,
            return_tensors="pt"
            #truncation=True,
            #max_length=self._params["max_length"],
            #padding="max_length"
        ).to(self._device)

        with self._tokenizer.as_target_tokenizer():
            label_token_ids = self._tokenizer(label_word, add_special_tokens=False).input_ids
        
        labels = inputs["input_ids"].clone()
        labels[:, :-len(label_token_ids)] = -100  # Mask all tokens except the label
        
        outputs = self._model(**inputs, labels=labels)
        return outputs.loss
    
    def training_classification_style(self, prompt, label_word):
        #print(f"Fake token id: {self.label_token_fake}")
        #print(f"Real token id: {self.label_token_real}")
        messages = [
            {"role": "system", "content": "You are a fact checking journalist."},
            {"role": "user", "content": prompt}
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        input_ids = inputs["input_ids"]

        # Forward pass
        outputs = self._model(**inputs)
        logits = outputs.logits  # shape: [1, seq_len, vocab_size]

        # Take the logits of the next token (prediction after the last token)
        next_token_logits = logits[0, -1, :]  # [vocab_size]

        # Collect the logits for only the label tokens (you can add both fake/real here)
        label_logits = next_token_logits[[self.label_token_fake, self.label_token_real]]  # [2]
        label_index = 0 if label_word == "real" else 1

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("⚠️ NaNs or Infs found in logits!")
            print(f"logits: {logits}")
            
            
        # Label words: "fake" or "real"
        label_token_ids = self._tokenizer([label_word], add_special_tokens=False).input_ids
        if len(label_token_ids[0]) != 1:
            raise ValueError("Label word must map to exactly one token")
        label_token_id = label_token_ids[0][0]    
        
        # Compute classification loss manually
        label_tensor = torch.tensor([label_index], device=self._device)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(label_logits.unsqueeze(0), label_tensor)  # [1, 2] vs [1]

        return loss,label_logits

    def classification_style_inference(self, prompt):
        #print(f"Fake token id: {self.label_token_fake}")
        #print(f"Real token id: {self.label_token_real}")
        messages = [
            {"role": "system", "content": "You are a fact checking journalist."},
            {"role": "user", "content": prompt}
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        input_ids = inputs["input_ids"]

        # Forward pass
        outputs = self._model(**inputs)
        logits = outputs.logits  # shape: [1, seq_len, vocab_size]

        # Take the logits of the next token (prediction after the last token)
        next_token_logits = logits[0, -1, :]  # [vocab_size]

        # Collect the logits for only the label tokens (you can add both fake/real here)
        label_logits = next_token_logits[[self.label_token_fake, self.label_token_real]]  # [2]     

        return label_logits
    
    def sent_attack(self, prompt):    
        messages = [
        {"role": "system", "content": "You are a fact checking journalist."},
        {"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt"
                                 #,truncation=True, max_length=self._params['max_length']
                                 ).to(self._device)
        #inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self._params['max_length'], padding="max_length")
        with torch.no_grad():

            generated_ids = self._model.generate(**inputs, 
                                        max_new_tokens=512,
                                        pad_token_id=self._tokenizer.eos_token_id,
                                        eos_token_id=self._tokenizer.eos_token_id,
                                        early_stopping=True
                                           #pad_token_id=self._tokenizer.eos_token_id
                                           )
        #response = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        filtered_generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        full_output = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        trimed = self._tokenizer.batch_decode(filtered_generated_ids, skip_special_tokens=True)[0]
        return full_output,trimed
       
class adSent_qwen:
    def __init__(
        self,
        model="Qwen/Qwen2.5-72B-Instruct",
        tokenizer="Qwen/Qwen2.5-72B-Instruct",
        device='cuda',
        max_length=1024,
    ):
        #quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)  # ✅ This enables CPU offloading
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # ✅ Enables 4-bit quantization
        bnb_4bit_compute_dtype=torch.float16,  # ✅ Uses FP16 for better performance
        bnb_4bit_use_double_quant=True,  # ✅ Further reduces memory usage
        bnb_4bit_quant_type="nf4")
        self._model = AutoModelForCausalLM.from_pretrained(model,
                                                            torch_dtype="auto",
                                                            #quantization_config=quantization_config, 
                                                            #device_map="auto"
                                                            device_map="balanced"
                                                            )

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
       
        #self._tokenizer.eos_token = '<\s>'
        #self._tokenizer.pad_token=self._tokenizer.eos_token
        self._device = device
        self._params = {"max_length": max_length}

    
    def sent_attack(self, prompt):
        messages = [
        {"role": "system", "content": "You are a fact checking journalist."},
        {"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        #print(f'This is text:{text}')
        inputs = self._tokenizer([text],
                                 return_tensors="pt"
                                 #,truncation=True, max_length=self._params['max_length']
                                 ).to(self._device)
        #print(inputs)
        with torch.no_grad():
            #generated_ids = self._model.generate(**inputs, 
                                           #max_new_tokens=512 
                                           #pad_token_id=self._tokenizer.eos_token_id
                                           #)
            generated_ids = self._model.generate(**inputs,
            max_new_tokens=512,
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            early_stopping=True
            )

        filtered_generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        ############### Decode only the newly generated tokens####################################
        full_output = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        trimed = self._tokenizer.batch_decode(filtered_generated_ids, skip_special_tokens=True)[0]

        return full_output,trimed
    
    def get_response_classification(self, prompt):    
        messages = [
        {"role": "system", "content": "You are a fact checking journalist."},
        {"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt"
                                 #,truncation=True, max_length=self._params['max_length']
                                 ).to(self._device)
        #inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self._params['max_length'], padding="max_length")
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, 
                                            max_new_tokens=1,
                                            pad_token_id=self._tokenizer.eos_token_id,
                                            eos_token_id=self._tokenizer.eos_token_id,
                                            early_stopping=True 
                                           #pad_token_id=self._tokenizer.eos_token_id
                                           )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.lower()
    
    def get_response_classification_pbc(self, prompt):    
        messages = [
        {"role": "system", "content": "You are a fact checking journalist."},
        {"role": "user", "content": prompt}]
        
        index2label={0: "real", 1: "fake"}
        answer_sets = {
             "real": ['real', 'Real'],
             "fake": ['fake', 'Fake'],
        }
        
        answer_sets_token_id = {}
        for label, answer_set in answer_sets.items():
            answer_sets_token_id[label] = []
            for answer in answer_set:
                answer_sets_token_id[label].append(self._tokenizer(answer).input_ids[-1])
                

        text = self._tokenizer.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt"
                                 #,truncation=True, max_length=self._params['max_length']
                                 ).to(self._device)
        
        with torch.no_grad():
            output = self._model.generate(**inputs, 
                                           max_new_tokens=1,output_scores=True, return_dict_in_generate=True,
                                            pad_token_id=self._tokenizer.eos_token_id,
                                            eos_token_id=self._tokenizer.eos_token_id,
                                            early_stopping=True
                                           #pad_token_id=self._tokenizer.eos_token_id
                                           )
        
        #generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        
        pbc_probas = output.scores[0][:, answer_sets_token_id['real']+answer_sets_token_id['fake']].softmax(-1)
        yes_proba_matrix = pbc_probas[:, :len(answer_sets['real'])].sum(dim=1)
        no_proba_matrix = pbc_probas[:, len(answer_sets['real']):].sum(dim=1)
        probas = torch.cat((yes_proba_matrix.reshape(-1, 1), no_proba_matrix.reshape(-1, 1)), -1)
        probas_per_first_token=torch.max(probas, dim=1)
        sequence_probas = [float(proba) for proba in probas_per_first_token.values]
        sequences = [index2label[int(indice)] for indice in probas_per_first_token.indices]

        return sequences[0], sequence_probas