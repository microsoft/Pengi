from wrapper import PengiWrapper as Pengi

pengi = Pengi(config="base") #base or base_no_text_enc
audio_file_paths = ["FILE_PATH_1", "FILE_PATH_2"]
text_prompts = ["generate metadata", "generate metadata"]
add_texts = ["",""]

generated_response = pengi.generate(
                                    audio_paths=audio_file_paths,
                                    text_prompts=text_prompts, 
                                    add_texts=add_texts, 
                                    max_len=30, 
                                    beam_size=3, 
                                    temperature=1.0, 
                                    stop_token=' <|endoftext|>',
                                    )

generated_summary = pengi.describe(
                                    audio_paths=audio_file_paths,
                                    max_len=30, 
                                    beam_size=3,  
                                    temperature=1.0,  
                                    stop_token=' <|endoftext|>',
                                    )

audio_prefix, audio_embeddings = pengi.get_audio_embeddings(audio_paths=audio_file_paths)

text_prefix, text_embeddings = pengi.get_prompt_embeddings(prompts=text_prompts)