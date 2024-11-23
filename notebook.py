#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Cell 1: Setup and Imports

# Import necessary libraries
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import logging
import io
import json
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('story_generation.log'),
        logging.StreamHandler()
    ]
)

# Create timestamped directory for saving outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = Path(f"story_generation_{timestamp}")
save_dir.mkdir(exist_ok=True)


# In[ ]:


# Cell 2: Load and Prepare Documents

# Load all markdown files from chapters directory
loader = DirectoryLoader('./chapters', glob="**/*.md")
documents = loader.load()

# Add counter index to each document
for i, doc in enumerate(documents):
    doc.metadata['index'] = i

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_documents(documents)

# Preserve the original document index in the chunks
for text in texts:
    text.metadata['doc_index'] = text.metadata['index']

# Add chunk index and master index to each text
master_index = 0
for doc_index in range(max(text.metadata['doc_index'] for text in texts) + 1):
    # Get all chunks for this document
    doc_chunks = [t for t in texts if t.metadata['doc_index'] == doc_index]
    
    # Add chunk index within document
    for chunk_idx, chunk in enumerate(doc_chunks):
        chunk.metadata['chunk_index'] = chunk_idx
        chunk.metadata['master_index'] = master_index
        master_index += 1

# Create embeddings and store in Chroma vector database
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Create retriever interface
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": len(texts)}  # Retrieve all texts
)


# In[ ]:


# Cell 3: Initialize Language Models and Prompts

# Initialize LLMs with different temperatures for different creative needs
experience_llm = ChatOpenAI(temperature=0.7)
story_llm = ChatOpenAI(temperature=0.9)
reflection_llm = ChatOpenAI(temperature=0.4)
refinement_llm = ChatOpenAI(temperature=0.7)
latex_llm = ChatOpenAI(temperature=0.3)

# Create prompts for each stage
experience_prompt = PromptTemplate(
    input_variables=["context"],
    template="""Given these story elements and memories:

{context}

Suggest deeply human experiences, emotions, and universal themes that relate to these elements. 
Consider fundamental human experiences like love, loss, growth, fear, triumph, etc.
Be specific and draw meaningful connections."""
)

story_prompt = PromptTemplate(
    input_variables=["experiences", "related_memories"],
    template="""Using these human experiences and thematic elements:

{experiences}

And drawing from these related memories and story elements:

{related_memories}

Craft a rich and engaging story segment that weaves these elements together.
Focus on vivid imagery, emotional resonance, and narrative flow."""
)

reflection_prompt = PromptTemplate(
    input_variables=["story", "experiences"],
    template="""Reflect on this story segment and its themes:

{story}

Consider these underlying experiences and emotions:
{experiences}

Analyze the narrative for:
1. Thematic consistency
2. Character development
3. Emotional resonance
4. Plot coherence
5. Symbolic depth

Provide specific suggestions for deepening and enriching the narrative."""
)

refinement_prompt = PromptTemplate(
    input_variables=["story", "reflection", "experiences"],
    template="""Enhance this story segment:

{story}

Based on this reflection:
{reflection}

And maintaining these core experiences:
{experiences}

Rewrite the segment with deeper emotional resonance, richer symbolism, and stronger narrative cohesion."""
)

latex_prompt = PromptTemplate(
    input_variables=["story"],
    template="""Convert this story segment into properly formatted LaTeX:

{story}

Use appropriate LaTeX formatting for literary text, including proper paragraph breaks,
quotation marks, and any needed structural elements."""
)

# Create chains for each stage
experience_chain = LLMChain(llm=experience_llm, prompt=experience_prompt)
story_chain = LLMChain(llm=story_llm, prompt=story_prompt)
reflection_chain = LLMChain(llm=reflection_llm, prompt=reflection_prompt)
refinement_chain = LLMChain(llm=refinement_llm, prompt=refinement_prompt)
latex_chain = LLMChain(llm=latex_llm, prompt=latex_prompt)


# In[ ]:


# Cell 4: Process Documents through Experience Chain

# Initialize output containers
experiences_list = []

# Get all documents (in order)
all_docs = retriever.get_relevant_documents("")

# Create a mapping from doc content to index
doc_indices = {doc.page_content: idx for idx, doc in enumerate(all_docs)}

# Process each document chunk to generate experiences
for current_idx, current_doc in enumerate(all_docs):
    logging.info(f"Experience Chain: Processing document {current_idx + 1}/{len(all_docs)}")
    
    current_content = current_doc.page_content
    
    # Only retrieve documents that came before this one
    related_docs = [doc for doc in all_docs if doc_indices[doc.page_content] < current_idx]
    related_memories = "\n\n".join([doc.page_content for doc in related_docs])
    
    # Combine current content with related memories for context
    full_context = f"Current Memory:\n{current_content}\n\nRelated Memories:\n{related_memories}"
    
    # Generate human experiences
    experiences = experience_chain.run(context=full_context)
    experiences_list.append(experiences)
    
    # Save experiences to file
    with open(save_dir / f"doc_{current_idx}_experiences.txt", 'w') as f:
        f.write(experiences)


# In[ ]:


# Cell 5: Process Documents through Story Chain

# Initialize output container
stories_list = []

# Process each experiences to generate initial stories
for current_idx, experiences in enumerate(experiences_list):
    logging.info(f"Story Chain: Processing document {current_idx + 1}/{len(experiences_list)}")
    
    # Get related memories (only prior docs)
    related_docs = [doc for idx, doc in enumerate(all_docs) if idx < current_idx]
    related_memories = "\n\n".join([doc.page_content for doc in related_docs])
    
    # Generate story
    story = story_chain.run(experiences=experiences, related_memories=related_memories)
    stories_list.append(story)
    
    # Save initial story to file
    with open(save_dir / f"doc_{current_idx}_initial_story.txt", 'w') as f:
        f.write(story)


# In[ ]:


# Cell 6: Process Stories through Reflection Chain

# Initialize output container
reflections_list = []

# Process each story to generate reflections
for current_idx, story in enumerate(stories_list):
    logging.info(f"Reflection Chain: Processing story {current_idx + 1}/{len(stories_list)}")
    
    experiences = experiences_list[current_idx]
    
    # Generate reflection
    reflection = reflection_chain.run(story=story, experiences=experiences)
    reflections_list.append(reflection)
    
    # Save reflection to file
    with open(save_dir / f"doc_{current_idx}_reflection.txt", 'w') as f:
        f.write(reflection)


# In[ ]:


# Cell 7: Process Stories through Refinement Chain

# Initialize output container
refined_stories_list = []

# Refine each story based on reflections
for current_idx, story in enumerate(stories_list):
    logging.info(f"Refinement Chain: Processing story {current_idx + 1}/{len(stories_list)}")
    
    reflection = reflections_list[current_idx]
    experiences = experiences_list[current_idx]
    
    # Refine the story
    refined_story = refinement_chain.run(
        story=story,
        reflection=reflection,
        experiences=experiences
    )
    refined_stories_list.append(refined_story)
    
    # Save refined story to file
    with open(save_dir / f"doc_{current_idx}_refined_story.txt", 'w') as f:
        f.write(refined_story)


# In[ ]:


# Cell 8: Convert Refined Stories to LaTeX

# Initialize output container
latex_output = io.StringIO()

# Convert each refined story to LaTeX
for current_idx, refined_story in enumerate(refined_stories_list):
    logging.info(f"LaTeX Conversion: Processing refined story {current_idx + 1}/{len(refined_stories_list)}")
    
    # Convert to LaTeX
    latex_text = latex_chain.run(story=refined_story)
    
    # Save LaTeX to individual file
    with open(save_dir / f"doc_{current_idx}_story.tex", 'w') as f:
        f.write(latex_text)
    
    # Append to final LaTeX output
    latex_output.write(latex_text + "\n\n")

# Save the complete LaTeX document
final_latex = latex_output.getvalue()
with open(save_dir / "complete_story.tex", 'w') as f:
    f.write(final_latex)

logging.info("Story generation complete")
print("Generated LaTeX Story:")
print(final_latex)

