import streamlit as st
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI

def analyze_text(input_text):
    """
    Text Analyzer using LangChain and OpenAI Chat API.
    This script takes text input, sends it to an LLM for analysis,
    and returns the number of characters, words, paragraphs, and sentences
    in JSON format.
    """
    
    # Use the provided input text
    text = input_text
    
    # Load environment variables from .env
    load_dotenv()
    
    # Prompt template for counting
    PROMPT = """
    You are an expert text analyzer. Given the text below, return:
    1) Number of characters in clude everything
    2) Number of words
    3) Number of paragraphs
    4) Number of sentences
    Text:
    {text}
    Provide the output in JSON format with keys: characters, words, paragraphs, sentences
    """
    
    prompt = PromptTemplate(input_variables=["text"], template=PROMPT)
    
    # Initialize LLM
    llm = ChatOpenAI(temperature=0)
    
    # Build chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Get result
    result = chain.invoke({"text": text})
    
    return result

# Streamlit App
def main():
    st.title("📝 Text Analyzer")
    st.write("This app analyzes text using LangChain and OpenAI Chat API to count characters, words, paragraphs, and sentences.")
    
    # Text input
    st.subheader("Enter Text for Analysis")
    user_text = st.text_area("Enter your text here:", height=200, placeholder="Type or paste your text here...")
    
    # Submit button
    if st.button("🚀 Analyze Text", type="primary"):
        if not user_text.strip():
            st.error("❌ Please enter some text to analyze.")
            return
        
        try:
            # Show loading spinner
            with st.spinner("Analyzing text... Please wait."):
                result = analyze_text(user_text)
            
            # Display results
            st.success("✅ Analysis completed!")
            
            # Display the raw result
            st.subheader("📊 Analysis Results")
            
            # Try to parse JSON from result if it's in the expected format
            try:
                if hasattr(result, 'content'):
                    result_content = result.content
                elif isinstance(result, dict) and 'text' in result:
                    result_content = result['text']
                else:
                    result_content = str(result)
                
                # Try to extract JSON from the content
                import re
                json_match = re.search(r'\{.*\}', result_content, re.DOTALL)
                if json_match:
                    json_data = json.loads(json_match.group())
                    
                    # Display in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Characters", json_data.get('characters', 'N/A'))
                    with col2:
                        st.metric("Words", json_data.get('words', 'N/A'))
                    with col3:
                        st.metric("Paragraphs", json_data.get('paragraphs', 'N/A'))
                    with col4:
                        st.metric("Sentences", json_data.get('sentences', 'N/A'))
                    
                    # Display JSON in expandable section
                    with st.expander("📋 Raw JSON Output"):
                        st.json(json_data)
                else:
                    # If no JSON found, display raw result
                    st.write("Raw Result:")
                    st.write(result_content)
                    
            except (json.JSONDecodeError, KeyError) as e:
                st.write("Raw Result:")
                st.write(result)
                st.warning("Could not parse JSON from result. Displaying raw output above.")
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("Please check your OpenAI API key and internet connection.")

if __name__ == "__main__":
    main()