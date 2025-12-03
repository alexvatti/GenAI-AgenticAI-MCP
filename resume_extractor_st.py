import streamlit as st
import os
import PyPDF2
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # ✅ updated import
from langchain.prompts import ChatPromptTemplate
import json
import traceback
import tempfile

# Your original functions - unchanged
def extract_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def resume_to_json(input_path=None, resume_text=None):
    # Load environment variables
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    if input_path:
        resume_text = extract_pdf_text(input_path)
    elif not resume_text:
        raise ValueError("Either input_path or resume_text must be provided.")
    
    # Prompt template
    prompt_template = """
    Extract the following information from this resume text and return it strictly as valid JSON:
    {{
      "Name": "",
      "Email": "",
      "Phone": "",
      "Education": "",
      "Experience": [],
      "Skills": []
    }}
    
    Resume Text:
    {text}
    
    IMPORTANT:
    - Return ONLY valid JSON.
    - Do not include ```json or any extra text.
    """
    chat_prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    
    # Run pipeline
    pipeline = chat_prompt | llm
    output_message = pipeline.invoke({"text": resume_text})
    return output_message.content.strip()  # ensure clean string

def process_resume(input_path=None, resume_text=None):
    """
    Extract resume info to JSON using gpt-4o-mini
    """
    try:
        result_str = resume_to_json(input_path=input_path, resume_text=resume_text)
        
        # Try parsing JSON safely
        try:
            data = json.loads(result_str)
        except json.JSONDecodeError:
            st.warning("⚠️ Model returned invalid JSON. Cleaning response...")
            # Auto-fix common cases like ```json wrappers
            cleaned = result_str.strip().replace("```json", "").replace("```", "")
            data = json.loads(cleaned)
        
        return data, result_str
        
    except Exception as e:
        st.error(f"Error: {e}")
        traceback.print_exc()
        return None, None

# Streamlit App
def main():
    st.title("📄 Resume to JSON Extractor")
    st.write("Extract structured information from resumes using GPT-4o-mini and convert to JSON format.")
    
    # Input method selection
    st.subheader("📋 Input Method")
    input_method = st.radio(
        "Choose how to provide resume:",
        ["Upload PDF file", "Enter text directly"]
    )
    
    resume_file_path = None
    resume_text = None
    
    if input_method == "Upload PDF file":
        uploaded_file = st.file_uploader(
            "Choose PDF resume file", 
            type=['pdf'],
            help="Upload a PDF resume file"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                resume_file_path = tmp_file.name
            st.success(f"✅ PDF uploaded: {uploaded_file.name}")
    
    else:
        resume_text = st.text_area(
            "Enter resume text:",
            height=300,
            placeholder="Paste the resume text here..."
        )
        if resume_text:
            st.success("✅ Resume text entered")
    
    # Output file name
    st.subheader("💾 Output Settings")
    output_filename = st.text_input(
        "JSON output filename:",
        value="resume_output.json",
        help="Name for the downloaded JSON file"
    )
    
    # Submit button
    if st.button("🚀 Extract to JSON", type="primary"):
        # Validation
        if input_method == "Upload PDF file" and resume_file_path is None:
            st.error("❌ Please upload a PDF file.")
            return
        
        if input_method == "Enter text directly" and not resume_text.strip():
            st.error("❌ Please enter resume text.")
            return
        
        try:
            # Show loading spinner
            with st.spinner("Extracting information... Please wait."):
                if input_method == "Upload PDF file":
                    data, raw_result = process_resume(input_path=resume_file_path)
                    # Clean up temporary file
                    os.unlink(resume_file_path)
                else:
                    data, raw_result = process_resume(resume_text=resume_text)
            
            if data is not None:
                st.success("✅ Extraction completed!")
                
                # Display structured results
                st.subheader("📊 Extracted Information")
                
                # Personal Information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**👤 Personal Information**")
                    st.write(f"**Name:** {data.get('Name', 'Not found')}")
                    st.write(f"**Email:** {data.get('Email', 'Not found')}")
                    st.write(f"**Phone:** {data.get('Phone', 'Not found')}")
                
                with col2:
                    st.write("**🎓 Education**")
                    education = data.get('Education', 'Not found')
                    if isinstance(education, list):
                        for edu in education:
                            st.write(f"• {edu}")
                    else:
                        st.write(education)
                
                # Experience
                st.write("**💼 Experience**")
                experience = data.get('Experience', [])
                if experience:
                    for i, exp in enumerate(experience):
                        st.write(f"**{i+1}.** {exp}")
                else:
                    st.write("No experience found")
                
                # Skills
                st.write("**🛠️ Skills**")
                skills = data.get('Skills', [])
                if skills:
                    # Display skills as tags
                    skills_text = " • ".join(skills)
                    st.write(skills_text)
                else:
                    st.write("No skills found")
                
                # JSON Output
                st.subheader("📋 JSON Output")
                pretty_json = json.dumps(data, indent=4)
                
                # Display JSON in code block
                st.code(pretty_json, language='json')
                
                # Download button
                st.download_button(
                    label="📥 Download JSON",
                    data=pretty_json,
                    file_name=output_filename,
                    mime="application/json"
                )
                
                # Raw response (expandable)
                with st.expander("🔍 Raw Model Response"):
                    st.text(raw_result)
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("Please check your OpenAI API key and ensure all dependencies are installed.")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ How it works")
        st.write("""
        1. **Input Resume**: Upload PDF or enter text
        2. **AI Processing**: Uses GPT-4o-mini to extract info
        3. **JSON Structure**: Gets Name, Email, Phone, Education, Experience, Skills
        4. **Download**: Get structured JSON output
        """)
        
        st.header("📋 JSON Structure")
        st.code('''
{
  "Name": "",
  "Email": "",
  "Phone": "",
  "Education": "",
  "Experience": [],
  "Skills": []
}
        ''', language='json')
        
        st.header("⚙️ Requirements")
        st.code("pip install streamlit langchain-openai PyPDF2 python-dotenv")
        
        st.header("🔑 API Key")
        st.write("Make sure to set your OpenAI API key in a .env file:")
        st.code('OPENAI_API_KEY="sk-..."')
        
        st.header("📄 Supported Formats")
        st.write("""
        - PDF files (upload)
        - Plain text (direct input)
        """)

if __name__ == "__main__":
    main()