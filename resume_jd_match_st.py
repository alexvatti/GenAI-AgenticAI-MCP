import streamlit as st
import re
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os

# Your original functions - unchanged
def read_resume(file_path):
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"{file_path} not found")
        
    if p.suffix.lower() == ".txt":
        return p.read_text()
        
    elif p.suffix.lower() == ".pdf":
        text = ""
        reader = PdfReader(str(p))
        for page in reader.pages:
            text += page.extract_text() + " "
        return text
        
    elif p.suffix.lower() in [".doc", ".docx"]:
        doc = docx.Document(str(p))
        text = " ".join([para.text for para in doc.paragraphs])
        return text
        
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    return " ".join(words)

def compute_similarity(resume_text, jd_text):
    emb = OpenAIEmbeddings()
    resume_vec = emb.embed_query(resume_text)
    jd_vec = emb.embed_query(jd_text)
    
    # cosine similarity
    score = cosine_similarity([resume_vec], [jd_vec])[0][0]
    return score

def analyze_match(resume_file_path, jd_text):
    """
    Resume vs Job Description Matching using LangChain embeddings.
    Supports: .txt, .pdf, .doc/.docx resumes
    Usage:
        export OPENAI_API_KEY="sk-..."
        python resume_jd_match.py resume.pdf job_description.txt
    """
    
    # Load .env variables
    load_dotenv()
    
    resume_text = read_resume(resume_file_path)
    
    # Preprocess
    resume_text_clean = preprocess_text(resume_text)
    jd_text_clean = preprocess_text(jd_text)
    
    # Compute semantic similarity
    similarity_score = compute_similarity(resume_text_clean, jd_text_clean)
    
    # Optional: simple keyword overlap for reference
    resume_words = set(resume_text_clean.split())
    jd_words = set(jd_text_clean.split())
    matching_keywords = resume_words.intersection(jd_words)
    
    return {
        'similarity_score': similarity_score,
        'matching_keywords': matching_keywords,
        'resume_only_keywords': resume_words - jd_words,
        'jd_only_keywords': jd_words - resume_words,
        'resume_text': resume_text,
        'jd_text': jd_text
    }

# Streamlit App
def main():
    st.title("📄 Resume vs Job Description Matcher")
    st.write("This app uses LangChain embeddings to match resumes against job descriptions. Supports .txt, .pdf, .doc/.docx files.")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Upload Resume")
        resume_file = st.file_uploader(
            "Choose resume file", 
            type=['txt', 'pdf', 'doc', 'docx'],
            help="Supported formats: .txt, .pdf, .doc, .docx"
        )
        
        if resume_file is not None:
            st.success(f"✅ Resume uploaded: {resume_file.name}")
    
    with col2:
        st.subheader("📝 Job Description")
        jd_input_method = st.radio(
            "Choose input method:",
            ["Enter text directly", "Upload text file"]
        )
        
        jd_text = ""
        if jd_input_method == "Enter text directly":
            jd_text = st.text_area(
                "Enter job description:", 
                height=200,
                placeholder="Paste the job description here..."
            )
        else:
            jd_file = st.file_uploader("Choose job description file", type=['txt'])
            if jd_file is not None:
                jd_text = jd_file.getvalue().decode("utf-8")
                st.success(f"✅ Job description uploaded: {jd_file.name}")
    
    # Submit button
    if st.button("🚀 Analyze Match", type="primary"):
        if resume_file is None:
            st.error("❌ Please upload a resume file.")
            return
        
        if not jd_text.strip():
            st.error("❌ Please provide a job description.")
            return
        
        try:
            # Save uploaded resume file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{resume_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(resume_file.getvalue())
                temp_resume_path = tmp_file.name
            
            # Show loading spinner
            with st.spinner("Analyzing match... Please wait."):
                result = analyze_match(temp_resume_path, jd_text)
            
            # Clean up temporary file
            os.unlink(temp_resume_path)
            
            # Display results
            st.success("✅ Analysis completed!")
            
            # Main similarity score
            st.subheader("🎯 Match Results")
            
            score = result['similarity_score']
            score_color = "green" if score >= 0.7 else "orange" if score >= 0.5 else "red"
            
            st.markdown(f"""
            ### Semantic Match Score: <span style="color: {score_color}; font-weight: bold;">{score:.2f}</span>
            """, unsafe_allow_html=True)
            
            # Score interpretation
            if score >= 0.8:
                st.success("🔥 Excellent match! This resume aligns very well with the job requirements.")
            elif score >= 0.6:
                st.info("✨ Good match! The resume shows relevance to the job description.")
            elif score >= 0.4:
                st.warning("⚡ Moderate match. Some alignment but could be improved.")
            else:
                st.error("❌ Low match. Significant gaps between resume and job requirements.")
            
            # Keyword analysis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Matching Keywords", len(result['matching_keywords']))
            with col2:
                st.metric("Resume-only Keywords", len(result['resume_only_keywords']))
            with col3:
                st.metric("JD-only Keywords", len(result['jd_only_keywords']))
            
            # Detailed keyword breakdown
            st.subheader("🔍 Keyword Analysis")
            
            with st.expander(f"🎯 Matching Keywords ({len(result['matching_keywords'])})", expanded=True):
                if result['matching_keywords']:
                    # Convert to sorted list for better display
                    matching_list = sorted(list(result['matching_keywords']))
                    st.write(", ".join(matching_list))
                else:
                    st.write("No matching keywords found")
            
            with st.expander(f"📄 Resume-only Keywords ({len(result['resume_only_keywords'])})"):
                if result['resume_only_keywords']:
                    resume_only_list = sorted(list(result['resume_only_keywords']))
                    st.write(", ".join(resume_only_list))
                else:
                    st.write("No unique resume keywords")
            
            with st.expander(f"📋 Job Description-only Keywords ({len(result['jd_only_keywords'])})"):
                if result['jd_only_keywords']:
                    jd_only_list = sorted(list(result['jd_only_keywords']))
                    st.write(", ".join(jd_only_list))
                    st.info("💡 Consider adding these keywords to your resume if they're relevant to your experience.")
                else:
                    st.write("No unique job description keywords")
            
            # Raw content preview
            with st.expander("📄 Resume Content Preview"):
                st.text_area("Resume text:", result['resume_text'][:1000] + "..." if len(result['resume_text']) > 1000 else result['resume_text'], height=200, disabled=True)
            
            with st.expander("📋 Job Description Content"):
                st.text_area("Job description text:", result['jd_text'], height=200, disabled=True)
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("Please check your OpenAI API key and ensure all dependencies are installed.")
            
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ How it works")
        st.write("""
        1. **Upload Resume**: Supports .txt, .pdf, .doc, .docx
        2. **Add Job Description**: Enter text or upload file
        3. **AI Analysis**: Uses OpenAI embeddings for semantic matching
        4. **Get Results**: Similarity score + keyword analysis
        """)
        
        st.header("📊 Score Guide")
        st.write("""
        - **0.8-1.0**: Excellent match 🔥
        - **0.6-0.8**: Good match ✨
        - **0.4-0.6**: Moderate match ⚡
        - **0.0-0.4**: Low match ❌
        """)
        
        st.header("⚙️ Requirements")
        st.code("pip install streamlit langchain-openai PyPDF2 python-docx scikit-learn python-dotenv")
        
        st.header("🔑 API Key")
        st.write("Make sure to set your OpenAI API key in a .env file:")
        st.code('OPENAI_API_KEY="sk-..."')

if __name__ == "__main__":
    main()