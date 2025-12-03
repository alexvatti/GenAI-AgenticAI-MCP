import streamlit as st
import sys
from dotenv import load_dotenv
from newspaper import Article  # ✅ Updated imports
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re

# Your original constants and functions - unchanged
SUMMARY_PROMPT = """You are a concise news summarizer. Given the article text, produce:
1) Headline (single line)
2) 2-3 sentence summary
3) 1-sentence "why it matters"

Article: {article}
"""

def fetch_article(url: str) -> str:
    art = Article(url)
    art.download()
    art.parse()
    return art.title + "\n\n" + art.text

def summarize_text(text: str) -> str:
    # Load environment variables from .env
    load_dotenv()
    
    llm = ChatOpenAI(temperature=0.2)
    prompt = PromptTemplate(input_variables=["article"], template=SUMMARY_PROMPT)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # ✅ use invoke instead of run
    result = chain.invoke({"article": text})
    return result["text"]

def process_articles(urls):
    """
    Given a list of article URLs, fetch (via newspaper3k), summarize each article,
    and produce a short digest.
    
    Usage:
        export OPENAI_API_KEY="sk-..."
        python news_summarizer.py https://example.com/article1 https://example.com/article2
    """
    results = []
    
    for u in urls:
        st.write(f"📰 **Fetching:** {u}")
        try:
            text = fetch_article(u)
            st.write("🤖 **Summarizing...**")
            summary = summarize_text(text)
            
            results.append({
                'url': u,
                'success': True,
                'summary': summary,
                'full_text': text
            })
            
        except Exception as e:
            st.error(f"❌ **Failed for** {u}: {str(e)}")
            results.append({
                'url': u,
                'success': False,
                'error': str(e)
            })
    
    return results

# Streamlit App
def main():
    st.title("📰 News Article Summarizer")
    st.write("Fetch articles from URLs and generate concise summaries using newspaper3k and ChatGPT.")
    
    # Input methods
    st.subheader("🔗 Article URLs")
    input_method = st.radio(
        "Choose input method:",
        ["Enter URLs manually", "Upload text file with URLs"]
    )
    
    urls = []
    
    if input_method == "Enter URLs manually":
        # Manual URL input
        url_input = st.text_area(
            "Enter article URLs (one per line):",
            height=150,
            placeholder="https://example.com/article1\nhttps://example.com/article2\nhttps://example.com/article3"
        )
        
        if url_input:
            urls = [url.strip() for url in url_input.split('\n') if url.strip()]
            
            if urls:
                st.success(f"✅ Found {len(urls)} URLs")
                with st.expander("📋 URLs to process"):
                    for i, url in enumerate(urls, 1):
                        st.write(f"{i}. {url}")
    
    else:
        # File upload
        uploaded_file = st.file_uploader("Choose a text file with URLs", type=['txt'])
        if uploaded_file is not None:
            content = uploaded_file.getvalue().decode("utf-8")
            urls = [url.strip() for url in content.split('\n') if url.strip()]
            
            if urls:
                st.success(f"✅ Found {len(urls)} URLs in file")
                with st.expander("📋 URLs to process"):
                    for i, url in enumerate(urls, 1):
                        st.write(f"{i}. {url}")
    
    # Processing options
    if urls:
        st.subheader("⚙️ Processing Options")
        
        col1, col2 = st.columns(2)
        with col1:
            show_full_text = st.checkbox("Show full article text", value=False)
        with col2:
            auto_scroll = st.checkbox("Auto-scroll to results", value=True)
    
    # Submit button
    if st.button("🚀 Summarize Articles", type="primary"):
        if not urls:
            st.error("❌ Please provide at least one article URL.")
            return
        
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process articles
            results = []
            
            for i, url in enumerate(urls):
                status_text.text(f"Processing article {i+1} of {len(urls)}...")
                progress_bar.progress((i) / len(urls))
                
                st.write(f"📰 **Fetching:** {url}")
                try:
                    with st.spinner("Downloading article..."):
                        text = fetch_article(url)
                    
                    with st.spinner("Generating summary..."):
                        summary = summarize_text(text)
                    
                    results.append({
                        'url': url,
                        'success': True,
                        'summary': summary,
                        'full_text': text
                    })
                    
                    st.success(f"✅ Completed article {i+1}")
                    
                except Exception as e:
                    st.error(f"❌ **Failed for** {url}: {str(e)}")
                    results.append({
                        'url': url,
                        'success': False,
                        'error': str(e)
                    })
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.text("✅ All articles processed!")
            
            # Display results
            st.subheader("📋 Summary Results")
            
            successful_results = [r for r in results if r['success']]
            failed_results = [r for r in results if not r['success']]
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Articles", len(urls))
            with col2:
                st.metric("Successful", len(successful_results))
            with col3:
                st.metric("Failed", len(failed_results))
            
            # Display successful summaries
            for i, result in enumerate(successful_results, 1):
                st.write("---")
                st.subheader(f"📰 Article {i}")
                st.write(f"**URL:** {result['url']}")
                
                # Display summary
                st.write("**Summary:**")
                st.info(result['summary'])
                
                # Show full text if requested
                if show_full_text:
                    with st.expander("📄 Full Article Text"):
                        st.text_area(
                            f"Full text for article {i}:",
                            result['full_text'],
                            height=200,
                            disabled=True,
                            key=f"full_text_{i}"
                        )
            
            # Display failed articles
            if failed_results:
                st.subheader("❌ Failed Articles")
                for result in failed_results:
                    st.error(f"**{result['url']}**: {result['error']}")
            
            # Export option
            if successful_results:
                st.subheader("💾 Export Results")
                
                # Create text digest
                digest_text = "NEWS DIGEST\n" + "="*50 + "\n\n"
                for i, result in enumerate(successful_results, 1):
                    digest_text += f"ARTICLE {i}\n"
                    digest_text += f"URL: {result['url']}\n\n"
                    digest_text += f"SUMMARY:\n{result['summary']}\n\n"
                    digest_text += "-" * 50 + "\n\n"
                
                st.download_button(
                    label="📥 Download News Digest",
                    data=digest_text,
                    file_name="news_digest.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.error("Please check your OpenAI API key and internet connection.")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ How it works")
        st.write("""
        1. **Add URLs**: Enter article URLs manually or upload file
        2. **Fetch Articles**: Uses newspaper3k to extract content
        3. **AI Summarization**: ChatGPT creates concise summaries
        4. **View Results**: Get headline, summary, and "why it matters"
        """)
        
        st.header("📋 Summary Format")
        st.write("""
        Each summary includes:
        - **Headline** (single line)
        - **Summary** (2-3 sentences)
        - **Why it matters** (1 sentence)
        """)
        
        st.header("⚙️ Requirements")
        st.code("pip install streamlit langchain-openai newspaper3k python-dotenv")
        
        st.header("🔑 API Key")
        st.write("Make sure to set your OpenAI API key in a .env file:")
        st.code('OPENAI_API_KEY="sk-..."')
        
        st.header("📰 Supported Sources")
        st.write("""
        - Most news websites
        - Blog articles
        - Online publications
        
        Note: Some sites may block automated access.
        """)
        
        st.header("💡 Tips")
        st.write("""
        - Use direct article URLs (not homepage)
        - Check URLs are accessible
        - Processing time depends on article length
        """)

if __name__ == "__main__":
    main()