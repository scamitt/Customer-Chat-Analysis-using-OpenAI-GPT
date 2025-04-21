import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
import os
import tempfile
from dotenv import load_dotenv

load_dotenv() 

api_key = os.getenv("OPENAI_API_KEY") 

# Set up the app
st.set_page_config(page_title="Chat Analytics Dashboard", layout="wide")
st.title("💬 Chat Analytics Dashboard")

# Initialize LLM and embeddings
@st.cache_resource
def load_models():
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        openai_api_key=api_key
    )
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key
    )
    return llm, embeddings

llm, embeddings = load_models()

# Create prompt templates
@st.cache_data
def create_prompt_templates(categories=None):
    sentiment_template = ChatPromptTemplate.from_messages([
        ("system", """Analyze the sentiment of this chat message. 
         Respond ONLY with one word: 'Positive', 'Negative', or 'Neutral'.
         Positive = happy, satisfied, pleased
         Negative = angry, frustrated, unhappy
         Neutral = factual, neither positive nor negative"""),
        ("human", "{text}")
    ])
    
    #
    return {
        "sentiment": sentiment_template,
    }

# Example usage
prompt_templates = create_prompt_templates()  # The user will be prompted to input categories if none are provided


# Analysis functions
def analyze_sentiment(text):
    chain = LLMChain(llm=llm, prompt=prompt_templates["sentiment"])
    result = chain.run(text=text)
    return result.strip()

def analyze_category(text):
    chain = LLMChain(llm=llm, prompt=prompt_templates["category"])
    result = chain.run(text=text)
    return result.strip()

# FAQ Vector Store
@st.cache_resource
def create_faq_vectorstore(df):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    loader = DataFrameLoader(df, page_content_column="chat")
    documents = loader.load_and_split(text_splitter)
    return FAISS.from_documents(documents, embeddings)

# Page 1: File Upload
def page_upload():
    st.header("Upload Your Chat Data")
    st.write("Please upload a CSV file with columns: 'id' and 'chat'")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = ['id','chat']
            if not all(col in df.columns for col in ['id', 'chat']):
                st.error("CSV must contain 'id' and 'chat' columns")
                return None
            
            # Cache the uploaded data
            st.session_state.df = df
            st.success("File uploaded successfully!")
            st.dataframe(df.head())
            
            # Add navigation button
            if st.button("Go to Analysis",key="run_analysis_btn"):
                st.session_state.page = "analysis"
                st.rerun()
                
            return df
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    return None

# Page 2: Analysis Dashboard
def page_analysis():
    if 'df' not in st.session_state:
        st.warning("Please upload data first")
        st.session_state.page = "upload"
        st.rerun()
        return
    
    df = st.session_state.df
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "Category Classification", "Knowledge Base"])
    
    with tab1:
        st.header("Sentiment Analysis")
        
        if st.button("Run Sentiment Analysis",key="run_sent_analysis_btn"):
            with st.spinner("Analyzing sentiment (this may take a few minutes)..."):
                # Analyze a sample if dataset is large
                sample_size = min(100, len(df))
                sample_df = df.sample(sample_size).copy()
                sample_df['sentiment'] = sample_df['chat'].apply(analyze_sentiment)
                st.session_state.sentiment_df = sample_df
            
        if 'sentiment_df' in st.session_state:
            sentiment_df = st.session_state.sentiment_df
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", f"{len(sentiment_df[sentiment_df['sentiment'] == 'Positive'])} ({len(sentiment_df[sentiment_df['sentiment'] == 'Positive'])/len(sentiment_df)*100:.1f}%)")
            col2.metric("Negative", f"{len(sentiment_df[sentiment_df['sentiment'] == 'Negative'])} ({len(sentiment_df[sentiment_df['sentiment'] == 'Negative'])/len(sentiment_df)*100:.1f}%)")
            col3.metric("Neutral", f"{len(sentiment_df[sentiment_df['sentiment'] == 'Neutral'])} ({len(sentiment_df[sentiment_df['sentiment'] == 'Neutral'])/len(sentiment_df)*100:.1f}%)")
            
            # Visualization
            fig = px.histogram(sentiment_df, x='sentiment', 
                              title='Sentiment Distribution',
                              color='sentiment',
                              color_discrete_map={
                                  'Positive': 'green',
                                  'Negative': 'red',
                                  'Neutral': 'blue'
                              })
            st.plotly_chart(fig, use_container_width=True)
            
            # Show sample data
            st.subheader("Sample Data with Sentiment")
            st.dataframe(sentiment_df[['id', 'chat', 'sentiment']].head(10))
        else:
            st.info("Click the button above to run sentiment analysis")
    
    with tab2:
        st.header("Category Classification")
        
        # User-defined categories input
        st.subheader("Define Your Categories")
        default_categories = ["Billing", "Bug", "Request", "General"]
        user_categories = st.text_area(
            "Enter your custom categories (comma separated):",
            value=", ".join(default_categories),
            help="For example: Sales, Support, Technical, Billing, Other"
        )
        
        # Process user categories
        categories = [cat.strip() for cat in user_categories.split(",") if cat.strip()]
        
        if not categories:
            st.warning("Please enter at least one category")
            st.stop()
        
        if st.button("Run Category Classification with Custom Categories",key="run_cat_btn"):
            with st.spinner("Classifying chats (this may take a few minutes)..."):
                # Create dynamic prompt template with user categories
                category_prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""Classify this chat message into one of these categories:
                    {', '.join(categories)}
                    Respond ONLY with the category name that best matches. 
                    If none match well, choose the most similar one."""),
                    ("human", "{text}")
                ])
                
                # Analyze a sample if dataset is large
                sample_size = min(100, len(df))
                sample_df = df.sample(sample_size).copy()
                
                # Create chain with dynamic prompt
                category_chain = LLMChain(llm=llm, prompt=category_prompt)
                sample_df['category'] = sample_df['chat'].apply(
                    lambda x: category_chain.run(text=x).strip()
                )
                
                st.session_state.category_df = sample_df
                st.session_state.user_categories = categories
        
        if 'category_df' in st.session_state and 'user_categories' in st.session_state:
            category_df = st.session_state.category_df
            user_categories = st.session_state.user_categories
            
            
            
            # Metrics
            st.subheader("Category Distribution")
            category_counts = category_df['category'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            
            # Visualization - using bar chart instead of pie for better readability
            fig = px.bar(category_counts, 
                        x='Category', 
                        y='Count',
                        title='Chat Categories Distribution',
                        color='Category')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show distribution metrics
            st.subheader("Category Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Classified Chats", len(category_df))
            with col2:
                st.metric("Unique Categories Found", category_df['category'].nunique())
            
            # Show sample data with ability to filter by category
            st.subheader("Sample Data with Categories")
            selected_category = st.selectbox(
                "Filter by category:",
                options=["All"] + sorted(category_df['category'].unique())
            )
            
            display_df = category_df[['id', 'chat', 'category']]
            if selected_category != "All":
                display_df = display_df[display_df['category'] == selected_category]
            
            st.dataframe(display_df.head(20))
            
            # Download button for classified data
            st.download_button(
                label="Download Classified Data",
                data=category_df.to_csv(index=False).encode('utf-8'),
                file_name='classified_chats.csv',
                mime='text/csv'
            )
        else:
            st.info("Enter your categories above and click the button to run classification")
    
    with tab3:
        st.header("Knowledge Base")
        st.write("Ask questions about the content in your chat data")
        
        # Create vector store if not exists
        if 'faiss_store' not in st.session_state:
            with st.spinner("Creating knowledge base..."):
                st.session_state.faiss_store = create_faq_vectorstore(df)
        
        # Question answering
        user_question = st.text_input("Ask a question about your chat data:")
        
        if user_question:
            faiss_store = st.session_state.faiss_store
            docs = faiss_store.similarity_search(user_question, k=3)
            
            # Use LLM to generate answer from context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Answer the user's question based on the following context.
                 If you don't know the answer, say you don't know. Be concise.
                 
                 Context: {context}"""),
                ("human", "Question: {question}")
            ])
            
            chain = LLMChain(llm=llm, prompt=prompt)
            answer = chain.run(context=context, question=user_question)
            
            st.subheader("Answer")
            st.write(answer)
            
            with st.expander("See relevant chat excerpts"):
                for i, doc in enumerate(docs, 1):
                    st.write(f"**Excerpt {i}:**")
                    st.write(doc.page_content)
                    st.write("---")

# Main app flow
if 'page' not in st.session_state:
    st.session_state.page = "upload"

if st.session_state.page == "upload":
    page_upload()
elif st.session_state.page == "analysis":
    page_analysis()

# Add navigation in sidebar
st.sidebar.title("Navigation")
if st.session_state.page == "upload" and 'df' in st.session_state:
    if st.sidebar.button("Go to Analysis",key="go_to_analysis_btn"):
        st.session_state.page = "analysis"
        st.rerun()
elif st.session_state.page == "analysis":
    if st.sidebar.button("Back to Upload",key="back_to_analysis_btn"):
        st.session_state.page = "upload"
        st.rerun()
