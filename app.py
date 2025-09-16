import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.prompts import PromptTemplate

# Load API key
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Prompt Template
PROMPT_TEMPLATE = """
You are an expert resume parser and job match evaluator. 
Given the resume text and job description, do two tasks:

---

### 1. Extract Resume Information
Present the resume details in a **professional Markdown layout** with sections:

### 👤 Personal Details
- **Name:** ...
- **Email:** ...
- **Phone:** ...
- **LinkedIn:** ...

### 🛠 Skills
- Skill1, Skill2, Skill3

### 🎓 Education
- Degree | Institute | Year

### 💼 Experience
- Role at Company (Duration)
- Short bullet point achievements

### 📂 Projects
- Project Name — short description

### 🏅 Certifications
- Certification Name — Issuer

### 🌐 Languages
- Language1, Language2

---

### 2. Compare Resume Skills with Job Description Skills
If job description is provided, add a section:

### 🔍 Resume vs Job Description Skills
- ✅ **Matching Skills:** list of skills found in both
- ❌ **Missing Skills (JD requires but resume lacks):** list of missing skills
- 💡 **Extra Skills (in resume but not in JD):** list of extra skills
- 📊 **Match Score:** percentage of JD skills that appear in resume

Rules:
- If no JD is provided, skip the comparison.
- Use bullet points for clarity.
- Keep it short, clean, and professional.

---

Resume text:
{text}

Job description:
{jd_text}
"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["text", "jd_text"])


def load_resume_docs(uploaded_file):
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(temp_path)
    elif uploaded_file.name.endswith(".txt"):
        loader = TextLoader(temp_path)
    else:
        return None
    return loader.load()


def clean_text_block(text: str) -> str:
    """Remove markdown headers and unnecessary symbols, keep plain text."""
    text = text.replace("###", "").replace("##", "").strip()
    # Remove any starting/ending dash or bullet
    lines = [line.strip(" -•") for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def main():
    st.set_page_config(page_title="Resume Parser", page_icon="📄", layout="centered")
    st.title("📄 CVision - Resume Parser & JD Comparison")

    uploaded_file = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])

    if uploaded_file:
        with st.spinner("Loading resume..."):
            docs = load_resume_docs(uploaded_file)
            if not docs:
                st.error("Unsupported file type.")
                return

        # Expander for preview text
        with st.expander("📖 Resume Extracted Text Preview"):
            preview_text = "\n\n".join([d.page_content for d in docs])[:4000]
            st.text_area("Preview (first 4000 chars)", value=preview_text, height=200)

        # Expander for JD input
        with st.expander("📝 Paste Job Description (Optional)"):
            jd_text = st.text_area("Job Description", placeholder="Paste the job description here...")

        # Button
        if st.button("🚀 Analyze Resume"):
            with st.spinner("Sending to LLM..."):
                full_text = "\n\n".join([d.page_content for d in docs])
                formatted_prompt = prompt.format(
                    text=full_text,
                    jd_text=jd_text if jd_text else "No JD provided"
                )

                response = llm.invoke(formatted_prompt)

                # Split Resume Info & Comparison
                content = response.content
                if "### 🔍 Resume vs Job Description Skills" in content:
                    resume_info, comparison = content.split("### 🔍 Resume vs Job Description Skills", 1)
                else:
                    resume_info, comparison = content, None

                # Display Resume Info
                st.markdown("## 📌 Extracted Resume Information")
                st.markdown(resume_info, unsafe_allow_html=True)

                # Display Comparison if JD is provided
                if comparison:
                    st.markdown("## 🔍 Resume vs Job Description Skills")

                    # Matching Skills
                    if "✅" in comparison:
                        matched_section = comparison.split("✅")[1].split("❌")[0]
                        st.subheader("✅ Matching Skills")
                        st.success(clean_text_block(matched_section))

                    # Missing Skills
                    if "❌" in comparison:
                        missing_section = comparison.split("❌")[1].split("💡")[0]
                        st.subheader("❌ Missing Skills")
                        st.error(clean_text_block(missing_section))

                    # Extra Skills
                    if "💡" in comparison:
                        extra_section = comparison.split("💡")[1].split("📊")[0]
                        st.subheader("💡 Extra Skills")
                        st.warning(clean_text_block(extra_section))

                    # Match Score
                    if "📊" in comparison:
                        score_section = comparison.split("📊")[1]
                        st.subheader("📊 Match Score")
                        st.info(clean_text_block(score_section))


if __name__ == "__main__":
    main()
