from crewai import Agent,LLM
import streamlit as st

api_key="AIzaSyC73jhsmOuRNzlPNL61GikZqUnrpNq9m3w"
llm=LLM(model="gemini-2.5-flash",api_key=api_key,max_tokens=200)

# Question Generator Agent
question_agent = Agent(
    role="Data Science Interviewer",
    goal="Ask relevant interview questions for a Data Scientist role",
    backstory="An experienced senior data scientist who interviews candidates and asks conceptual and technical questions.",
    llm=llm
)

# Feedback Agent
feedback_agent = Agent(
    role="Interview Feedback Evaluator",
    goal="Provide short feedback on the candidate's answers",
    backstory="An expert hiring manager who evaluates answers and provides helpful interview feedback.",
    llm=llm
)


st.title("🤖 Adaptive AI Mock Interviewer (Data Scientist)")

if "history" not in st.session_state:
    st.session_state.history = []

if "current_question" not in st.session_state:
    
    response = question_agent.run(
        "Start the interview with the first Data Scientist interview question."
    )
    
    st.session_state.current_question = response

st.subheader("Interview Question")
st.write(st.session_state.current_question)

answer = st.text_area("Your Answer")

if st.button("Submit Answer"):

    st.session_state.history.append({
        "question": st.session_state.current_question,
        "answer": answer
    })

    feedback_prompt = f"""
    Question: {st.session_state.current_question}

    Candidate Answer:
    {answer}

    Provide short interview feedback.
    """

    st.subheader("Feedback")

    stream = feedback_agent.run(feedback_prompt, stream=True)

    feedback_text = ""

    for chunk in stream:
        feedback_text += chunk
        st.write(feedback_text)

    next_question_prompt = f"""
    Interview History:
    {st.session_state.history}

    Based on the previous answer, ask the next Data Scientist interview question.
    """
    next_q = question_agent.run(next_question_prompt)

    st.session_state.current_question = next_q
    st.rerun()