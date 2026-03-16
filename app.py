from crewai import Agent,LLM,Task,Crew
import streamlit as st

api_key=st.secrets["GOOGLE_API_KEY"]
llm=LLM(model="gemini-2.5-flash",api_key=api_key,max_tokens=200)

# Question Generator Agent
question_agent = Agent(
    role="Data Science Interviewer",
    goal="Ask relevant interview questions for a Data Scientist role",
    backstory="An experienced senior data scientist who interviews candidates and asks conceptual and technical questions.",
    llm=llm,
    max_iteration=1
)

# Feedback Agent
feedback_agent = Agent(
    role="Interview Feedback Evaluator",
    goal="Provide short feedback on the candidate's answers",
    backstory="An expert hiring manager who evaluates answers and provides helpful interview feedback.",
    llm=llm,
    max_iteration=1
)

question_task = Task(
    description="""You are conducting a Data Scientist interview.

Conversation so far:
{history}

Based on the previous answer, generate the NEXT interview question.
Ask only ONE question.""",
    agent=question_agent,
    expected_output="A list of 1 interview questions."
)

feedback_task = Task(
    description="""
You are evaluating a candidate's interview answer.

Conversation:
{history}

Give short feedback on the candidate's latest answer.
Be concise.
""",
    agent=feedback_agent,
    expected_output="Short interview feedback."
)

question_crew=Crew(agents=[question_agent],task=[question_task])
feedback_crew=Crew(agents=[feedback_agent],tasks=[feedback_task])


st.title("🤖 Adaptive AI Mock Interviewer (Data Scientist)")

if "history" not in st.session_state:
    st.session_state.history = []

if "current_question" not in st.session_state:
    
    response = question_crew.kickoff(inputs=
        {"history":"Start the interview with the first Data Scientist interview question."}
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

    stream = feedback_crew.kickoff(inputs={"history":feedback_prompt})

    feedback_text = ""

    for chunk in stream:
        feedback_text += chunk
        st.write(feedback_text)

    next_question_prompt = f"""
    Interview History:
    {st.session_state.history}

    Based on the previous answer, ask the next Data Scientist interview question.
    """
    next_q = question_crew.kickoff(inputs={"history":next_question_prompt})

    st.session_state.current_question = next_q
    st.rerun()