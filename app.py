# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from typing import List, Optional, Dict, Any
# from fastapi.middleware.cors import CORSMiddleware
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema import HumanMessage, SystemMessage
# import json
# import uuid
# from datetime import datetime
# import logging
# from dotenv import load_dotenv
# import os
# load_dotenv() 

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # FastAPI app initialization
# app = FastAPI(
#     title="Agentic AI Interview System",
#     description="AI-powered interview system with question generation and answer evaluation",
#     version="1.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize the LLM (replace "llm_model" with your actual model)
# llm = ChatGroq(
#     model="openai/gpt-oss-120b",
#     temperature=0.7,
#     max_tokens=2000
# )

# # Pydantic models for request/response
# class InterviewRequest(BaseModel):
#     topic: str = Field(..., description="Interview topic (e.g., 'Python Programming', 'Data Science', 'Machine Learning')")
#     difficulty: str = Field(..., description="Difficulty level: 'beginner', 'intermediate', 'advanced'")
#     num_questions: int = Field(..., ge=1, le=20, description="Number of questions (1-20)")

# class Question(BaseModel):
#     id: str
#     question: str
#     expected_points: List[str]
#     difficulty: str

# class QuestionsResponse(BaseModel):
#     session_id: str
#     topic: str
#     difficulty: str
#     questions: List[Question]
#     created_at: str

# class AnswerSubmission(BaseModel):
#     session_id: str
#     question_id: str
#     answer: str

# class AnswersEvaluation(BaseModel):
#     session_id: str
#     answers: List[AnswerSubmission]

# class QuestionEvaluation(BaseModel):
#     question_id: str
#     question: str
#     answer: str
#     score: float
#     max_score: float
#     feedback: str
#     strengths: List[str]
#     improvements: List[str]

# class InterviewResults(BaseModel):
#     session_id: str
#     topic: str
#     difficulty: str
#     overall_score: float
#     max_possible_score: float
#     percentage: float
#     grade: str
#     question_evaluations: List[QuestionEvaluation]
#     overall_feedback: str
#     recommendations: List[str]
#     strengths: List[str]
#     areas_for_improvement: List[str]

# # In-memory storage for interview sessions
# interview_sessions: Dict[str, Dict] = {}

# class InterviewAgent:
#     """Agentic AI system for conducting interviews"""
    
#     def __init__(self, llm):
#         self.llm = llm
    
#     def generate_questions(self, topic: str, difficulty: str, num_questions: int) -> List[Question]:
#         """Generate interview questions based on topic and difficulty"""
        
#         difficulty_descriptions = {
#             "beginner": "Basic concepts, fundamental understanding, simple applications",
#             "intermediate": "Practical applications, problem-solving, connecting concepts",
#             "advanced": "Complex scenarios, optimization, system design, expert-level thinking"
#         }
        
#         prompt = ChatPromptTemplate.from_template("""
# You are an expert interviewer creating {difficulty} level questions about {topic}.

# Generate exactly {num_questions} interview questions that:
# 1. Are appropriate for {difficulty} level ({difficulty_desc})
# 2. Cover different aspects of {topic}
# 3. Allow for comprehensive evaluation of knowledge
# 4. Are clear and specific

# For each question, also provide 3-5 key points that a good answer should cover.

# Return the response in the following JSON format:
# {{
#   "questions": [
#     {{
#       "question": "The actual question text",
#       "expected_points": ["point1", "point2", "point3"]
#     }}
#   ]
# }}

# Topic: {topic}
# Difficulty: {difficulty}
# Number of questions: {num_questions}
# """)
        
#         try:
#             response = self.llm.invoke(
#                 prompt.format(
#                     topic=topic,
#                     difficulty=difficulty,
#                     num_questions=num_questions,
#                     difficulty_desc=difficulty_descriptions[difficulty.lower()]
#                 )
#             )
            
#             # Parse the JSON response
#             content = response.content.strip()
#             if content.startswith('```json'):
#                 content = content[7:-3]
#             elif content.startswith('```'):
#                 content = content[3:-3]
            
#             parsed_response = json.loads(content)
#             questions = []
            
#             for i, q_data in enumerate(parsed_response.get("questions", [])):
#                 questions.append(Question(
#                     id=str(uuid.uuid4()),
#                     question=q_data["question"],
#                     expected_points=q_data["expected_points"],
#                     difficulty=difficulty
#                 ))
            
#             return questions
        
#         except Exception as e:
#             logger.error(f"Error generating questions: {str(e)}")
#             raise HTTPException(status_code=500, detail=f"Failed to generate questions: {str(e)}")
    
#     def evaluate_answer(self, question: str, answer: str, expected_points: List[str], difficulty: str) -> Dict[str, Any]:
#         """Evaluate a single answer"""
        
#         scoring_criteria = {
#             "beginner": "Focus on basic understanding and correct fundamental concepts",
#             "intermediate": "Look for practical application, problem-solving approach, and connection between concepts",
#             "advanced": "Evaluate depth of understanding, optimization thinking, system-level considerations, and expert insights"
#         }
        
#         prompt = ChatPromptTemplate.from_template("""
# You are an expert interviewer evaluating an answer to a {difficulty} level question about programming/technical topics.

# Question: {question}
# Expected key points: {expected_points}
# Candidate's Answer: {answer}

# Evaluation Criteria for {difficulty} level: {criteria}

# Please evaluate this answer and provide:
# 1. A score out of 10
# 2. Detailed feedback on the answer
# 3. Specific strengths demonstrated
# 4. Areas for improvement
# 5. Whether each expected point was addressed

# Return your evaluation in the following JSON format:
# {{
#   "score": 8.5,
#   "max_score": 10,
#   "feedback": "Detailed feedback about the answer quality, accuracy, and completeness",
#   "strengths": ["strength1", "strength2"],
#   "improvements": ["improvement1", "improvement2"],
#   "points_covered": ["point1", "point2"],
#   "points_missed": ["point3"]
# }}

# Be fair but thorough in your evaluation. Consider accuracy, completeness, clarity, and demonstration of understanding.
# """)
        
#         try:
#             response = self.llm.invoke(
#                 prompt.format(
#                     question=question,
#                     answer=answer,
#                     expected_points=expected_points,
#                     difficulty=difficulty,
#                     criteria=scoring_criteria[difficulty.lower()]
#                 )
#             )
            
#             content = response.content.strip()
#             if content.startswith('```json'):
#                 content = content[7:-3]
#             elif content.startswith('```'):
#                 content = content[3:-3]
            
#             return json.loads(content)
        
#         except Exception as e:
#             logger.error(f"Error evaluating answer: {str(e)}")
#             return {
#                 "score": 5.0,
#                 "max_score": 10,
#                 "feedback": "Error occurred during evaluation",
#                 "strengths": [],
#                 "improvements": ["Please try again"],
#                 "points_covered": [],
#                 "points_missed": expected_points
#             }
    
#     def generate_overall_feedback(self, topic: str, difficulty: str, evaluations: List[Dict]) -> Dict[str, Any]:
#         """Generate overall interview feedback and recommendations"""
        
#         scores = [eval_data["score"] for eval_data in evaluations]
#         overall_score = sum(scores)
#         max_possible = len(scores) * 10
#         percentage = (overall_score / max_possible) * 100
        
#         # Determine grade
#         if percentage >= 90:
#             grade = "A"
#         elif percentage >= 80:
#             grade = "B"
#         elif percentage >= 70:
#             grade = "C"
#         elif percentage >= 60:
#             grade = "D"
#         else:
#             grade = "F"
        
#         # Collect all strengths and improvements
#         all_strengths = []
#         all_improvements = []
        
#         for eval_data in evaluations:
#             all_strengths.extend(eval_data.get("strengths", []))
#             all_improvements.extend(eval_data.get("improvements", []))
        
#         # Remove duplicates while preserving order
#         unique_strengths = list(dict.fromkeys(all_strengths))
#         unique_improvements = list(dict.fromkeys(all_improvements))
        
#         prompt = ChatPromptTemplate.from_template("""
# Based on an interview assessment for {topic} at {difficulty} level:

# Overall Score: {overall_score}/{max_possible} ({percentage:.1f}%)
# Grade: {grade}

# Key Strengths Demonstrated: {strengths}
# Areas for Improvement: {improvements}

# Please provide:
# 1. Overall feedback summary
# 2. 3-5 specific recommendations for improvement
# 3. Learning path suggestions

# Return in JSON format:
# {{
#   "overall_feedback": "Comprehensive summary of performance",
#   "recommendations": ["recommendation1", "recommendation2", "recommendation3"]
# }}
# """)
        
#         try:
#             response = self.llm.invoke(
#                 prompt.format(
#                     topic=topic,
#                     difficulty=difficulty,
#                     overall_score=overall_score,
#                     max_possible=max_possible,
#                     percentage=percentage,
#                     grade=grade,
#                     strengths=unique_strengths,
#                     improvements=unique_improvements
#                 )
#             )
            
#             content = response.content.strip()
#             if content.startswith('```json'):
#                 content = content[7:-3]
#             elif content.startswith('```'):
#                 content = content[3:-3]
            
#             feedback_data = json.loads(content)
            
#             return {
#                 "overall_feedback": feedback_data["overall_feedback"],
#                 "recommendations": feedback_data["recommendations"],
#                 "strengths": unique_strengths,
#                 "areas_for_improvement": unique_improvements
#             }
        
#         except Exception as e:
#             logger.error(f"Error generating overall feedback: {str(e)}")
#             return {
#                 "overall_feedback": f"Interview completed with {percentage:.1f}% score. Review individual question feedback for details.",
#                 "recommendations": ["Review fundamental concepts", "Practice more problems", "Improve explanation clarity"],
#                 "strengths": unique_strengths,
#                 "areas_for_improvement": unique_improvements
#             }

# # Initialize the interview agent
# interview_agent = InterviewAgent(llm)

# @app.get("/")
# async def root():
#     return {
#         "message": "Agentic AI Interview System API",
#         "version": "1.0.0",
#         "endpoints": {
#             "generate_questions": "/generate-questions",
#             "evaluate_interview": "/evaluate-interview"
#         }
#     }

# @app.post("/generate-questions", response_model=QuestionsResponse)
# async def generate_questions(request: InterviewRequest):
#     """Generate interview questions based on topic, difficulty, and number"""
    
#     try:
#         # Validate difficulty level
#         valid_difficulties = ["beginner", "intermediate", "advanced"]
#         if request.difficulty.lower() not in valid_difficulties:
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Invalid difficulty level. Must be one of: {valid_difficulties}"
#             )
        
#         # Generate questions using the agent
#         questions = interview_agent.generate_questions(
#             topic=request.topic,
#             difficulty=request.difficulty.lower(),
#             num_questions=request.num_questions
#         )
        
#         # Create session
#         session_id = str(uuid.uuid4())
#         session_data = {
#             "topic": request.topic,
#             "difficulty": request.difficulty.lower(),
#             "questions": {q.id: q.dict() for q in questions},
#             "created_at": datetime.now().isoformat(),
#             "status": "active"
#         }
        
#         interview_sessions[session_id] = session_data
        
#         logger.info(f"Generated {len(questions)} questions for session {session_id}")
        
#         return QuestionsResponse(
#             session_id=session_id,
#             topic=request.topic,
#             difficulty=request.difficulty,
#             questions=questions,
#             created_at=session_data["created_at"]
#         )
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error in generate_questions: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.post("/evaluate-interview", response_model=InterviewResults)
# async def evaluate_interview(evaluation_request: AnswersEvaluation):
#     """Evaluate all answers and provide comprehensive feedback"""
    
#     try:
#         session_id = evaluation_request.session_id
        
#         # Validate session exists
#         if session_id not in interview_sessions:
#             raise HTTPException(status_code=404, detail="Interview session not found")
        
#         session_data = interview_sessions[session_id]
        
#         if session_data["status"] != "active":
#             raise HTTPException(status_code=400, detail="Interview session is not active")
        
#         # Evaluate each answer
#         question_evaluations = []
#         evaluation_data = []
        
#         for answer_submission in evaluation_request.answers:
#             question_id = answer_submission.question_id
            
#             if question_id not in session_data["questions"]:
#                 logger.warning(f"Question ID {question_id} not found in session {session_id}")
#                 continue
            
#             question_data = session_data["questions"][question_id]
            
#             # Evaluate the answer
#             eval_result = interview_agent.evaluate_answer(
#                 question=question_data["question"],
#                 answer=answer_submission.answer,
#                 expected_points=question_data["expected_points"],
#                 difficulty=question_data["difficulty"]
#             )
            
#             evaluation_data.append(eval_result)
            
#             question_evaluations.append(QuestionEvaluation(
#                 question_id=question_id,
#                 question=question_data["question"],
#                 answer=answer_submission.answer,
#                 score=eval_result["score"],
#                 max_score=eval_result["max_score"],
#                 feedback=eval_result["feedback"],
#                 strengths=eval_result["strengths"],
#                 improvements=eval_result["improvements"]
#             ))
        
#         if not evaluation_data:
#             raise HTTPException(status_code=400, detail="No valid answers to evaluate")
        
#         # Calculate overall results
#         overall_score = sum(eval_data["score"] for eval_data in evaluation_data)
#         max_possible_score = len(evaluation_data) * 10
#         percentage = (overall_score / max_possible_score) * 100
        
#         # Determine grade
#         if percentage >= 90:
#             grade = "A"
#         elif percentage >= 80:
#             grade = "B"
#         elif percentage >= 70:
#             grade = "C"
#         elif percentage >= 60:
#             grade = "D"
#         else:
#             grade = "F"
        
#         # Generate overall feedback
#         overall_feedback_data = interview_agent.generate_overall_feedback(
#             topic=session_data["topic"],
#             difficulty=session_data["difficulty"],
#             evaluations=evaluation_data
#         )
        
#         # Mark session as completed
#         session_data["status"] = "completed"
#         session_data["completed_at"] = datetime.now().isoformat()
        
#         results = InterviewResults(
#             session_id=session_id,
#             topic=session_data["topic"],
#             difficulty=session_data["difficulty"],
#             overall_score=overall_score,
#             max_possible_score=max_possible_score,
#             percentage=percentage,
#             grade=grade,
#             question_evaluations=question_evaluations,
#             overall_feedback=overall_feedback_data["overall_feedback"],
#             recommendations=overall_feedback_data["recommendations"],
#             strengths=overall_feedback_data["strengths"],
#             areas_for_improvement=overall_feedback_data["areas_for_improvement"]
#         )
        
#         logger.info(f"Completed evaluation for session {session_id}. Score: {percentage:.1f}%")
        
#         return results
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error in evaluate_interview: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.get("/session/{session_id}")
# async def get_session_info(session_id: str):
#     """Get information about a specific interview session"""
    
#     if session_id not in interview_sessions:
#         raise HTTPException(status_code=404, detail="Interview session not found")
    
#     session_data = interview_sessions[session_id]
    
#     return {
#         "session_id": session_id,
#         "topic": session_data["topic"],
#         "difficulty": session_data["difficulty"],
#         "status": session_data["status"],
#         "created_at": session_data["created_at"],
#         "num_questions": len(session_data["questions"])
#     }

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "active_sessions": len([s for s in interview_sessions.values() if s["status"] == "active"])
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import json
import uuid
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Agentic AI Interview System",
    description="AI-powered interview system with question generation and answer evaluation",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "mixtral-8x7b-32768")  # Default model
PORT = int(os.getenv("PORT", 8000))

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable is required")
    raise ValueError("GROQ_API_KEY environment variable is required")

# Initialize the LLM
try:
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.7,
        max_tokens=2000
    )
    logger.info(f"Initialized LLM with model: {LLM_MODEL}")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {str(e)}")
    raise

# Pydantic models for request/response
class InterviewRequest(BaseModel):
    topic: str = Field(..., description="Interview topic (e.g., 'Python Programming', 'Data Science', 'Machine Learning')")
    difficulty: str = Field(..., description="Difficulty level: 'beginner', 'intermediate', 'advanced'")
    num_questions: int = Field(..., ge=1, le=20, description="Number of questions (1-20)")

class Question(BaseModel):
    id: str
    question: str
    expected_points: List[str]
    difficulty: str

class QuestionsResponse(BaseModel):
    session_id: str
    topic: str
    difficulty: str
    questions: List[Question]
    created_at: str

class AnswerSubmission(BaseModel):
    session_id: str
    question_id: str
    answer: str

class AnswersEvaluation(BaseModel):
    session_id: str
    answers: List[AnswerSubmission]

class QuestionEvaluation(BaseModel):
    question_id: str
    question: str
    answer: str
    score: float
    max_score: float
    feedback: str
    strengths: List[str]
    improvements: List[str]

class InterviewResults(BaseModel):
    session_id: str
    topic: str
    difficulty: str
    overall_score: float
    max_possible_score: float
    percentage: float
    grade: str
    question_evaluations: List[QuestionEvaluation]
    overall_feedback: str
    recommendations: List[str]
    strengths: List[str]
    areas_for_improvement: List[str]

# In-memory storage for interview sessions
interview_sessions: Dict[str, Dict] = {}

class InterviewAgent:
    """Agentic AI system for conducting interviews"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_questions(self, topic: str, difficulty: str, num_questions: int) -> List[Question]:
        """Generate interview questions based on topic and difficulty"""
        
        difficulty_descriptions = {
            "beginner": "Basic concepts, fundamental understanding, simple applications",
            "intermediate": "Practical applications, problem-solving, connecting concepts",
            "advanced": "Complex scenarios, optimization, system design, expert-level thinking"
        }
        
        prompt = ChatPromptTemplate.from_template("""
You are an expert interviewer creating {difficulty} level questions about {topic}.

Generate exactly {num_questions} interview questions that:
1. Are appropriate for {difficulty} level ({difficulty_desc})
2. Cover different aspects of {topic}
3. Allow for comprehensive evaluation of knowledge
4. Are clear and specific

For each question, also provide 3-5 key points that a good answer should cover.

Return the response in the following JSON format:
{{
  "questions": [
    {{
      "question": "The actual question text",
      "expected_points": ["point1", "point2", "point3"]
    }}
  ]
}}

Topic: {topic}
Difficulty: {difficulty}
Number of questions: {num_questions}
""")
        
        try:
            response = self.llm.invoke(
                prompt.format(
                    topic=topic,
                    difficulty=difficulty,
                    num_questions=num_questions,
                    difficulty_desc=difficulty_descriptions[difficulty.lower()]
                )
            )
            
            # Parse the JSON response
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
            
            parsed_response = json.loads(content)
            questions = []
            
            for i, q_data in enumerate(parsed_response.get("questions", [])):
                questions.append(Question(
                    id=str(uuid.uuid4()),
                    question=q_data["question"],
                    expected_points=q_data["expected_points"],
                    difficulty=difficulty
                ))
            
            return questions
        
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate questions: {str(e)}")
    
    def evaluate_answer(self, question: str, answer: str, expected_points: List[str], difficulty: str) -> Dict[str, Any]:
        """Evaluate a single answer"""
        
        scoring_criteria = {
            "beginner": "Focus on basic understanding and correct fundamental concepts",
            "intermediate": "Look for practical application, problem-solving approach, and connection between concepts",
            "advanced": "Evaluate depth of understanding, optimization thinking, system-level considerations, and expert insights"
        }
        
        prompt = ChatPromptTemplate.from_template("""
You are an expert interviewer evaluating an answer to a {difficulty} level question about programming/technical topics.

Question: {question}
Expected key points: {expected_points}
Candidate's Answer: {answer}

Evaluation Criteria for {difficulty} level: {criteria}

Please evaluate this answer and provide:
1. A score out of 10
2. Detailed feedback on the answer
3. Specific strengths demonstrated
4. Areas for improvement
5. Whether each expected point was addressed

Return your evaluation in the following JSON format:
{{
  "score": 8.5,
  "max_score": 10,
  "feedback": "Detailed feedback about the answer quality, accuracy, and completeness",
  "strengths": ["strength1", "strength2"],
  "improvements": ["improvement1", "improvement2"],
  "points_covered": ["point1", "point2"],
  "points_missed": ["point3"]
}}

Be fair but thorough in your evaluation. Consider accuracy, completeness, clarity, and demonstration of understanding.
""")
        
        try:
            response = self.llm.invoke(
                prompt.format(
                    question=question,
                    answer=answer,
                    expected_points=expected_points,
                    difficulty=difficulty,
                    criteria=scoring_criteria[difficulty.lower()]
                )
            )
            
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
            
            return json.loads(content)
        
        except Exception as e:
            logger.error(f"Error evaluating answer: {str(e)}")
            return {
                "score": 5.0,
                "max_score": 10,
                "feedback": "Error occurred during evaluation",
                "strengths": [],
                "improvements": ["Please try again"],
                "points_covered": [],
                "points_missed": expected_points
            }
    
    def generate_overall_feedback(self, topic: str, difficulty: str, evaluations: List[Dict]) -> Dict[str, Any]:
        """Generate overall interview feedback and recommendations"""
        
        scores = [eval_data["score"] for eval_data in evaluations]
        overall_score = sum(scores)
        max_possible = len(scores) * 10
        percentage = (overall_score / max_possible) * 100
        
        # Determine grade
        if percentage >= 90:
            grade = "A"
        elif percentage >= 80:
            grade = "B"
        elif percentage >= 70:
            grade = "C"
        elif percentage >= 60:
            grade = "D"
        else:
            grade = "F"
        
        # Collect all strengths and improvements
        all_strengths = []
        all_improvements = []
        
        for eval_data in evaluations:
            all_strengths.extend(eval_data.get("strengths", []))
            all_improvements.extend(eval_data.get("improvements", []))
        
        # Remove duplicates while preserving order
        unique_strengths = list(dict.fromkeys(all_strengths))
        unique_improvements = list(dict.fromkeys(all_improvements))
        
        prompt = ChatPromptTemplate.from_template("""
Based on an interview assessment for {topic} at {difficulty} level:

Overall Score: {overall_score}/{max_possible} ({percentage:.1f}%)
Grade: {grade}

Key Strengths Demonstrated: {strengths}
Areas for Improvement: {improvements}

Please provide:
1. Overall feedback summary
2. 3-5 specific recommendations for improvement
3. Learning path suggestions

Return in JSON format:
{{
  "overall_feedback": "Comprehensive summary of performance",
  "recommendations": ["recommendation1", "recommendation2", "recommendation3"]
}}
""")
        
        try:
            response = self.llm.invoke(
                prompt.format(
                    topic=topic,
                    difficulty=difficulty,
                    overall_score=overall_score,
                    max_possible=max_possible,
                    percentage=percentage,
                    grade=grade,
                    strengths=unique_strengths,
                    improvements=unique_improvements
                )
            )
            
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
            
            feedback_data = json.loads(content)
            
            return {
                "overall_feedback": feedback_data["overall_feedback"],
                "recommendations": feedback_data["recommendations"],
                "strengths": unique_strengths,
                "areas_for_improvement": unique_improvements
            }
        
        except Exception as e:
            logger.error(f"Error generating overall feedback: {str(e)}")
            return {
                "overall_feedback": f"Interview completed with {percentage:.1f}% score. Review individual question feedback for details.",
                "recommendations": ["Review fundamental concepts", "Practice more problems", "Improve explanation clarity"],
                "strengths": unique_strengths,
                "areas_for_improvement": unique_improvements
            }

# Initialize the interview agent
interview_agent = InterviewAgent(llm)

@app.get("/")
async def root():
    return {
        "message": "Agentic AI Interview System API",
        "version": "1.0.0",
        "endpoints": {
            "generate_questions": "/generate-questions",
            "evaluate_interview": "/evaluate-interview"
        }
    }

@app.post("/generate-questions", response_model=QuestionsResponse)
async def generate_questions(request: InterviewRequest):
    """Generate interview questions based on topic, difficulty, and number"""
    
    try:
        # Validate difficulty level
        valid_difficulties = ["beginner", "intermediate", "advanced"]
        if request.difficulty.lower() not in valid_difficulties:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid difficulty level. Must be one of: {valid_difficulties}"
            )
        
        # Generate questions using the agent
        questions = interview_agent.generate_questions(
            topic=request.topic,
            difficulty=request.difficulty.lower(),
            num_questions=request.num_questions
        )
        
        # Create session
        session_id = str(uuid.uuid4())
        session_data = {
            "topic": request.topic,
            "difficulty": request.difficulty.lower(),
            "questions": {q.id: q.dict() for q in questions},
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        interview_sessions[session_id] = session_data
        
        logger.info(f"Generated {len(questions)} questions for session {session_id}")
        
        return QuestionsResponse(
            session_id=session_id,
            topic=request.topic,
            difficulty=request.difficulty,
            questions=questions,
            created_at=session_data["created_at"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate_questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/evaluate-interview", response_model=InterviewResults)
async def evaluate_interview(evaluation_request: AnswersEvaluation):
    """Evaluate all answers and provide comprehensive feedback"""
    
    try:
        session_id = evaluation_request.session_id
        
        # Validate session exists
        if session_id not in interview_sessions:
            raise HTTPException(status_code=404, detail="Interview session not found")
        
        session_data = interview_sessions[session_id]
        
        if session_data["status"] != "active":
            raise HTTPException(status_code=400, detail="Interview session is not active")
        
        # Evaluate each answer
        question_evaluations = []
        evaluation_data = []
        
        for answer_submission in evaluation_request.answers:
            question_id = answer_submission.question_id
            
            if question_id not in session_data["questions"]:
                logger.warning(f"Question ID {question_id} not found in session {session_id}")
                continue
            
            question_data = session_data["questions"][question_id]
            
            # Evaluate the answer
            eval_result = interview_agent.evaluate_answer(
                question=question_data["question"],
                answer=answer_submission.answer,
                expected_points=question_data["expected_points"],
                difficulty=question_data["difficulty"]
            )
            
            evaluation_data.append(eval_result)
            
            question_evaluations.append(QuestionEvaluation(
                question_id=question_id,
                question=question_data["question"],
                answer=answer_submission.answer,
                score=eval_result["score"],
                max_score=eval_result["max_score"],
                feedback=eval_result["feedback"],
                strengths=eval_result["strengths"],
                improvements=eval_result["improvements"]
            ))
        
        if not evaluation_data:
            raise HTTPException(status_code=400, detail="No valid answers to evaluate")
        
        # Calculate overall results
        overall_score = sum(eval_data["score"] for eval_data in evaluation_data)
        max_possible_score = len(evaluation_data) * 10
        percentage = (overall_score / max_possible_score) * 100
        
        # Determine grade
        if percentage >= 90:
            grade = "A"
        elif percentage >= 80:
            grade = "B"
        elif percentage >= 70:
            grade = "C"
        elif percentage >= 60:
            grade = "D"
        else:
            grade = "F"
        
        # Generate overall feedback
        overall_feedback_data = interview_agent.generate_overall_feedback(
            topic=session_data["topic"],
            difficulty=session_data["difficulty"],
            evaluations=evaluation_data
        )
        
        # Mark session as completed
        session_data["status"] = "completed"
        session_data["completed_at"] = datetime.now().isoformat()
        
        results = InterviewResults(
            session_id=session_id,
            topic=session_data["topic"],
            difficulty=session_data["difficulty"],
            overall_score=overall_score,
            max_possible_score=max_possible_score,
            percentage=percentage,
            grade=grade,
            question_evaluations=question_evaluations,
            overall_feedback=overall_feedback_data["overall_feedback"],
            recommendations=overall_feedback_data["recommendations"],
            strengths=overall_feedback_data["strengths"],
            areas_for_improvement=overall_feedback_data["areas_for_improvement"]
        )
        
        logger.info(f"Completed evaluation for session {session_id}. Score: {percentage:.1f}%")
        
        return results
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in evaluate_interview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific interview session"""
    
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Interview session not found")
    
    session_data = interview_sessions[session_id]
    
    return {
        "session_id": session_id,
        "topic": session_data["topic"],
        "difficulty": session_data["difficulty"],
        "status": session_data["status"],
        "created_at": session_data["created_at"],
        "num_questions": len(session_data["questions"])
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len([s for s in interview_sessions.values() if s["status"] == "active"])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)