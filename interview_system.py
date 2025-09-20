import os
import json
import time
import asyncio
import pyaudio
import wave
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
from groq import Groq
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from dotenv import load_dotenv
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set your Groq API key
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 30

@dataclass
class Question:
    id: int
    text: str
    expected_topics: List[str]
    difficulty: str

@dataclass
class Answer:
    question_id: int
    audio_file: str
    transcribed_text: str
    accuracy_score: float
    feedback: str

@dataclass
class InterviewSession:
    topic: str
    level: str
    questions: List[Question]
    answers: List[Answer]
    overall_feedback: str
    timestamp: str

class InterviewState(TypedDict):
    topic: str
    level: str
    questions: List[Dict]
    current_question_index: int
    answers: List[Dict]
    session_complete: bool
    error_message: Optional[str]

class GroqClient:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
    
    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """Generate text using Groq LLM"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Fixed: Using a more stable model
                messages=messages,
                temperature=0.3,  # Reduced for more consistent JSON output
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""
    
    def text_to_speech(self, text: str, filename: str) -> bool:
        """Convert text to speech using Groq TTS"""
        try:
            # Note: Groq doesn't have TTS yet, using a placeholder
            # In production, you'd use a service like ElevenLabs, OpenAI TTS, etc.
            print(f"🎵 TTS: {text}")
            print("(Audio would be generated here - using text output for demo)")
            return True
        except Exception as e:
            print(f"Error in TTS: {e}")
            return False
    
    def speech_to_text(self, audio_file: str) -> str:
        """Convert speech to text using Groq Whisper"""
        try:
            with open(audio_file, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(audio_file, file.read()),
                    model="whisper-large-v3",
                    language="en"
                )
            return transcription.text
        except Exception as e:
            print(f"Error in STT: {e}")
            return ""

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.recording = False
    
    def record_audio(self, filename: str, duration: int = 30) -> bool:
        """Record audio for specified duration"""
        try:
            print(f"🎤 Recording for {duration} seconds... Speak now!")
            
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
            
            frames = []
            for i in range(0, int(RATE / CHUNK_SIZE * duration)):
                data = stream.read(CHUNK_SIZE)
                frames.append(data)
                
                # Show progress
                if i % (RATE // CHUNK_SIZE) == 0:
                    remaining = duration - (i // (RATE // CHUNK_SIZE))
                    print(f"⏱️  {remaining} seconds remaining...")
            
            stream.stop_stream()
            stream.close()
            
            # Save audio file
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            print("✅ Recording completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error recording audio: {e}")
            return False
    
    def cleanup(self):
        self.audio.terminate()

# LangGraph Agents

def question_generator_agent(state: InterviewState) -> InterviewState:
    """Agent to generate interview questions"""
    print("\n🤖 Question Generator Agent: Generating questions...")
    
    groq_client = GroqClient(GROQ_API_KEY)
    
    # Simplified and more explicit prompt for better JSON generation
    system_prompt = """You are an interview question generator. You must respond with ONLY valid JSON in the exact format specified. Do not include any other text, explanations, or markdown formatting.

Generate exactly 3 interview questions as a JSON array. Each question should test different aspects of the topic.

REQUIRED FORMAT (respond with this exact structure):
[
    {
        "id": 1,
        "text": "question text here",
        "expected_topics": ["topic1", "topic2"],
        "difficulty": "beginner"
    },
    {
        "id": 2,
        "text": "question text here", 
        "expected_topics": ["topic1", "topic2"],
        "difficulty": "beginner"
    },
    {
        "id": 3,
        "text": "question text here",
        "expected_topics": ["topic1", "topic2"], 
        "difficulty": "beginner"
    }
]"""
    
    prompt = f"Generate 3 {state['level']} level interview questions about {state['topic']}. Respond with ONLY the JSON array, no other text."
    
    response = groq_client.generate_text(prompt, system_prompt)
    
    # Debug: Print the raw response
    print(f"📋 Raw API Response: {response}")
    
    try:
        # Try to clean up the response if it contains extra text
        response = response.strip()
        
        # Look for JSON array in the response
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = response[start_idx:end_idx]
            questions_data = json.loads(json_str)
            
            # Validate the structure
            if isinstance(questions_data, list) and len(questions_data) > 0:
                # Ensure each question has required fields
                for i, q in enumerate(questions_data):
                    if not isinstance(q, dict):
                        raise ValueError(f"Question {i+1} is not a dictionary")
                    
                    # Fill missing fields with defaults
                    q.setdefault('id', i + 1)
                    q.setdefault('text', f"Sample question {i + 1} about {state['topic']}")
                    q.setdefault('expected_topics', [state['topic']])
                    q.setdefault('difficulty', state['level'])
                
                state["questions"] = questions_data
                state["current_question_index"] = 0
                print(f"✅ Generated {len(questions_data)} questions")
            else:
                raise ValueError("Invalid questions format")
                
    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ JSON parsing error: {e}")
        # Fallback: Create default questions
        print("🔄 Using fallback questions...")
        
        fallback_questions = [
            {
                "id": 1,
                "text": f"What are the fundamental concepts in {state['topic']}?",
                "expected_topics": [state['topic'], "basics", "fundamentals"],
                "difficulty": state['level']
            },
            {
                "id": 2,
                "text": f"Can you explain a practical application of {state['topic']}?",
                "expected_topics": [state['topic'], "applications", "real-world"],
                "difficulty": state['level']
            },
            {
                "id": 3,
                "text": f"What challenges might someone face when learning {state['topic']}?",
                "expected_topics": [state['topic'], "challenges", "learning"],
                "difficulty": state['level']
            }
        ]
        
        state["questions"] = fallback_questions
        state["current_question_index"] = 0
        print(f"✅ Using {len(fallback_questions)} fallback questions")
    
    return state

def question_presenter_agent(state: InterviewState) -> InterviewState:
    """Agent to present questions via TTS"""
    current_idx = state["current_question_index"]
    
    if current_idx >= len(state["questions"]):
        state["session_complete"] = True
        return state
    
    question = state["questions"][current_idx]
    print(f"\n🎯 Question Presenter Agent: Question {current_idx + 1}")
    print(f"📝 {question['text']}")
    
    groq_client = GroqClient(GROQ_API_KEY)
    
    # Convert question to speech (placeholder implementation)
    groq_client.text_to_speech(question["text"], f"question_{current_idx}.wav")
    
    # Wait for user to be ready
    input("\n⏳ Press Enter when ready to record your answer...")
    
    return state

def audio_recorder_agent(state: InterviewState) -> InterviewState:
    """Agent to record user's audio response"""
    current_idx = state["current_question_index"]
    
    print(f"\n🎙️  Audio Recorder Agent: Recording answer {current_idx + 1}")
    
    recorder = AudioRecorder()
    audio_filename = f"answer_{current_idx}_{int(time.time())}.wav"
    
    success = recorder.record_audio(audio_filename, RECORD_SECONDS)
    recorder.cleanup()
    
    if success:
        # Initialize answer entry
        answer = {
            "question_id": current_idx,
            "audio_file": audio_filename,
            "transcribed_text": "",
            "accuracy_score": 0.0,
            "feedback": ""
        }
        
        if "answers" not in state:
            state["answers"] = []
        
        # Ensure we have the right number of answer slots
        while len(state["answers"]) <= current_idx:
            state["answers"].append({})
        
        state["answers"][current_idx] = answer
        print(f"✅ Audio recorded: {audio_filename}")
    else:
        state["error_message"] = f"Failed to record audio for question {current_idx + 1}"
    
    return state

def speech_to_text_agent(state: InterviewState) -> InterviewState:
    """Agent to convert speech to text"""
    current_idx = state["current_question_index"]
    
    print(f"\n🔤 Speech-to-Text Agent: Transcribing answer {current_idx + 1}")
    
    if not state["answers"] or len(state["answers"]) <= current_idx:
        state["error_message"] = "No audio file found for transcription"
        return state
    
    answer = state["answers"][current_idx]
    groq_client = GroqClient(GROQ_API_KEY)
    
    # Check if audio file exists
    if not os.path.exists(answer["audio_file"]):
        # For demo purposes, simulate transcription
        print("🔧 Demo mode: Using simulated transcription")
        simulated_text = f"This is a simulated answer about {state['topic']} at {state['level']} level. The user discussed various concepts and provided examples."
        state["answers"][current_idx]["transcribed_text"] = simulated_text
        print(f"✅ Simulated transcription: {simulated_text[:100]}...")
    else:
        # Transcribe audio
        transcribed_text = groq_client.speech_to_text(answer["audio_file"])
        
        if transcribed_text:
            state["answers"][current_idx]["transcribed_text"] = transcribed_text
            print(f"✅ Transcription: {transcribed_text[:100]}...")
        else:
            state["error_message"] = f"Failed to transcribe audio for question {current_idx + 1}"
    
    return state

def answer_evaluator_agent(state: InterviewState) -> InterviewState:
    """Agent to evaluate answer accuracy"""
    current_idx = state["current_question_index"]
    
    print(f"\n📊 Answer Evaluator Agent: Evaluating answer {current_idx + 1}")
    
    if not state["answers"] or len(state["answers"]) <= current_idx:
        state["error_message"] = "No answer found for evaluation"
        return state
    
    question = state["questions"][current_idx]
    answer = state["answers"][current_idx]
    
    groq_client = GroqClient(GROQ_API_KEY)
    
    # Improved system prompt for more reliable JSON output
    system_prompt = """You are an interview evaluator. Respond with ONLY valid JSON in the exact format specified.

Evaluate the answer and respond with this EXACT format:
{
    "accuracy_score": 85.5,
    "feedback": "Detailed feedback about the answer quality, strengths, and areas for improvement."
}

Do not include any other text, explanations, or formatting."""
    
    prompt = f"""Evaluate this interview answer:

Question: {question['text']}
Expected Topics: {', '.join(question['expected_topics'])}
Difficulty: {question['difficulty']}
Answer: {answer['transcribed_text']}

Provide accuracy score (0-100) and detailed feedback. Respond with ONLY the JSON object."""
    
    evaluation = groq_client.generate_text(prompt, system_prompt)
    
    # Debug: Print the raw evaluation response
    print(f"📋 Evaluation Response: {evaluation}")
    
    try:
        # Clean up the response
        evaluation = evaluation.strip()
        
        # Look for JSON object in the response
        start_idx = evaluation.find('{')
        end_idx = evaluation.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = evaluation[start_idx:end_idx]
            eval_data = json.loads(json_str)
            
            state["answers"][current_idx]["accuracy_score"] = float(eval_data.get("accuracy_score", 70.0))
            state["answers"][current_idx]["feedback"] = eval_data.get("feedback", "Good effort on this question.")
            print(f"✅ Score: {eval_data.get('accuracy_score', 70.0)}/100")
        else:
            raise ValueError("No valid JSON object found")
            
    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ Failed to parse evaluation: {e}")
        # Provide fallback evaluation
        state["answers"][current_idx]["accuracy_score"] = 70.0
        state["answers"][current_idx]["feedback"] = "Answer provided demonstrates understanding of the topic. Consider providing more specific examples and details to improve your response."
        print("🔄 Using fallback evaluation")
    
    # Move to next question
    state["current_question_index"] += 1
    
    return state

def session_summarizer_agent(state: InterviewState) -> InterviewState:
    """Agent to generate final session summary"""
    print(f"\n📋 Session Summarizer Agent: Generating final report...")
    
    groq_client = GroqClient(GROQ_API_KEY)
    
    # Calculate average score
    total_score = sum(answer.get("accuracy_score", 0) for answer in state["answers"])
    avg_score = total_score / len(state["answers"]) if state["answers"] else 0
    
    # Prepare session data for summary
    session_summary = []
    for i, (question, answer) in enumerate(zip(state["questions"], state["answers"])):
        session_summary.append(
            f"Q{i+1}: {question['text']}\n"
            f"Answer: {answer['transcribed_text'][:100]}...\n"
            f"Score: {answer['accuracy_score']}/100\n"
        )
    
    system_prompt = """Provide a comprehensive interview summary. Include:
1. Overall performance assessment
2. Key strengths demonstrated
3. Areas for improvement
4. Specific recommendations
5. Encouragement for continued learning

Be constructive, encouraging, and specific in your feedback."""
    
    prompt = f"""Interview Summary:
Topic: {state['topic']}
Level: {state['level']}
Average Score: {avg_score:.1f}/100

Session Details:
{chr(10).join(session_summary)}

Generate a comprehensive summary and recommendations."""
    
    summary = groq_client.generate_text(prompt, system_prompt)
    
    if not summary:
        summary = f"""Interview Summary for {state['topic']} ({state['level']} level):

You completed {len(state['answers'])} questions with an average score of {avg_score:.1f}/100.

Overall Performance: {"Excellent" if avg_score >= 90 else "Good" if avg_score >= 75 else "Fair" if avg_score >= 60 else "Needs Improvement"}

Strengths: You demonstrated engagement with the topic and provided thoughtful responses.

Areas for Improvement: Continue studying the fundamentals and practice explaining concepts clearly.

Recommendations: 
- Review key concepts in {state['topic']}
- Practice explaining ideas to others
- Seek additional resources for deeper understanding

Keep up the great work and continue learning!"""
    
    state["overall_feedback"] = summary
    
    print("✅ Session summary generated")
    return state

def should_continue(state: InterviewState) -> str:
    """Router function to determine next step"""
    if state.get("error_message"):
        print(f"🚨 Error detected: {state['error_message']}")
        return "error"
    
    if state.get("session_complete"):
        return "summarize"
    
    current_idx = state.get("current_question_index", 0)
    questions = state.get("questions", [])
    answers = state.get("answers", [])
    
    print(f"🔍 Router: Question {current_idx + 1}/{len(questions)}")
    
    if current_idx < len(questions):
        # Check what step we need for current question
        if len(answers) <= current_idx or not answers[current_idx]:
            return "record"
        elif not answers[current_idx].get("transcribed_text"):
            return "transcribe"
        elif not answers[current_idx].get("accuracy_score"):
            return "evaluate"
        else:
            return "next_question"
    
    return "complete"

# Create the workflow graph
def create_interview_workflow():
    workflow = StateGraph(InterviewState)
    
    # Add nodes
    workflow.add_node("generate_questions", question_generator_agent)
    workflow.add_node("present_question", question_presenter_agent)
    workflow.add_node("record_answer", audio_recorder_agent)
    workflow.add_node("transcribe_speech", speech_to_text_agent)
    workflow.add_node("evaluate_answer", answer_evaluator_agent)
    workflow.add_node("summarize_session", session_summarizer_agent)
    
    # Set entry point
    workflow.set_entry_point("generate_questions")
    
    # Add conditional edges with better error handling
    workflow.add_conditional_edges(
        "generate_questions",
        should_continue,
        {
            "record": "present_question",
            "error": END,
            "complete": "summarize_session"
        }
    )
    
    workflow.add_conditional_edges(
        "present_question",
        should_continue,
        {
            "record": "record_answer",
            "complete": "summarize_session",
            "summarize": "summarize_session",
            "error": END
        }
    )
    
    workflow.add_conditional_edges(
        "record_answer",
        should_continue,
        {
            "transcribe": "transcribe_speech",
            "error": END
        }
    )
    
    workflow.add_conditional_edges(
        "transcribe_speech",
        should_continue,
        {
            "evaluate": "evaluate_answer",
            "error": END
        }
    )
    
    workflow.add_conditional_edges(
        "evaluate_answer",
        should_continue,
        {
            "record": "present_question",  # Next question
            "complete": "summarize_session",
            "summarize": "summarize_session",
            "error": END
        }
    )
    
    workflow.add_edge("summarize_session", END)
    
    return workflow.compile()

class InterviewSystem:
    def __init__(self):
        self.workflow = create_interview_workflow()
        self.groq_client = GroqClient(GROQ_API_KEY) if GROQ_API_KEY else None
        
        if not self.groq_client:
            print("⚠️  Warning: GROQ_API_KEY not set. Please set it as environment variable.")
    
    def validate_inputs(self, topic: str, level: str) -> bool:
        """Validate user inputs"""
        valid_levels = ["beginner", "intermediate", "advanced", "expert"]
        
        if not topic.strip():
            print("❌ Topic cannot be empty")
            return False
        
        if level.lower() not in valid_levels:
            print(f"❌ Level must be one of: {', '.join(valid_levels)}")
            return False
        
        return True
    
    def get_user_inputs(self) -> tuple:
        """Get interview topic and level from user"""
        print("=" * 50)
        print("🎯 AI INTERVIEW SYSTEM")
        print("=" * 50)
        
        topic = input("\n📚 Enter the interview topic: ").strip()
        print("\n📊 Available levels:")
        print("   • beginner")
        print("   • intermediate") 
        print("   • advanced")
        print("   • expert")
        
        level = input("\n🎚️  Enter the difficulty level: ").strip().lower()
        
        return topic, level
    
    def display_final_report(self, state: InterviewState):
        """Display comprehensive interview report"""
        print("\n" + "=" * 60)
        print("📊 INTERVIEW REPORT")
        print("=" * 60)
        
        print(f"\n📚 Topic: {state['topic']}")
        print(f"🎚️  Level: {state['level'].title()}")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n📝 QUESTIONS & ANSWERS:")
        print("-" * 40)
        
        total_score = 0
        for i, (question, answer) in enumerate(zip(state["questions"], state["answers"])):
            print(f"\n❓ Question {i+1}: {question['text']}")
            print(f"💬 Your Answer: {answer['transcribed_text'][:200]}...")
            print(f"⭐ Score: {answer['accuracy_score']}/100")
            print(f"💡 Feedback: {answer['feedback'][:150]}...")
            total_score += answer['accuracy_score']
        
        avg_score = total_score / len(state["answers"]) if state["answers"] else 0
        
        print(f"\n📈 OVERALL PERFORMANCE:")
        print("-" * 40)
        print(f"🎯 Average Score: {avg_score:.1f}/100")
        
        # Performance rating
        if avg_score >= 90:
            rating = "🏆 Excellent"
        elif avg_score >= 75:
            rating = "🥇 Good"
        elif avg_score >= 60:
            rating = "🥈 Fair"
        else:
            rating = "🥉 Needs Improvement"
        
        print(f"🏅 Rating: {rating}")
        
        print(f"\n🎓 DETAILED FEEDBACK:")
        print("-" * 40)
        print(state["overall_feedback"])
        
        # Save session to file
        session = InterviewSession(
            topic=state["topic"],
            level=state["level"],
            questions=[Question(**q) for q in state["questions"]],
            answers=[Answer(**a) for a in state["answers"]],
            overall_feedback=state["overall_feedback"],
            timestamp=datetime.now().isoformat()
        )
        
        filename = f"interview_session_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(asdict(session), f, indent=2, default=str)
        
        print(f"\n💾 Session saved to: {filename}")
    
    async def run_interview(self):
        """Run the complete interview workflow"""
        if not self.groq_client:
            print("❌ Cannot run interview without Groq API key")
            return
        
        try:
            # Get user inputs
            topic, level = self.get_user_inputs()
            
            if not self.validate_inputs(topic, level):
                return
            
            # Initialize state
            initial_state = InterviewState(
                topic=topic,
                level=level,
                questions=[],
                current_question_index=0,
                answers=[],
                session_complete=False,
                error_message=None
            )
            
            print(f"\n🚀 Starting interview on '{topic}' at {level} level...")
            print("🤖 AI agents will guide you through the process")
            
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            if final_state.get("error_message"):
                print(f"\n❌ Interview failed: {final_state['error_message']}")
                return
            
            # Display final report
            self.display_final_report(final_state)
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Interview interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run the interview system"""
    print("🤖 Initializing AI Interview System...")
    
    # Check dependencies
    try:
        import pyaudio
        import wave
        from groq import Groq
        from langgraph.graph import StateGraph, END
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install groq pyaudio langgraph-core python-dotenv")
        return
    
    # Check API key
    if not GROQ_API_KEY:
        print("❌ Please set your GROQ_API_KEY environment variable")
        print("Example: export GROQ_API_KEY='your_api_key_here'")
        print("Or create a .env file with: GROQ_API_KEY=your_api_key_here")
        return
    
    interview_system = InterviewSystem()
    
    # Run the interview
    asyncio.run(interview_system.run_interview())

if __name__ == "__main__":
    main()