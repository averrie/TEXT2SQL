import streamlit as st
import time

from spider_agent.agent.agents import PromptAgent
from spider_agent.envs.spider_agent import Spider_Agent_Env

# Configure Streamlit page
st.set_page_config(page_title="PromptAgent Demo", layout="wide")

st.title("Query with PromptAgent")

# --- 1) Collect Model/Agent Parameters from the user
with st.sidebar:
    st.header("Model Settings")
    model = st.selectbox("Model:", ["gemma-3-27b-it"])
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.5, 0.1)
    top_p = st.slider("Top P:", 0.0, 1.0, 0.9, 0.05)
    max_tokens = st.number_input(
        "Max Tokens:", min_value=100, max_value=4000, value=1000, step=50
    )
    max_steps = st.number_input(
        "Max Steps (agent planning):", min_value=1, max_value=50, value=20
    )
    max_memory_length = st.number_input(
        "Max Memory Length:", min_value=1, max_value=100, value=25
    )

# --- 2) Collect the user question
question = st.text_area("Enter your natural language question here:")

# Initialize session state for the step-by-step execution
if "step_state" not in st.session_state:
    st.session_state.step_state = {
        "initialized": False,
        "agent": None,
        "env": None,
        "step_idx": 0,
        "done": False,
        "obs": "You are in the folder now.",
        "steps": [],
        "retry_count": 0,
        "last_action": None,
        "repeat_action": False,
        "running": False,
        "result": None,
        "result_files": None,
        "error": None,
    }

# --- 3) Provide a button to run
start_button = st.button(
    "Submit Query", disabled=st.session_state.step_state["running"]
)

if start_button and question.strip():
    # Reset the state
    state = st.session_state.step_state
    state["initialized"] = False
    state["agent"] = None
    state["env"] = None
    state["step_idx"] = 0
    state["done"] = False
    state["obs"] = "You are in the folder now."
    state["steps"] = []
    state["retry_count"] = 0
    state["last_action"] = None
    state["repeat_action"] = False
    state["running"] = True
    state["result"] = None
    state["result_files"] = None
    state["error"] = None

# Container for logs and steps
logs_container = st.empty()
steps_container = st.container()
result_container = st.empty()

# Main execution loop - runs one step at a time when the state is running
state = st.session_state.step_state

if state["running"]:
    try:
        # Initialize the agent and environment if not already done
        if not state["initialized"]:
            # --- 3.1) Initialize the Agent
            state["agent"] = PromptAgent(
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_memory_length=max_memory_length,
                max_steps=max_steps,
                use_plan=False,
            )

            # --- 3.2) Construct the Task Config
            task_config = {
                "instance_id": f"local-quickstart-{int(time.time())}",
                "type": "Local",
                "question": question,
                "config": [
                    {
                        "type": "copy_all_subfiles",
                        "parameters": {"dirs": ["./local_db"]},
                    }
                ],
            }

            # --- 3.3) Construct environment config
            env_config = {
                "image_name": "spider_agent-image",
                "init_args": {
                    "name": f"streamlit_demo_{int(time.time())}",
                    "work_dir": "/workspace",
                },
            }

            # Where data is cached or saved
            cache_dir = "./cache"
            output_dir = "./output"

            # Create the environment
            state["env"] = Spider_Agent_Env(
                env_config=env_config,
                task_config=task_config,
                cache_dir=cache_dir,
                mnt_dir=output_dir,
            )

            # Set up the agent with the environment
            state["agent"].set_env_and_task(state["env"])
            state["initialized"] = True
            st.rerun()  # Refresh after initialization

        # Execute one step if not done
        if not state["done"] and state["step_idx"] < max_steps:
            # Get the agent's response and action
            response, action = state["agent"].predict(state["obs"])

            # Record step information
            step_info = {
                "step": state["step_idx"] + 1,
                "thought": (
                    state["agent"].thoughts[-1] if state["agent"].thoughts else ""
                ),
                "action": str(action) if action else "No valid action detected",
            }

            if action is None:
                state["retry_count"] += 1
                if state["retry_count"] > 3:
                    state["done"] = True
                    state["error"] = (
                        "Failed to parse action from response after multiple attempts."
                    )
                else:
                    state["obs"] = (
                        "Failed to parse action from your response, make sure you provide a valid action."
                    )
                    step_info["observation"] = state["obs"]
            else:
                if state["last_action"] is not None and str(
                    state["last_action"]
                ) == str(action):
                    if state["repeat_action"]:
                        state["error"] = "ERROR: Repeated action"
                        state["done"] = True
                    else:
                        state["obs"] = (
                            "The action is the same as the last one, you MUST provide a DIFFERENT SQL code or Python Code or different action."
                        )
                        state["repeat_action"] = True
                        step_info["observation"] = state["obs"]
                else:
                    # Execute the action and get observation
                    state["obs"], state["done"] = state["env"].step(action)
                    state["last_action"] = action
                    state["repeat_action"] = False
                    step_info["observation"] = state["obs"]

                    # Check if task is complete
                    if state["done"] and hasattr(action, "output"):
                        state["result"] = action.output

            # Add the step info to our steps list
            state["steps"].append(step_info)
            state["step_idx"] += 1

        # If we're done or hit max steps, finalize
        if state["done"] or state["step_idx"] >= max_steps:
            if not state["done"]:
                state["error"] = (
                    "Reached maximum number of steps without completing the task."
                )

            # Get result files if not already done
            if state["result_files"] is None:
                try:
                    state["result_files"] = state["env"].post_process()
                    # Clean up environment
                    state["env"].close()
                except Exception as e:
                    state["error"] = f"Error during post-processing: {str(e)}"

            # Mark as no longer running
            state["running"] = False

        # Display steps
        if state["steps"]:
            with steps_container:
                st.subheader("Agent Steps")
                for i, step in enumerate(state["steps"]):
                    with st.expander(
                        f"Step {step['step']}", expanded=(i == len(state["steps"]) - 1)
                    ):
                        st.markdown("### Thought")
                        st.text(step["thought"])

                        st.markdown("### Action")
                        st.code(step["action"])

                        if "observation" in step:
                            st.markdown("### Observation")
                            st.text(step["observation"])

        # Display results
        if state["result"]:
            with result_container:
                st.subheader("Final Result")
                st.write(state["result"])

                if state["result_files"]:
                    st.subheader("Result Files")
                    st.json(state["result_files"])

        # Display any errors
        if state["error"]:
            st.error(state["error"])
            state["running"] = False

        # Rerun if still processing
        if state["running"]:
            time.sleep(0.5)  # Small delay
            st.rerun()

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        st.error(f"Error: {str(e)}\n\n{error_details}")
        state["running"] = False
        state["error"] = f"Error: {str(e)}"
